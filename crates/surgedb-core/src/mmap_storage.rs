//! Memory-mapped vector storage
//!
//! Provides disk-resident vector storage using memory mapping (mmap).
//! The OS handles paging vectors in/out of memory automatically,
//! allowing datasets much larger than available RAM.
//!
//! ## Design
//! - Vectors stored in a flat file (contiguous f32 values)
//! - Memory-mapped for zero-copy access
//! - ID mappings stored in a separate index file
//! - Append-only for simplicity and crash safety

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::types::{InternalId, VectorId};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Memory-mapped vector storage
pub struct MmapStorage {
    /// Vector dimensions
    dimensions: usize,

    /// Data directory
    dir: PathBuf,

    /// Vector data file
    data_file: RwLock<File>,

    /// Memory-mapped region (read-only view of data)
    mmap: RwLock<Option<Mmap>>,

    /// Number of vectors stored
    count: RwLock<usize>,

    /// Map from external ID to internal ID
    id_to_internal: RwLock<HashMap<VectorId, InternalId>>,

    /// Map from internal ID to external ID
    internal_to_id: RwLock<Vec<VectorId>>,

    /// Current file size in bytes
    file_size: RwLock<u64>,
}

/// Simple mmap wrapper
struct Mmap {
    ptr: *mut u8,
    len: usize,
}

// Safety: Mmap is Send+Sync because we only read from it
// and the underlying file is protected by RwLock
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    #[cfg(unix)]
    fn new(file: &File, len: usize) -> Result<Self> {
        use std::os::unix::io::AsRawFd;

        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(Error::Storage("mmap failed".into()));
        }

        // Advise the kernel that we'll access randomly
        unsafe {
            libc::madvise(ptr, len, libc::MADV_RANDOM);
        }

        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    #[cfg(not(unix))]
    fn new(_file: &File, len: usize) -> Result<Self> {
        // Fallback for non-Unix platforms
        Ok(Self {
            ptr: std::ptr::null_mut(),
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() || self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

/// Header for the data file
const DATA_MAGIC: &[u8; 4] = b"ZVEC";
const DATA_VERSION: u8 = 1;
const HEADER_SIZE: usize = 16; // magic(4) + version(1) + dimensions(4) + reserved(7)

impl MmapStorage {
    /// Create or open mmap storage at the given path
    pub fn open(dir: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let data_path = dir.join("vectors.dat");
        let index_path = dir.join("index.dat");

        let (data_file, file_size, existing_dims) = if data_path.exists() {
            // Open existing file
            let mut file = OpenOptions::new().read(true).write(true).open(&data_path)?;

            // Read and verify header
            let mut header = [0u8; HEADER_SIZE];
            file.read_exact(&mut header)?;

            if &header[0..4] != DATA_MAGIC {
                return Err(Error::Storage("Invalid vector data file".into()));
            }
            if header[4] != DATA_VERSION {
                return Err(Error::Storage(format!(
                    "Unsupported version: {}",
                    header[4]
                )));
            }

            let existing_dims =
                u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;
            let file_size = file.metadata()?.len();

            (file, file_size, Some(existing_dims))
        } else {
            // Create new file with header
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&data_path)?;

            let mut header = [0u8; HEADER_SIZE];
            header[0..4].copy_from_slice(DATA_MAGIC);
            header[4] = DATA_VERSION;
            header[5..9].copy_from_slice(&(dimensions as u32).to_le_bytes());

            file.write_all(&header)?;
            file.sync_all()?;

            (file, HEADER_SIZE as u64, None)
        };

        // Verify dimensions match
        if let Some(existing) = existing_dims {
            if existing != dimensions {
                return Err(Error::InvalidConfig(format!(
                    "Dimension mismatch: file has {}, requested {}",
                    existing, dimensions
                )));
            }
        }

        // Load ID mappings
        let (id_to_internal, internal_to_id) = Self::load_index(&index_path)?;
        let count = internal_to_id.len();

        // Create mmap
        let mmap = if file_size > HEADER_SIZE as u64 {
            Some(Mmap::new(&data_file, file_size as usize)?)
        } else {
            None
        };

        Ok(Self {
            dimensions,
            dir,
            data_file: RwLock::new(data_file),
            mmap: RwLock::new(mmap),
            count: RwLock::new(count),
            id_to_internal: RwLock::new(id_to_internal),
            internal_to_id: RwLock::new(internal_to_id),
            file_size: RwLock::new(file_size),
        })
    }

    /// Load ID index from file
    fn load_index(path: &Path) -> Result<(HashMap<VectorId, InternalId>, Vec<VectorId>)> {
        if !path.exists() {
            return Ok((HashMap::new(), Vec::new()));
        }

        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }

        let ids: Vec<String> =
            bincode::deserialize(&data).map_err(|e| Error::Storage(e.to_string()))?;

        let mut id_to_internal = HashMap::with_capacity(ids.len());
        let mut internal_to_id = Vec::with_capacity(ids.len());

        for (i, id_str) in ids.into_iter().enumerate() {
            let id = VectorId::from(id_str);
            let internal = InternalId::from(i);
            id_to_internal.insert(id.clone(), internal);
            internal_to_id.push(id);
        }

        Ok((id_to_internal, internal_to_id))
    }

    /// Save ID index to file
    fn save_index(&self) -> Result<()> {
        let index_path = self.dir.join("index.dat");
        let internal_to_id = self.internal_to_id.read();

        let ids: Vec<String> = internal_to_id
            .iter()
            .map(|id| id.as_str().to_string())
            .collect();

        let data = bincode::serialize(&ids).map_err(|e| Error::Storage(e.to_string()))?;

        let mut file = File::create(&index_path)?;
        file.write_all(&data)?;
        file.sync_all()?;

        Ok(())
    }

    /// Insert a vector and return its internal ID
    pub fn insert(&self, id: VectorId, vector: &[f32]) -> Result<InternalId> {
        self.insert_internal(id, vector, false)
    }

    /// Insert or update a vector
    pub fn upsert(&self, id: VectorId, vector: &[f32]) -> Result<InternalId> {
        self.insert_internal(id, vector, true)
    }

    fn insert_internal(
        &self,
        id: VectorId,
        vector: &[f32],
        allow_update: bool,
    ) -> Result<InternalId> {
        if vector.len() != self.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        // Check for duplicate
        {
            let id_to_internal = self.id_to_internal.read();
            if !allow_update && id_to_internal.contains_key(&id) {
                return Err(Error::DuplicateId(id.to_string()));
            }
        }

        // Append vector to file
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        let internal_id;
        {
            let mut data_file = self.data_file.write();
            let mut count = self.count.write();
            let mut file_size = self.file_size.write();
            let mut id_to_internal = self.id_to_internal.write();
            let mut internal_to_id = self.internal_to_id.write();

            // Seek to end and write
            data_file.seek(SeekFrom::End(0))?;
            data_file.write_all(&vector_bytes)?;
            data_file.sync_data()?;

            internal_id = InternalId::from(*count);

            // Update mappings
            id_to_internal.insert(id.clone(), internal_id);
            internal_to_id.push(id);
            *count += 1;
            *file_size += vector_bytes.len() as u64;
        }

        // Remap the file
        self.remap()?;

        // Save index periodically (every 100 inserts)
        if internal_id.as_usize() % 100 == 0 {
            self.save_index()?;
        }

        Ok(internal_id)
    }

    /// Remap the file after it grows
    fn remap(&self) -> Result<()> {
        let data_file = self.data_file.read();
        let file_size = *self.file_size.read();

        if file_size > HEADER_SIZE as u64 {
            let new_mmap = Mmap::new(&data_file, file_size as usize)?;
            let mut mmap = self.mmap.write();
            *mmap = Some(new_mmap);
        }

        Ok(())
    }

    /// Get a vector by internal ID (zero-copy when possible)
    #[inline]
    pub fn get(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        let mmap = self.mmap.read();
        let mmap = mmap.as_ref()?;

        let bytes = mmap.as_slice();
        if bytes.is_empty() {
            return None;
        }

        let vector_size = self.dimensions * 4; // f32 = 4 bytes
        let offset = HEADER_SIZE + internal_id.as_usize() * vector_size;
        let end = offset + vector_size;

        if end > bytes.len() {
            return None;
        }

        let vector_bytes = &bytes[offset..end];
        let vector: Vec<f32> = vector_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Some(vector)
    }

    /// Get vector data for distance calculation
    #[inline]
    pub fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        self.get(internal_id)
    }

    /// Get external ID from internal ID
    pub fn get_external_id(&self, internal_id: InternalId) -> Option<VectorId> {
        let internal_to_id = self.internal_to_id.read();
        internal_to_id.get(internal_id.as_usize()).cloned()
    }

    /// Get internal ID from external ID
    pub fn get_internal_id(&self, id: &VectorId) -> Option<InternalId> {
        let id_to_internal = self.id_to_internal.read();
        id_to_internal.get(id).copied()
    }

    /// Get all internal IDs
    pub fn all_internal_ids(&self) -> Vec<InternalId> {
        let count = *self.count.read();
        (0..count).map(InternalId::from).collect()
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        *self.count.read()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Sync all data to disk
    pub fn sync(&self) -> Result<()> {
        self.data_file.read().sync_all()?;
        self.save_index()?;
        Ok(())
    }

    /// Get disk usage in bytes
    pub fn disk_usage(&self) -> u64 {
        *self.file_size.read()
    }

    /// Create a view of the storage that holds a read lock
    /// This is optimized for bulk operations like search
    pub fn view(&self) -> MmapStorageView<'_> {
        MmapStorageView {
            guard: self.mmap.read(),
            dimensions: self.dimensions,
        }
    }
}

/// A view into MmapStorage that holds a read lock on the data
pub struct MmapStorageView<'a> {
    guard: parking_lot::RwLockReadGuard<'a, Option<Mmap>>,
    dimensions: usize,
}

impl<'a> crate::storage::VectorStorageTrait for MmapStorageView<'a> {
    fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        let mmap = self.guard.as_ref()?;
        let bytes = mmap.as_slice();

        if bytes.is_empty() {
            return None;
        }

        let vector_size = self.dimensions * 4;
        let offset = HEADER_SIZE + internal_id.as_usize() * vector_size;
        let end = offset + vector_size;

        if end > bytes.len() {
            return None;
        }

        let vector_bytes = &bytes[offset..end];
        let vector: Vec<f32> = vector_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Some(vector)
    }

    fn distance(
        &self,
        internal_id: InternalId,
        query: &[f32],
        metric: DistanceMetric,
    ) -> Option<f32> {
        let mmap = self.guard.as_ref()?;
        let bytes = mmap.as_slice();

        if bytes.is_empty() {
            return None;
        }

        let vector_size = self.dimensions * 4;
        let offset = HEADER_SIZE + internal_id.as_usize() * vector_size;
        let end = offset + vector_size;

        if end > bytes.len() {
            return None;
        }

        let vector_bytes = &bytes[offset..end];

        let (prefix, vector_slice, suffix) = unsafe { vector_bytes.align_to::<f32>() };

        if !prefix.is_empty() || !suffix.is_empty() {
            let vector: Vec<f32> = vector_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Some(metric.distance(query, &vector))
        } else {
            Some(metric.distance(query, vector_slice))
        }
    }
}

/// Implement VectorStorageTrait for MmapStorage
impl crate::storage::VectorStorageTrait for MmapStorage {
    fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        self.get(internal_id)
    }

    fn distance(
        &self,
        internal_id: InternalId,
        query: &[f32],
        metric: DistanceMetric,
    ) -> Option<f32> {
        let mmap = self.mmap.read();
        let mmap = mmap.as_ref()?;

        let bytes = mmap.as_slice();
        if bytes.is_empty() {
            return None;
        }

        let vector_size = self.dimensions * 4; // f32 = 4 bytes
        let offset = HEADER_SIZE + internal_id.as_usize() * vector_size;
        let end = offset + vector_size;

        if end > bytes.len() {
            return None;
        }

        let vector_bytes = &bytes[offset..end];

        // This is safe because f32 alignment is usually 4, and mmap gives aligned pages.
        // However, to be strictly safe and avoid unaligned access on some archs,
        // we might need `align_to`, but for performance we assume alignment or handle it.
        // For now, we will perform a safe cast if aligned, or fallback to copy if not?
        // Actually, SIMD functions in distance.rs usually handle unaligned loads (vld1q_f32 on NEON handles unaligned).
        // But we need a &[f32] slice. casting &[u8] to &[f32] is unsafe if not aligned.

        let (prefix, vector_slice, suffix) = unsafe { vector_bytes.align_to::<f32>() };

        if !prefix.is_empty() || !suffix.is_empty() {
            // Unaligned access fallback (should be rare with mmap)
            // Or if the file offset isn't aligned (HEADER_SIZE=16 is aligned to 4/8/16)
            let vector: Vec<f32> = vector_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Some(metric.distance(query, &vector))
        } else {
            Some(metric.distance(query, vector_slice))
        }
    }
}

impl Drop for MmapStorage {
    fn drop(&mut self) {
        // Save index on drop
        let _ = self.save_index();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_insert_and_get() {
        let dir = tempdir().unwrap();
        let storage = MmapStorage::open(dir.path(), 4).unwrap();

        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let id = storage.insert("test".into(), &vector).unwrap();

        let retrieved = storage.get(id).unwrap();
        assert_eq!(retrieved, vector);
    }

    #[test]
    fn test_mmap_multiple_vectors() {
        let dir = tempdir().unwrap();
        let storage = MmapStorage::open(dir.path(), 4).unwrap();

        for i in 0..100 {
            let vector: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
            storage.insert(format!("v{}", i).into(), &vector).unwrap();
        }

        assert_eq!(storage.len(), 100);

        // Verify a few vectors
        let v50 = storage.get(InternalId::from(50)).unwrap();
        assert_eq!(v50, vec![200.0, 201.0, 202.0, 203.0]);
    }

    #[test]
    fn test_mmap_persistence() {
        let dir = tempdir().unwrap();

        // Create and populate
        {
            let storage = MmapStorage::open(dir.path(), 4).unwrap();
            for i in 0..10 {
                let vector: Vec<f32> = (0..4).map(|j| (i + j) as f32).collect();
                storage.insert(format!("v{}", i).into(), &vector).unwrap();
            }
            storage.sync().unwrap();
        }

        // Reopen and verify
        {
            let storage = MmapStorage::open(dir.path(), 4).unwrap();
            assert_eq!(storage.len(), 10);

            let v5 = storage.get(InternalId::from(5)).unwrap();
            assert_eq!(v5, vec![5.0, 6.0, 7.0, 8.0]);

            let ext_id = storage.get_external_id(InternalId::from(5)).unwrap();
            assert_eq!(ext_id.as_str(), "v5");
        }
    }

    #[test]
    fn test_mmap_large_vectors() {
        let dir = tempdir().unwrap();
        let dims = 384; // Realistic embedding size
        let storage = MmapStorage::open(dir.path(), dims).unwrap();

        for i in 0..1000 {
            let vector: Vec<f32> = (0..dims).map(|j| ((i * j) as f32).sin()).collect();
            storage.insert(format!("v{}", i).into(), &vector).unwrap();
        }

        storage.sync().unwrap();

        assert_eq!(storage.len(), 1000);

        // Verify disk usage is reasonable
        let expected_size = HEADER_SIZE + 1000 * dims * 4;
        assert!(storage.disk_usage() >= expected_size as u64);
    }
}
