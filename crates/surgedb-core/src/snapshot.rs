//! Snapshot (checkpoint) management for fast recovery
//!
//! Snapshots contain the complete database state at a point in time.
//! Combined with WAL, they enable fast recovery without replaying the entire history.

use crate::error::{Error, Result};
use crate::hnsw::HnswState;
use crate::types::VectorId;
use bincode::{deserialize_from, serialize_into};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

/// Magic bytes for snapshot files
const SNAPSHOT_MAGIC: &[u8; 4] = b"ZSNP";

/// Snapshot format version
const SNAPSHOT_VERSION: u8 = 2;

/// Stored vector data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredVector {
    pub id: VectorId,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

/// Complete database snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Snapshot ID (timestamp or sequence)
    pub id: u64,
    /// WAL sequence number at snapshot time
    pub wal_seq: u64,
    /// Database dimensions
    pub dimensions: usize,
    /// All vectors
    pub vectors: Vec<StoredVector>,
    /// HNSW index state
    pub hnsw_state: Option<HnswState>,
}

impl Snapshot {
    /// Create a new snapshot
    pub fn new(id: u64, wal_seq: u64, dimensions: usize) -> Self {
        Self {
            id,
            wal_seq,
            dimensions,
            vectors: Vec::new(),
            hnsw_state: None,
        }
    }

    /// Add a vector to the snapshot
    pub fn add_vector(&mut self, id: VectorId, vector: Vec<f32>, metadata: Option<Value>) {
        self.vectors.push(StoredVector {
            id,
            vector,
            metadata,
        });
    }

    /// Set the index state
    pub fn set_hnsw_state(&mut self, state: HnswState) {
        self.hnsw_state = Some(state);
    }

    /// Get the number of vectors
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Snapshot file header
#[derive(Debug, Serialize, Deserialize)]
struct SnapshotHeader {
    magic: [u8; 4],
    version: u8,
    id: u64,
    wal_seq: u64,
    dimensions: usize,
    vector_count: usize,
}

/// Snapshot manager
pub struct SnapshotManager {
    dir: PathBuf,
    /// Keep this many snapshots
    retain_count: usize,
}

impl SnapshotManager {
    /// Create a new snapshot manager
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        Ok(Self {
            dir,
            retain_count: 3,
        })
    }

    /// Set how many snapshots to retain
    pub fn set_retain_count(&mut self, count: usize) {
        self.retain_count = count.max(1);
    }

    /// Save a snapshot to disk
    pub fn save(&self, snapshot: &Snapshot) -> Result<PathBuf> {
        let filename = format!("snapshot_{:016}.snap", snapshot.id);
        let path = self.dir.join(&filename);

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        let header = SnapshotHeader {
            magic: *SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            id: snapshot.id,
            wal_seq: snapshot.wal_seq,
            dimensions: snapshot.dimensions,
            vector_count: snapshot.vectors.len(),
        };

        serialize_into(&mut writer, &header).map_err(|e| Error::Storage(e.to_string()))?;

        // Write HNSW state
        serialize_into(&mut writer, &snapshot.hnsw_state)
            .map_err(|e| Error::Storage(e.to_string()))?;

        // Write vectors in batches for efficiency
        const BATCH_SIZE: usize = 1000;
        for chunk in snapshot.vectors.chunks(BATCH_SIZE) {
            serialize_into(&mut writer, &chunk.to_vec())
                .map_err(|e| Error::Storage(e.to_string()))?;
        }

        // Cleanup old snapshots
        self.cleanup()?;

        Ok(path)
    }

    /// Load the latest snapshot
    pub fn load_latest(&self) -> Result<Option<Snapshot>> {
        let snapshots = self.list_snapshots()?;

        if let Some((_, path)) = snapshots.last() {
            self.load(path).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Load a specific snapshot
    pub fn load(&self, path: &Path) -> Result<Snapshot> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let header: SnapshotHeader =
            deserialize_from(&mut reader).map_err(|e| Error::Storage(e.to_string()))?;

        // Verify magic
        if header.magic != *SNAPSHOT_MAGIC {
            return Err(Error::Storage("Invalid snapshot magic bytes".into()));
        }

        if header.version != SNAPSHOT_VERSION {
            return Err(Error::Storage(format!(
                "Unsupported snapshot version: {}",
                header.version
            )));
        }

        // Read HNSW state
        let hnsw_state: Option<HnswState> =
            deserialize_from(&mut reader).map_err(|e| Error::Storage(e.to_string()))?;

        // Read vectors
        let mut vectors = Vec::with_capacity(header.vector_count);
        let mut remaining = header.vector_count;

        while remaining > 0 {
            let batch: Vec<StoredVector> =
                deserialize_from(&mut reader).map_err(|e| Error::Storage(e.to_string()))?;
            remaining = remaining.saturating_sub(batch.len());
            vectors.extend(batch);
        }

        Ok(Snapshot {
            id: header.id,
            wal_seq: header.wal_seq,
            dimensions: header.dimensions,
            vectors,
            hnsw_state,
        })
    }

    /// List all snapshots sorted by ID
    pub fn list_snapshots(&self) -> Result<Vec<(u64, PathBuf)>> {
        let mut snapshots = Vec::new();

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("snapshot_") && name.ends_with(".snap") {
                    // Parse ID from filename
                    if let Some(id_str) = name
                        .strip_prefix("snapshot_")
                        .and_then(|s| s.strip_suffix(".snap"))
                    {
                        if let Ok(id) = id_str.parse::<u64>() {
                            snapshots.push((id, path));
                        }
                    }
                }
            }
        }

        snapshots.sort_by_key(|(id, _)| *id);
        Ok(snapshots)
    }

    /// Delete old snapshots, keeping only retain_count
    fn cleanup(&self) -> Result<()> {
        let snapshots = self.list_snapshots()?;

        if snapshots.len() > self.retain_count {
            let to_delete = snapshots.len() - self.retain_count;
            for (_, path) in snapshots.into_iter().take(to_delete) {
                fs::remove_file(path)?;
            }
        }

        Ok(())
    }

    /// Get snapshot directory
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_snapshot_save_and_load() {
        let dir = tempdir().unwrap();
        let manager = SnapshotManager::new(dir.path()).unwrap();

        let mut snapshot = Snapshot::new(1, 100, 4);
        snapshot.add_vector("v1".into(), vec![1.0, 2.0, 3.0, 4.0], None);
        snapshot.add_vector("v2".into(), vec![5.0, 6.0, 7.0, 8.0], None);

        let path = manager.save(&snapshot).unwrap();
        assert!(path.exists());

        let loaded = manager.load(&path).unwrap();
        assert_eq!(loaded.id, 1);
        assert_eq!(loaded.wal_seq, 100);
        assert_eq!(loaded.dimensions, 4);
        assert_eq!(loaded.vectors.len(), 2);
        assert_eq!(loaded.vectors[0].id.as_str(), "v1");
    }

    #[test]
    fn test_snapshot_load_latest() {
        let dir = tempdir().unwrap();
        let manager = SnapshotManager::new(dir.path()).unwrap();

        // Create multiple snapshots
        for i in 1..=5 {
            let mut snapshot = Snapshot::new(i, i * 10, 4);
            snapshot.add_vector(format!("v{}", i).into(), vec![i as f32], None);
            manager.save(&snapshot).unwrap();
        }

        let latest = manager.load_latest().unwrap().unwrap();
        assert_eq!(latest.id, 5);
        assert_eq!(latest.wal_seq, 50);
    }

    #[test]
    fn test_snapshot_cleanup() {
        let dir = tempdir().unwrap();
        let mut manager = SnapshotManager::new(dir.path()).unwrap();
        manager.set_retain_count(2);

        // Create 5 snapshots
        for i in 1..=5 {
            let snapshot = Snapshot::new(i, i * 10, 4);
            manager.save(&snapshot).unwrap();
        }

        // Should only have 2 snapshots
        let snapshots = manager.list_snapshots().unwrap();
        assert_eq!(snapshots.len(), 2);

        // Should be the latest 2
        assert_eq!(snapshots[0].0, 4);
        assert_eq!(snapshots[1].0, 5);
    }

    #[test]
    fn test_large_snapshot() {
        let dir = tempdir().unwrap();
        let manager = SnapshotManager::new(dir.path()).unwrap();

        let mut snapshot = Snapshot::new(1, 100, 384);

        // Add 5000 vectors (realistic size)
        for i in 0..5000 {
            let vector: Vec<f32> = (0..384).map(|j| (i * j) as f32 / 1000.0).collect();
            snapshot.add_vector(format!("v{}", i).into(), vector, None);
        }

        let path = manager.save(&snapshot).unwrap();
        let loaded = manager.load(&path).unwrap();

        assert_eq!(loaded.vectors.len(), 5000);
        assert_eq!(loaded.vectors[4999].id.as_str(), "v4999");
    }
}
