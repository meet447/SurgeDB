//! Write-Ahead Log (WAL) for ACID-compliant persistence
//!
//! Provides crash recovery by logging all operations before applying them.
//! On startup, the WAL is replayed to restore the database to its last consistent state.
//!
//! ## Design
//! - Each operation is logged to disk before being applied
//! - Periodic snapshots reduce recovery time
//! - CRC32 checksums ensure data integrity

use crate::error::{Error, Result};
use crate::types::VectorId;
use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic bytes to identify WAL files
const WAL_MAGIC: &[u8; 4] = b"ZWAL";

/// Current WAL format version
const WAL_VERSION: u8 = 1;

/// WAL entry types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    /// Insert a new vector
    Insert {
        id: VectorId,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Delete a vector
    Delete { id: VectorId },
    /// Checkpoint marker (snapshot was taken)
    Checkpoint { snapshot_id: u64 },
}

/// WAL record with checksum
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WalRecord {
    /// Sequence number for ordering
    seq: u64,
    /// The actual entry
    entry: WalEntry,
    /// CRC32 checksum of serialized entry
    checksum: u32,
}

impl WalRecord {
    fn new(seq: u64, entry: WalEntry) -> Self {
        let entry_bytes = serialize(&entry).unwrap_or_default();
        let checksum = crc32(&entry_bytes);
        Self {
            seq,
            entry,
            checksum,
        }
    }

    fn verify(&self) -> bool {
        let entry_bytes = serialize(&self.entry).unwrap_or_default();
        crc32(&entry_bytes) == self.checksum
    }
}

/// Simple CRC32 implementation (IEEE polynomial)
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Write-Ahead Log manager
pub struct Wal {
    /// Directory containing WAL files
    dir: PathBuf,
    /// Current WAL file
    file: Option<BufWriter<File>>,
    /// Current sequence number
    seq: u64,
    /// Last checkpoint sequence
    last_checkpoint_seq: u64,
    /// Maximum WAL size before auto-checkpoint (bytes)
    max_wal_size: u64,
    /// Current WAL file size
    current_size: u64,
}

impl Wal {
    /// Create or open a WAL in the specified directory
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        let wal_path = dir.join("current.wal");
        let (file, seq, size) = if wal_path.exists() {
            // Open existing WAL and find last sequence number
            let mut f = OpenOptions::new().read(true).append(true).open(&wal_path)?;

            let metadata = f.metadata()?;
            let size = metadata.len();

            // Find the last sequence number
            let seq = Self::find_last_seq(&wal_path).unwrap_or(0);

            f.seek(SeekFrom::End(0))?;
            (Some(BufWriter::new(f)), seq, size)
        } else {
            // Create new WAL file
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&wal_path)?;

            let mut writer = BufWriter::new(f);
            // Write header
            writer.write_all(WAL_MAGIC)?;
            writer.write_all(&[WAL_VERSION])?;
            writer.flush()?;

            (Some(writer), 0, 5) // 5 bytes for header
        };

        Ok(Self {
            dir,
            file,
            seq,
            last_checkpoint_seq: 0,
            max_wal_size: 64 * 1024 * 1024, // 64MB default
            current_size: size,
        })
    }

    /// Find the last sequence number in a WAL file
    fn find_last_seq(path: &Path) -> Option<u64> {
        let file = File::open(path).ok()?;
        let mut reader = BufReader::new(file);

        // Skip header
        let mut header = [0u8; 5];
        reader.read_exact(&mut header).ok()?;

        let mut last_seq = 0u64;

        loop {
            // Read record length
            let mut len_bytes = [0u8; 4];
            if reader.read_exact(&mut len_bytes).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;

            // Read record data
            let mut data = vec![0u8; len];
            if reader.read_exact(&mut data).is_err() {
                break;
            }

            // Deserialize record
            if let Ok(record) = deserialize::<WalRecord>(&data) {
                if record.verify() {
                    last_seq = record.seq;
                }
            }
        }

        Some(last_seq)
    }

    /// Append an entry to the WAL
    pub fn append(&mut self, entry: WalEntry) -> Result<u64> {
        self.seq += 1;
        let record = WalRecord::new(self.seq, entry);

        let data = serialize(&record).map_err(|e| Error::Storage(e.to_string()))?;
        let len = data.len() as u32;

        if let Some(ref mut file) = self.file {
            file.write_all(&len.to_le_bytes())?;
            file.write_all(&data)?;
            file.flush()?;

            self.current_size += 4 + data.len() as u64;
        }

        Ok(self.seq)
    }

    /// Sync WAL to disk (fsync)
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref mut file) = self.file {
            file.flush()?;
            file.get_ref().sync_all()?;
        }
        Ok(())
    }

    /// Read all entries from the WAL (for recovery)
    pub fn read_all(&self) -> Result<Vec<WalEntry>> {
        let wal_path = self.dir.join("current.wal");
        if !wal_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&wal_path)?;
        let mut reader = BufReader::new(file);

        // Read and verify header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != WAL_MAGIC {
            return Err(Error::Storage("Invalid WAL magic bytes".into()));
        }

        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != WAL_VERSION {
            return Err(Error::Storage(format!(
                "Unsupported WAL version: {}",
                version[0]
            )));
        }

        let mut entries = Vec::new();

        loop {
            // Read record length
            let mut len_bytes = [0u8; 4];
            if reader.read_exact(&mut len_bytes).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;

            // Read record data
            let mut data = vec![0u8; len];
            if reader.read_exact(&mut data).is_err() {
                break;
            }

            // Deserialize and verify record
            match deserialize::<WalRecord>(&data) {
                Ok(record) => {
                    if record.verify() {
                        entries.push(record.entry);
                    } else {
                        // Corrupted record, stop reading
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        Ok(entries)
    }

    /// Read entries after a specific sequence number (for recovery after checkpoint)
    pub fn read_after(&self, after_seq: u64) -> Result<Vec<WalEntry>> {
        let wal_path = self.dir.join("current.wal");
        if !wal_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&wal_path)?;
        let mut reader = BufReader::new(file);

        // Skip header
        let mut header = [0u8; 5];
        reader.read_exact(&mut header)?;

        let mut entries = Vec::new();

        loop {
            let mut len_bytes = [0u8; 4];
            if reader.read_exact(&mut len_bytes).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;

            let mut data = vec![0u8; len];
            if reader.read_exact(&mut data).is_err() {
                break;
            }

            match deserialize::<WalRecord>(&data) {
                Ok(record) => {
                    if record.verify() && record.seq > after_seq {
                        entries.push(record.entry);
                    }
                }
                Err(_) => break,
            }
        }

        Ok(entries)
    }

    /// Clear the WAL (after successful checkpoint)
    pub fn clear(&mut self) -> Result<()> {
        // Close current file
        self.file = None;

        let wal_path = self.dir.join("current.wal");

        // Create new empty WAL
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&wal_path)?;

        let mut writer = BufWriter::new(f);
        writer.write_all(WAL_MAGIC)?;
        writer.write_all(&[WAL_VERSION])?;
        writer.flush()?;

        self.file = Some(writer);
        self.last_checkpoint_seq = self.seq;
        self.current_size = 5;

        Ok(())
    }

    /// Get current sequence number
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Get WAL directory
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Check if WAL needs checkpointing
    pub fn needs_checkpoint(&self) -> bool {
        self.current_size > self.max_wal_size
    }

    /// Set maximum WAL size before checkpoint
    pub fn set_max_size(&mut self, size: u64) {
        self.max_wal_size = size;
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        if let Some(ref mut file) = self.file {
            let _ = file.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_create_and_append() {
        let dir = tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();

        let seq = wal
            .append(WalEntry::Insert {
                id: "test".into(),
                vector: vec![1.0, 2.0, 3.0],
                metadata: None,
            })
            .unwrap();

        assert_eq!(seq, 1);
        wal.sync().unwrap();
    }

    #[test]
    fn test_wal_read_all() {
        let dir = tempdir().unwrap();

        // Write some entries
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            wal.append(WalEntry::Insert {
                id: "v1".into(),
                vector: vec![1.0, 2.0],
                metadata: None,
            })
            .unwrap();
            wal.append(WalEntry::Insert {
                id: "v2".into(),
                vector: vec![3.0, 4.0],
                metadata: None,
            })
            .unwrap();
            wal.append(WalEntry::Delete { id: "v1".into() }).unwrap();
            wal.sync().unwrap();
        }

        // Read them back
        {
            let wal = Wal::open(dir.path()).unwrap();
            let entries = wal.read_all().unwrap();

            assert_eq!(entries.len(), 3);

            match &entries[0] {
                WalEntry::Insert { id, vector, .. } => {
                    assert_eq!(id.as_str(), "v1");
                    assert_eq!(vector, &vec![1.0, 2.0]);
                }
                _ => panic!("Expected Insert"),
            }

            match &entries[2] {
                WalEntry::Delete { id } => {
                    assert_eq!(id.as_str(), "v1");
                }
                _ => panic!("Expected Delete"),
            }
        }
    }

    #[test]
    fn test_wal_recovery_after_reopen() {
        let dir = tempdir().unwrap();

        // Write entries and close
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            for i in 0..10 {
                wal.append(WalEntry::Insert {
                    id: format!("v{}", i).into(),
                    vector: vec![i as f32],
                    metadata: None,
                })
                .unwrap();
            }
            wal.sync().unwrap();
        }

        // Reopen and verify sequence continues
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            assert_eq!(wal.seq(), 10);

            let seq = wal
                .append(WalEntry::Insert {
                    id: "v10".into(),
                    vector: vec![10.0],
                    metadata: None,
                })
                .unwrap();
            assert_eq!(seq, 11);
        }
    }

    #[test]
    fn test_wal_clear() {
        let dir = tempdir().unwrap();

        let mut wal = Wal::open(dir.path()).unwrap();

        // Add some entries
        for i in 0..5 {
            wal.append(WalEntry::Insert {
                id: format!("v{}", i).into(),
                vector: vec![i as f32],
                metadata: None,
            })
            .unwrap();
        }

        // Clear the WAL
        wal.clear().unwrap();

        // Read should return empty
        let entries = wal.read_all().unwrap();
        assert!(entries.is_empty());

        // But we can still append
        let seq = wal
            .append(WalEntry::Insert {
                id: "new".into(),
                vector: vec![1.0],
                metadata: None,
            })
            .unwrap();
        assert_eq!(seq, 6); // Sequence continues
    }

    #[test]
    fn test_crc32() {
        let data = b"hello world";
        let checksum = crc32(data);
        assert_eq!(checksum, 0x0D4A1185);
    }
}
