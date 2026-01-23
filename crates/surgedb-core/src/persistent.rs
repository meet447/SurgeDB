//! Persistent vector database with WAL and snapshots
//!
//! Provides ACID-compliant persistence with crash recovery.

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::snapshot::{Snapshot, SnapshotManager};
use crate::storage::VectorStorage;
use crate::types::VectorId;
use crate::wal::{Wal, WalEntry};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for persistent database
#[derive(Debug, Clone)]
pub struct PersistentConfig {
    /// Dimensionality of vectors
    pub dimensions: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// HNSW configuration
    pub hnsw: HnswConfig,
    /// Sync WAL after every write (safer but slower)
    pub sync_writes: bool,
    /// Auto-checkpoint when WAL exceeds this size (bytes)
    pub checkpoint_threshold: u64,
    /// Number of snapshots to retain
    pub snapshot_retain_count: usize,
}

impl Default for PersistentConfig {
    fn default() -> Self {
        Self {
            dimensions: 384,
            distance_metric: DistanceMetric::Cosine,
            hnsw: HnswConfig::default(),
            sync_writes: false,
            checkpoint_threshold: 64 * 1024 * 1024, // 64MB
            snapshot_retain_count: 3,
        }
    }
}

/// Persistent vector database with ACID guarantees
pub struct PersistentVectorDb {
    config: PersistentConfig,
    storage: VectorStorage,
    index: HnswIndex,
    wal: Wal,
    snapshot_manager: SnapshotManager,
    data_dir: PathBuf,
}

impl PersistentVectorDb {
    /// Open or create a persistent database at the given path
    pub fn open(path: impl AsRef<Path>, config: PersistentConfig) -> Result<Self> {
        let data_dir = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        let wal_dir = data_dir.join("wal");
        let snapshot_dir = data_dir.join("snapshots");

        let mut wal = Wal::open(&wal_dir)?;
        wal.set_max_size(config.checkpoint_threshold);

        let mut snapshot_manager = SnapshotManager::new(&snapshot_dir)?;
        snapshot_manager.set_retain_count(config.snapshot_retain_count);

        let storage = VectorStorage::new(config.dimensions);
        let index = HnswIndex::new(config.hnsw.clone(), config.distance_metric);

        let mut db = Self {
            config,
            storage,
            index,
            wal,
            snapshot_manager,
            data_dir,
        };

        // Recover from snapshot and WAL
        db.recover()?;

        Ok(db)
    }

    /// Recover database state from snapshot and WAL
    fn recover(&mut self) -> Result<()> {
        let mut last_wal_seq = 0u64;

        // Load latest snapshot if available
        if let Some(snapshot) = self.snapshot_manager.load_latest()? {
            last_wal_seq = snapshot.wal_seq;

            // Verify dimensions match
            if snapshot.dimensions != self.config.dimensions {
                return Err(Error::InvalidConfig(format!(
                    "Snapshot dimensions ({}) don't match config ({})",
                    snapshot.dimensions, self.config.dimensions
                )));
            }

            // Restore vectors from snapshot
            for stored in snapshot.vectors {
                self.storage
                    .insert(stored.id, &stored.vector, stored.metadata)?;
            }

            // Restore HNSW state if available
            if let Some(state) = snapshot.hnsw_state {
                self.index.load_state(state);
            } else {
                // Fallback: rebuild index if state is missing
                for internal_id in self.storage.all_internal_ids() {
                    if let Some(vector) = self.storage.get_vector_data(internal_id) {
                        self.index.insert(internal_id, &vector, &self.storage)?;
                    }
                }
            }
        }

        // Replay WAL entries after snapshot
        let entries = self.wal.read_after(last_wal_seq)?;

        for entry in entries {
            match entry {
                WalEntry::Insert {
                    id,
                    vector,
                    metadata,
                } => {
                    // Skip if already in storage (duplicate)
                    if self.storage.get_internal_id(&id).is_none() {
                        let internal_id = self.storage.insert(id, &vector, metadata)?;
                        self.index.insert(internal_id, &vector, &self.storage)?;
                    }
                }
                WalEntry::Delete { id } => {
                    // TODO: Implement delete in storage/index
                    let _ = id;
                }
                WalEntry::Checkpoint { .. } => {
                    // Checkpoint markers don't need replay
                }
            }
        }

        Ok(())
    }

    /// Insert a vector with the given ID and optional metadata
    pub fn insert(
        &mut self,
        id: impl Into<VectorId>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();

        if vector.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        // Write to WAL first (durability)
        self.wal.append(WalEntry::Insert {
            id: id.clone(),
            vector: vector.to_vec(),
            metadata: metadata.clone(),
        })?;

        if self.config.sync_writes {
            self.wal.sync()?;
        }

        // Then apply to in-memory structures
        let internal_id = self.storage.insert(id, vector, metadata)?;
        self.index.insert(internal_id, vector, &self.storage)?;

        // Check if we need to checkpoint
        if self.wal.needs_checkpoint() {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Search for the k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32, Option<Value>)>> {
        if query.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        let results = self.index.search(query, k, &self.storage)?;

        let mapped: Vec<(VectorId, f32, Option<Value>)> = results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                self.storage.get_external_id(internal_id).map(|ext_id| {
                    let metadata = self.storage.get_metadata(internal_id);
                    (ext_id, distance, metadata)
                })
            })
            .collect();

        Ok(mapped)
    }

    /// Create a checkpoint (snapshot + clear WAL)
    pub fn checkpoint(&mut self) -> Result<()> {
        let snapshot_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let wal_seq = self.wal.seq();

        let mut snapshot = Snapshot::new(snapshot_id, wal_seq, self.config.dimensions);

        // Add all vectors to snapshot
        for internal_id in self.storage.all_internal_ids() {
            if let Some(ext_id) = self.storage.get_external_id(internal_id) {
                if let Some(vector) = self.storage.get(internal_id) {
                    let metadata = self.storage.get_metadata(internal_id);
                    snapshot.add_vector(ext_id, vector, metadata);
                }
            }
        }

        // Add index state to snapshot
        snapshot.set_hnsw_state(self.index.get_state());

        // Save snapshot
        self.snapshot_manager.save(&snapshot)?;

        // Clear WAL
        self.wal.clear()?;

        // Log checkpoint in new WAL
        self.wal.append(WalEntry::Checkpoint { snapshot_id })?;

        Ok(())
    }

    /// Force sync WAL to disk
    pub fn sync(&mut self) -> Result<()> {
        self.wal.sync()
    }

    /// Get the number of vectors in the database
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get configuration
    pub fn config(&self) -> &PersistentConfig {
        &self.config
    }

    /// Get data directory
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_persistent_insert_and_search() {
        let dir = tempdir().unwrap();
        let config = PersistentConfig {
            dimensions: 4,
            ..Default::default()
        };

        let mut db = PersistentVectorDb::open(dir.path(), config).unwrap();

        db.insert("vec1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        db.insert("vec2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        db.insert("vec3", &[0.9, 0.1, 0.0, 0.0], None).unwrap();

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.as_str(), "vec1");
    }

    #[test]
    fn test_persistent_recovery() {
        let dir = tempdir().unwrap();
        let config = PersistentConfig {
            dimensions: 4,
            ..Default::default()
        };

        // Insert some vectors
        {
            let mut db = PersistentVectorDb::open(dir.path(), config.clone()).unwrap();
            db.insert("vec1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
            db.insert("vec2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
            db.sync().unwrap();
        }

        // Reopen and verify recovery
        {
            let db = PersistentVectorDb::open(dir.path(), config).unwrap();
            assert_eq!(db.len(), 2);

            let results = db.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(results[0].0.as_str(), "vec1");
        }
    }

    #[test]
    fn test_persistent_checkpoint_and_recovery() {
        let dir = tempdir().unwrap();
        let config = PersistentConfig {
            dimensions: 4,
            ..Default::default()
        };

        // Insert vectors and checkpoint
        {
            let mut db = PersistentVectorDb::open(dir.path(), config.clone()).unwrap();
            db.insert("vec1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
            db.insert("vec2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
            db.checkpoint().unwrap();

            // Add more after checkpoint
            db.insert("vec3", &[0.0, 0.0, 1.0, 0.0], None).unwrap();
            db.sync().unwrap();
        }

        // Reopen and verify full recovery
        {
            let db = PersistentVectorDb::open(dir.path(), config).unwrap();
            assert_eq!(db.len(), 3);

            let results = db.search(&[0.0, 0.0, 1.0, 0.0], 1).unwrap();
            assert_eq!(results[0].0.as_str(), "vec3");
        }
    }

    #[test]
    fn test_persistent_large_dataset() {
        let dir = tempdir().unwrap();
        let config = PersistentConfig {
            dimensions: 128,
            checkpoint_threshold: 1024 * 1024, // 1MB - force frequent checkpoints
            ..Default::default()
        };

        // Insert many vectors
        {
            let mut db = PersistentVectorDb::open(dir.path(), config.clone()).unwrap();

            for i in 0..1000 {
                let vector: Vec<f32> = (0..128).map(|j| ((i * j) as f32).sin()).collect();
                db.insert(format!("v{}", i), &vector, None).unwrap();
            }
            db.sync().unwrap();
        }

        // Reopen and verify
        {
            let db = PersistentVectorDb::open(dir.path(), config).unwrap();
            assert_eq!(db.len(), 1000);
        }
    }
}
