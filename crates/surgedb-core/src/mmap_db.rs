//! Memory-mapped vector database
//!
//! Uses mmap for disk-resident vectors, allowing datasets larger than RAM.
//! The OS manages which pages are in memory, providing automatic caching.

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::mmap_storage::MmapStorage;
use crate::types::VectorId;
use std::path::{Path, PathBuf};

/// Configuration for mmap-based database
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Dimensionality of vectors
    pub dimensions: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// HNSW configuration
    pub hnsw: HnswConfig,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            dimensions: 384,
            distance_metric: DistanceMetric::Cosine,
            hnsw: HnswConfig::default(),
        }
    }
}

/// Memory-mapped vector database
///
/// Uses memory-mapped storage for vectors, allowing datasets larger than RAM.
/// The HNSW index is kept in memory for fast search, while vector data is
/// demand-paged from disk.
pub struct MmapVectorDb {
    config: MmapConfig,
    storage: MmapStorage,
    index: HnswIndex,
    data_dir: PathBuf,
}

impl MmapVectorDb {
    /// Open or create a mmap-based database at the given path
    pub fn open(path: impl AsRef<Path>, config: MmapConfig) -> Result<Self> {
        let data_dir = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        let storage = MmapStorage::open(&data_dir, config.dimensions)?;
        let index = HnswIndex::new(config.hnsw.clone(), config.distance_metric);

        let mut db = Self {
            config,
            storage,
            index,
            data_dir,
        };

        // Try to load index from disk, otherwise rebuild
        if db.load_index().is_err() {
            db.rebuild_index()?;
            db.save_index()?;
        }

        Ok(db)
    }

    /// Load HNSW index state from disk
    fn load_index(&self) -> Result<()> {
        let path = self.data_dir.join("index.state");
        if !path.exists() {
            return Err(Error::Storage("Index state not found".into()));
        }

        let data = std::fs::read(path)?;
        let state = bincode::deserialize(&data).map_err(|e| Error::Storage(e.to_string()))?;
        self.index.load_state(state);
        Ok(())
    }

    /// Save HNSW index state to disk
    pub fn save_index(&self) -> Result<()> {
        let path = self.data_dir.join("index.state");
        let state = self.index.get_state();
        let data = bincode::serialize(&state).map_err(|e| Error::Storage(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Rebuild HNSW index from storage
    fn rebuild_index(&mut self) -> Result<()> {
        let ids = self.storage.all_internal_ids();

        for internal_id in ids {
            if let Some(vector) = self.storage.get_vector_data(internal_id) {
                self.index.insert(internal_id, &vector, &self.storage)?;
            }
        }

        Ok(())
    }

    /// Insert a vector with the given ID
    pub fn insert(&mut self, id: impl Into<VectorId>, vector: &[f32]) -> Result<()> {
        let id = id.into();

        if vector.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        let internal_id = self.storage.insert(id, vector)?;
        self.index.insert(internal_id, vector, &self.storage)?;

        Ok(())
    }

    /// Search for the k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>> {
        if query.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        let results = self.index.search(query, k, &self.storage.view(), None)?;

        let mapped: Vec<(VectorId, f32)> = results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                self.storage
                    .get_external_id(internal_id)
                    .map(|ext_id| (ext_id, distance))
            })
            .collect();

        Ok(mapped)
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
    pub fn config(&self) -> &MmapConfig {
        &self.config
    }

    /// Get data directory
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Get disk usage in bytes
    pub fn disk_usage(&self) -> u64 {
        self.storage.disk_usage()
    }

    /// Sync data to disk
    pub fn sync(&self) -> Result<()> {
        self.storage.sync()?;
        self.save_index()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_db_insert_and_search() {
        let dir = tempdir().unwrap();
        let config = MmapConfig {
            dimensions: 4,
            ..Default::default()
        };

        let mut db = MmapVectorDb::open(dir.path(), config).unwrap();

        db.insert("vec1", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        db.insert("vec2", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        db.insert("vec3", &[0.9, 0.1, 0.0, 0.0]).unwrap();

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.as_str(), "vec1");
    }

    #[test]
    fn test_mmap_db_persistence() {
        let dir = tempdir().unwrap();
        let config = MmapConfig {
            dimensions: 4,
            ..Default::default()
        };

        // Create and populate
        {
            let mut db = MmapVectorDb::open(dir.path(), config.clone()).unwrap();
            db.insert("vec1", &[1.0, 0.0, 0.0, 0.0]).unwrap();
            db.insert("vec2", &[0.0, 1.0, 0.0, 0.0]).unwrap();
            db.insert("vec3", &[0.9, 0.1, 0.0, 0.0]).unwrap();
            db.sync().unwrap();
        }

        // Reopen and verify
        {
            let db = MmapVectorDb::open(dir.path(), config).unwrap();
            assert_eq!(db.len(), 3);

            let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0.as_str(), "vec1");
        }
    }

    #[test]
    fn test_mmap_db_large_dataset() {
        let dir = tempdir().unwrap();
        let config = MmapConfig {
            dimensions: 128,
            ..Default::default()
        };

        let mut db = MmapVectorDb::open(dir.path(), config).unwrap();

        // Insert 1000 vectors
        for i in 0..1000 {
            let vector: Vec<f32> = (0..128).map(|j| ((i * j) as f32).sin()).collect();
            db.insert(format!("v{}", i), &vector).unwrap();
        }

        assert_eq!(db.len(), 1000);

        // Search should work
        let query: Vec<f32> = (0..128).map(|j| (j as f32).sin()).collect();
        let results = db.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);

        // Check disk usage
        let expected_min = 1000 * 128 * 4; // At least the raw vector data
        assert!(db.disk_usage() >= expected_min as u64);
    }
}
