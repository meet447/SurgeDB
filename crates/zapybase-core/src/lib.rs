//! ZappyBase Core - High-performance vector database engine
//!
//! A lightweight, SIMD-optimized vector database designed for edge devices.
//!
//! # Features
//! - SIMD-accelerated distance calculations (NEON/AVX-512)
//! - Adaptive HNSW indexing (In-Memory, Mmap, Hybrid)
//! - Built-in quantization (SQ8, Binary)
//! - ACID-compliant persistence
//!
//! # Quick Start
//! ```rust,no_run
//! use zapybase_core::{VectorDb, Config, DistanceMetric};
//!
//! let config = Config::default();
//! let mut db = VectorDb::new(config).unwrap();
//!
//! // Insert vectors
//! db.insert("vec1", &[0.1, 0.2, 0.3, 0.4]).unwrap();
//!
//! // Search for similar vectors
//! let results = db.search(&[0.1, 0.2, 0.3, 0.4], 10).unwrap();
//! ```
//!
//! # Quantized Database (4x memory reduction)
//! ```rust,no_run
//! use zapybase_core::{QuantizedVectorDb, QuantizedConfig, QuantizationType};
//!
//! let config = QuantizedConfig {
//!     dimensions: 384,
//!     quantization: QuantizationType::SQ8,
//!     ..Default::default()
//! };
//! let mut db = QuantizedVectorDb::new(config).unwrap();
//!
//! // Same API as VectorDb
//! db.insert("vec1", &[0.1, 0.2, 0.3, 0.4]).unwrap();
//! ```
//!
//! # Persistent Database (with crash recovery)
//! ```rust,no_run
//! use zapybase_core::{PersistentVectorDb, PersistentConfig};
//!
//! let config = PersistentConfig::default();
//! let mut db = PersistentVectorDb::open("./my_db", config).unwrap();
//!
//! db.insert("vec1", &[0.1, 0.2, 0.3, 0.4]).unwrap();
//! db.checkpoint().unwrap(); // Create a snapshot
//! ```

pub mod db;
pub mod distance;
pub mod error;
pub mod hnsw;
pub mod mmap_db;
pub mod mmap_storage;
pub mod persistent;
pub mod quantization;
pub mod quantized_storage;
pub mod snapshot;
pub mod storage;
pub mod types;
pub mod wal;

// Re-exports
pub use db::{Database, DatabaseStats};
pub use distance::DistanceMetric;
pub use error::{Error, Result};
pub use hnsw::{HnswConfig, HnswIndex};
pub use mmap_db::{MmapConfig, MmapVectorDb};
pub use mmap_storage::MmapStorage;
pub use persistent::{PersistentConfig, PersistentVectorDb};
pub use quantization::{BinaryQuantizer, QuantizationType, SQ8Quantizer};
pub use quantized_storage::QuantizedStorage;
pub use snapshot::{Snapshot, SnapshotManager};
pub use storage::{VectorStorage, VectorStorageTrait};
pub use types::{Vector, VectorId};
pub use wal::{Wal, WalEntry};

/// Main database configuration (unquantized)
#[derive(Debug, Clone)]
pub struct Config {
    /// Dimensionality of vectors
    pub dimensions: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// HNSW configuration
    pub hnsw: HnswConfig,
    /// Maximum number of vectors (0 = unlimited)
    pub max_vectors: usize,
    /// Quantization type (None by default)
    pub quantization: QuantizationType,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dimensions: 384, // Common for MiniLM embeddings
            distance_metric: DistanceMetric::Cosine,
            hnsw: HnswConfig::default(),
            max_vectors: 0,
            quantization: QuantizationType::None,
        }
    }
}

/// Configuration for quantized vector database
#[derive(Debug, Clone)]
pub struct QuantizedConfig {
    /// Dimensionality of vectors
    pub dimensions: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// HNSW configuration
    pub hnsw: HnswConfig,
    /// Quantization type
    pub quantization: QuantizationType,
    /// Keep original vectors for re-ranking
    pub keep_originals: bool,
    /// Number of candidates to fetch before re-ranking (if keep_originals is true)
    pub rerank_multiplier: usize,
}

impl Default for QuantizedConfig {
    fn default() -> Self {
        Self {
            dimensions: 384,
            distance_metric: DistanceMetric::Cosine,
            hnsw: HnswConfig::default(),
            quantization: QuantizationType::SQ8,
            keep_originals: false,
            rerank_multiplier: 3,
        }
    }
}

use serde_json::Value;

/// The main vector database interface (unquantized)
pub struct VectorDb {
    config: Config,
    storage: VectorStorage,
    index: HnswIndex,
}

impl VectorDb {
    /// Create a new vector database with the given configuration
    pub fn new(config: Config) -> Result<Self> {
        let storage = VectorStorage::new(config.dimensions);
        let index = HnswIndex::new(config.hnsw.clone(), config.distance_metric);

        Ok(Self {
            config,
            storage,
            index,
        })
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

        let internal_id = self.storage.insert(id.clone(), vector, metadata)?;
        self.index.insert(internal_id, vector, &self.storage)?;

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

        // Map internal IDs back to external IDs and fetch metadata
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

    /// Get the number of vectors in the database
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get approximate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage() + self.index.memory_usage()
    }
}

/// Quantized vector database with configurable compression
///
/// Uses SQ8 (4x compression) or Binary (32x compression) quantization
/// to dramatically reduce memory usage with minimal accuracy loss.
pub struct QuantizedVectorDb {
    config: QuantizedConfig,
    storage: QuantizedStorage,
}

impl QuantizedVectorDb {
    /// Create a new quantized vector database
    pub fn new(config: QuantizedConfig) -> Result<Self> {
        let storage = QuantizedStorage::new(
            config.dimensions,
            config.quantization,
            config.keep_originals,
        );

        Ok(Self { config, storage })
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

        self.storage.insert(id, vector, metadata)?;
        Ok(())
    }

    /// Search for the k nearest neighbors using brute force on quantized vectors
    ///
    /// For large datasets, this should be combined with HNSW indexing.
    /// This implementation is optimized for small-medium datasets (< 100K vectors).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32, Option<Value>)>> {
        if query.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        if self.storage.is_empty() {
            return Err(Error::EmptyIndex);
        }

        let metric = self.config.distance_metric;

        // Calculate distances to all vectors
        let mut candidates: Vec<(types::InternalId, f32)> = self
            .storage
            .all_internal_ids()
            .into_iter()
            .filter_map(|id| {
                self.storage
                    .distance(query, id, metric)
                    .map(|dist| (id, dist))
            })
            .collect();

        // Sort by distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // If re-ranking is enabled, fetch more candidates and re-rank with originals
        let results: Vec<(types::InternalId, f32)> =
            if self.config.keep_originals && self.config.quantization != QuantizationType::None {
                let fetch_count = k * self.config.rerank_multiplier;
                let top_candidates: Vec<_> = candidates.into_iter().take(fetch_count).collect();

                // Re-rank using original vectors
                let mut reranked: Vec<_> = top_candidates
                    .into_iter()
                    .filter_map(|(id, _)| {
                        self.storage.get_original(id).map(|orig| {
                            let dist = metric.distance(query, &orig);
                            (id, dist)
                        })
                    })
                    .collect();

                reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                reranked.into_iter().take(k).collect()
            } else {
                candidates.into_iter().take(k).collect()
            };

        // Map to external IDs and fetch metadata
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

    /// Get the number of vectors in the database
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get configuration
    pub fn config(&self) -> &QuantizedConfig {
        &self.config
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }

    /// Get compression ratio compared to unquantized storage
    pub fn compression_ratio(&self) -> f32 {
        self.storage.compression_ratio()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_and_search() {
        let config = Config {
            dimensions: 4,
            ..Default::default()
        };

        let mut db = VectorDb::new(config).unwrap();

        db.insert("vec1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        db.insert("vec2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        db.insert("vec3", &[0.9, 0.1, 0.0, 0.0], None).unwrap();

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.as_str(), "vec1"); // Exact match should be first
    }

    #[test]
    fn test_insert_with_metadata() {
        let config = Config {
            dimensions: 4,
            ..Default::default()
        };

        let mut db = VectorDb::new(config).unwrap();
        let meta = serde_json::json!({"type": "test"});

        db.insert("vec1", &[1.0, 0.0, 0.0, 0.0], Some(meta.clone()))
            .unwrap();

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].2, Some(meta));
    }

    #[test]
    fn test_quantized_sq8_insert_and_search() {
        let config = QuantizedConfig {
            dimensions: 4,
            quantization: QuantizationType::SQ8,
            ..Default::default()
        };

        let mut db = QuantizedVectorDb::new(config).unwrap();

        db.insert("vec1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        db.insert("vec2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        db.insert("vec3", &[0.9, 0.1, 0.0, 0.0], None).unwrap();

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        // First result should be vec1 (exact match) or vec3 (very similar)
        assert!(
            results[0].0.as_str() == "vec1" || results[0].0.as_str() == "vec3",
            "First result: {}",
            results[0].0.as_str()
        );
    }

    #[test]
    fn test_quantized_binary_insert_and_search() {
        let config = QuantizedConfig {
            dimensions: 8,
            quantization: QuantizationType::Binary,
            ..Default::default()
        };

        let mut db = QuantizedVectorDb::new(config).unwrap();

        db.insert("vec1", &[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], None)
            .unwrap();
        db.insert("vec2", &[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], None)
            .unwrap();
        db.insert("vec3", &[1.0, 1.0, 1.0, 0.5, -1.0, -1.0, -1.0, -0.5], None)
            .unwrap();

        let results = db
            .search(&[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], 2)
            .unwrap();

        assert_eq!(results.len(), 2);
        // First result should be vec1 (exact match)
        assert_eq!(results[0].0.as_str(), "vec1");
    }

    #[test]
    fn test_quantized_with_reranking() {
        let config = QuantizedConfig {
            dimensions: 4,
            quantization: QuantizationType::SQ8,
            keep_originals: true,
            rerank_multiplier: 2,
            ..Default::default()
        };

        let mut db = QuantizedVectorDb::new(config).unwrap();

        db.insert("vec1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        db.insert("vec2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        db.insert("vec3", &[0.95, 0.05, 0.0, 0.0], None).unwrap();

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        // With re-ranking, vec1 should definitely be first
        assert_eq!(results[0].0.as_str(), "vec1");
    }

    #[test]
    fn test_compression_ratio() {
        let config = QuantizedConfig {
            dimensions: 384,
            quantization: QuantizationType::SQ8,
            keep_originals: false,
            ..Default::default()
        };

        let mut db = QuantizedVectorDb::new(config).unwrap();

        // Insert 100 vectors
        for i in 0..100 {
            let vector: Vec<f32> = (0..384).map(|j| ((i * j) as f32).sin()).collect();
            db.insert(format!("v{}", i), &vector, None).unwrap();
        }

        let ratio = db.compression_ratio();
        println!("SQ8 compression ratio: {:.2}x", ratio);
        assert!(ratio > 3.5, "Expected > 3.5x compression, got {}", ratio);
    }
}
