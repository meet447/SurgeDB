//! SurgeDB Core - High-performance vector database engine
//!
//! A lightweight, SIMD-optimized vector database designed for edge devices.
//!
//! # Features
//! - SIMD-accelerated distance calculations (NEON/AVX-512)
//! - Adaptive HNSW indexing (In-Memory, Mmap, Hybrid)
//! - Built-in quantization (SQ8, Binary)
//! - ACID-compliant persistence (native only, not WASM)
//!
//! # Quick Start
//! ```rust,no_run
//! use surgedb_core::{VectorDb, Config, DistanceMetric};
//!
//! let config = Config::default();
//! let mut db = VectorDb::new(config).unwrap();
//!
//! // Insert vectors
//! db.insert("vec1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
//!
//! // Search for similar vectors
//! let results = db.search(&[0.1, 0.2, 0.3, 0.4], 10, None).unwrap();
//! ```
//!
//! # Quantized Database (4x memory reduction)
//! ```rust,no_run
//! use surgedb_core::{QuantizedVectorDb, QuantizedConfig, QuantizationType};
//!
//! let config = QuantizedConfig {
//!     dimensions: 384,
//!     quantization: QuantizationType::SQ8,
//!     ..Default::default()
//! };
//! let mut db = QuantizedVectorDb::new(config).unwrap();
//!
//! // Same API as VectorDb
//! db.insert("vec1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
//! ```
//!
//! # Persistent Database (with crash recovery) - Native only
//! ```rust,no_run
//! use surgedb_core::{PersistentVectorDb, PersistentConfig};
//!
//! let config = PersistentConfig::default();
//! let mut db = PersistentVectorDb::open("./my_db", config).unwrap();
//!
//! db.insert("vec1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
//! db.checkpoint().unwrap(); // Create a snapshot
//! ```

// Core modules (always available)
pub mod distance;
pub mod error;
pub mod filter;
pub mod hnsw;
pub mod quantization;
pub mod quantized_storage;
pub mod storage;
pub mod sync;
pub mod types;

// Persistence modules (native only, requires filesystem)
#[cfg(feature = "persistence")]
pub mod mmap_db;
#[cfg(feature = "persistence")]
pub mod mmap_storage;
#[cfg(feature = "persistence")]
pub mod persistent;
#[cfg(feature = "persistence")]
pub mod snapshot;
#[cfg(feature = "persistence")]
pub mod wal;

// Multi-collection database (uses persistence features conditionally)
pub mod db;

// Re-exports - Core (always available)
pub use distance::DistanceMetric;
pub use error::{Error, Result};
pub use hnsw::{HnswConfig, HnswIndex};
pub use quantization::{BinaryQuantizer, QuantizationType, SQ8Quantizer};
pub use quantized_storage::QuantizedStorage;
pub use storage::{VectorStorage, VectorStorageTrait};
pub use types::{Vector, VectorId};

// Re-exports - Persistence (native only)
#[cfg(feature = "persistence")]
pub use mmap_db::{MmapConfig, MmapVectorDb};
#[cfg(feature = "persistence")]
pub use mmap_storage::MmapStorage;
#[cfg(feature = "persistence")]
pub use persistent::{PersistentConfig, PersistentVectorDb};
#[cfg(feature = "persistence")]
pub use snapshot::{Snapshot, SnapshotManager};
#[cfg(feature = "persistence")]
pub use wal::{Wal, WalEntry};

// Re-exports - Database (conditional based on features)
pub use db::{Database, DatabaseStats};

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

    /// Delete a vector by ID
    pub fn delete(&mut self, id: impl Into<VectorId>) -> Result<bool> {
        let id = id.into();
        self.storage.delete(&id)
    }

    /// Insert or update a vector with the given ID and optional metadata
    pub fn upsert(
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

        let internal_id = self.storage.upsert(id.clone(), vector, metadata)?;
        self.index.insert(internal_id, vector, &self.storage)?;

        Ok(())
    }

    /// Batch insert/upsert vectors
    pub fn upsert_batch(&mut self, items: Vec<(VectorId, Vec<f32>, Option<Value>)>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        // Validate dimensions
        for (_, vector, _) in &items {
            if vector.len() != self.config.dimensions {
                return Err(Error::DimensionMismatch {
                    expected: self.config.dimensions,
                    got: vector.len(),
                });
            }
        }

        // 1. Batch Upsert into Storage (Single lock acquisition)
        let internal_ids = self.storage.upsert_batch(&items)?;

        // 2. Batch Insert into HNSW
        // We prepare a slice of (InternalId, &[f32]) for HNSW
        // This avoids copying vectors again if possible, but here vectors are in `items`
        let hnsw_items: Vec<(types::InternalId, &[f32])> = internal_ids
            .iter()
            .zip(items.iter())
            .map(|(id, (_, vec, _))| (*id, vec.as_slice()))
            .collect();

        self.index.insert_batch(&hnsw_items, &self.storage)?;

        Ok(())
    }

    /// Retrieve a vector by its external ID
    pub fn get(&self, id: &str) -> Result<Option<(Vec<f32>, Option<Value>)>> {
        let id = VectorId::from(id);
        if let Some(internal_id) = self.storage.get_internal_id(&id) {
            let vector = self
                .storage
                .get(internal_id)
                .ok_or(Error::VectorNotFound(id.to_string()))?;
            let metadata = self.storage.get_metadata(internal_id);
            Ok(Some((vector, metadata)))
        } else {
            Ok(None)
        }
    }

    /// List all vector IDs (pagination)
    pub fn list(&self, offset: usize, limit: usize) -> Vec<VectorId> {
        let ids = self.storage.all_internal_ids();
        ids.iter()
            .skip(offset)
            .take(limit)
            .filter_map(|&internal_id| {
                // Filter stale
                let ext_id = self.storage.get_external_id(internal_id)?;
                let current_internal = self.storage.get_internal_id(&ext_id)?;
                if current_internal != internal_id {
                    return None;
                }
                Some(ext_id)
            })
            .collect()
    }

    /// Search for the k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&filter::Filter>,
    ) -> Result<Vec<(VectorId, f32, Option<Value>)>> {
        if query.len() != self.config.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        // We search for more candidates (2x k) to account for potential stale/deleted entries
        // that might be filtered out.
        let search_k = k * 2;
        let results = self
            .index
            .search(query, search_k, &self.storage.view(), filter)?;

        // Map internal IDs back to external IDs and fetch metadata
        // Filter out stale entries (where internal_id doesn't match current mapping)
        let mapped: Vec<(VectorId, f32, Option<Value>)> = results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                let ext_id = self.storage.get_external_id(internal_id)?;
                let current_internal = self.storage.get_internal_id(&ext_id)?;

                if current_internal != internal_id {
                    // This is a stale entry (overwritten by upsert)
                    return None;
                }

                let metadata = self.storage.get_metadata(internal_id);
                Some((ext_id, distance, metadata))
            })
            .take(k)
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
    index: Option<HnswIndex>,
}

impl QuantizedVectorDb {
    /// Create a new quantized vector database
    pub fn new(config: QuantizedConfig) -> Result<Self> {
        let storage = QuantizedStorage::new(
            config.dimensions,
            config.quantization,
            config.keep_originals,
        );

        let index = if config.quantization == QuantizationType::Binary {
            None
        } else {
            Some(HnswIndex::new(config.hnsw.clone(), config.distance_metric))
        };

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

        let internal_id = self.storage.insert(id, vector, metadata)?;

        if let Some(index) = &mut self.index {
            index.insert(internal_id, vector, &self.storage)?;
        }

        Ok(())
    }

    /// Delete a vector by ID
    pub fn delete(&mut self, id: impl Into<VectorId>) -> Result<bool> {
        let id = id.into();
        self.storage.delete(&id)
    }

    /// Insert or update a vector with the given ID and optional metadata
    pub fn upsert(
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

        let internal_id = self.storage.upsert(id, vector, metadata)?;

        if let Some(index) = &mut self.index {
            index.insert(internal_id, vector, &self.storage)?;
        }

        Ok(())
    }

    /// Batch insert/upsert vectors
    pub fn upsert_batch(&mut self, items: Vec<(VectorId, Vec<f32>, Option<Value>)>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        // Validate dimensions
        for (_, vector, _) in &items {
            if vector.len() != self.config.dimensions {
                return Err(Error::DimensionMismatch {
                    expected: self.config.dimensions,
                    got: vector.len(),
                });
            }
        }

        // 1. Batch Upsert into Storage (Single lock acquisition)
        let internal_ids = self.storage.upsert_batch(&items)?;

        // 2. Batch Insert into HNSW
        if let Some(index) = &mut self.index {
            let hnsw_items: Vec<(types::InternalId, &[f32])> = internal_ids
                .iter()
                .zip(items.iter())
                .map(|(id, (_, vec, _))| (*id, vec.as_slice()))
                .collect();

            index.insert_batch(&hnsw_items, &self.storage)?;
        }

        Ok(())
    }

    /// Retrieve a vector by its external ID
    pub fn get(&self, id: &str) -> Result<Option<(Vec<f32>, Option<Value>)>> {
        let id = VectorId::from(id);
        if let Some(internal_id) = self.storage.get_internal_id(&id) {
            let vector =
                crate::storage::VectorStorageTrait::get_vector_data(&self.storage, internal_id)
                    .ok_or(Error::VectorNotFound(id.to_string()))?;
            let metadata = self.storage.get_metadata(internal_id);
            Ok(Some((vector, metadata)))
        } else {
            Ok(None)
        }
    }

    /// List all vector IDs (pagination)
    pub fn list(&self, offset: usize, limit: usize) -> Vec<VectorId> {
        let ids = self.storage.all_internal_ids();
        ids.iter()
            .skip(offset)
            .take(limit)
            .filter_map(|&internal_id| {
                // Filter stale
                let ext_id = self.storage.get_external_id(internal_id)?;
                let current_internal = self.storage.get_internal_id(&ext_id)?;
                if current_internal != internal_id {
                    return None;
                }
                Some(ext_id)
            })
            .collect()
    }

    /// Search for the k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&filter::Filter>,
    ) -> Result<Vec<(VectorId, f32, Option<Value>)>> {
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

        // Increase K to account for stale entries and re-ranking
        let multiplier =
            if self.config.keep_originals && self.config.quantization != QuantizationType::None {
                self.config.rerank_multiplier
            } else {
                1
            };
        // Buffer for stale entries (2x)
        let search_k = k * multiplier * 2;

        // Use HNSW if available
        let results: Vec<(types::InternalId, f32)> = if let Some(index) = &self.index {
            // HNSW Search
            index.search(query, search_k, &self.storage.view(), filter)?
        } else {
            // Fallback to Brute Force
            let storage_view = self.storage.view();
            let quantized_query = self.storage.quantize_query(query);

            let mut candidates: Vec<(types::InternalId, f32)> = self
                .storage
                .all_internal_ids()
                .into_iter()
                .filter(|&id| {
                    if let Some(f) = filter {
                        if let Some(meta) = self.storage.get_metadata(id) {
                            f.matches(&meta)
                        } else {
                            false
                        }
                    } else {
                        true
                    }
                })
                .filter_map(|id| {
                    storage_view
                        .distance_quantized(query, &quantized_query, id, metric)
                        .map(|dist| (id, dist))
                })
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.into_iter().take(search_k).collect()
        };

        // Filter stale results
        let valid_candidates: Vec<(types::InternalId, f32)> = results
            .into_iter()
            .filter(|(internal_id, _)| {
                if let Some(ext_id) = self.storage.get_external_id(*internal_id) {
                    if let Some(current) = self.storage.get_internal_id(&ext_id) {
                        return current == *internal_id;
                    }
                }
                false
            })
            .collect();

        // If re-ranking is enabled
        let final_results: Vec<(types::InternalId, f32)> =
            if self.config.keep_originals && self.config.quantization != QuantizationType::None {
                let k_rerank = k * self.config.rerank_multiplier;
                let top_candidates: Vec<_> = valid_candidates.into_iter().take(k_rerank).collect();

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
                valid_candidates.into_iter().take(k).collect()
            };

        // Map to external IDs and fetch metadata
        let mapped: Vec<(VectorId, f32, Option<Value>)> = final_results
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

    /// Get approximate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage() + self.index.as_ref().map(|i| i.memory_usage()).unwrap_or(0)
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

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();

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

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 1, None).unwrap();
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

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();

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
            .search(&[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], 2, None)
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

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();

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
