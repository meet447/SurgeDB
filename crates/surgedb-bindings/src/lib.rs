#![allow(clippy::empty_line_after_doc_comments)]
//! SurgeDB UniFFI Bindings
//!
//! This crate provides a stable wrapper API for cross-language bindings.
//! The interface is designed to be stable across internal changes to surgedb-core.
//!
//! # Architecture
//! ```text
//! Python/Swift/Kotlin
//!         │
//!         ▼
//! ┌─────────────────────────┐
//! │   surgedb-bindings      │  ← Stable API (this crate)
//! │   (SurgeClient)         │
//! └───────────┬─────────────┘
//!             │
//!             ▼
//! ┌─────────────────────────┐
//! │     surgedb-core        │  ← Can change freely
//! │   (internal engine)     │
//! └─────────────────────────┘
//! ```

use parking_lot::RwLock;
use std::sync::Arc;

// Import the generated UniFFI scaffolding
uniffi::include_scaffolding!("surgedb");

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SurgeError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: u32, got: u32 },

    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },

    #[error("Duplicate vector ID: {id}")]
    DuplicateId { id: String },

    #[error("Index is empty, cannot search")]
    EmptyIndex,

    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    #[error("Storage error: {message}")]
    StorageError { message: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Collection not found: {name}")]
    CollectionNotFound { name: String },

    #[error("Duplicate collection: {name}")]
    DuplicateCollection { name: String },

    #[error("WAL corrupted: {message}")]
    WalCorrupted { message: String },

    #[error("Snapshot corrupted: {message}")]
    SnapshotCorrupted { message: String },

    #[error("Index corrupted: {message}")]
    IndexCorrupted { message: String },

    #[error("Checksum mismatch")]
    ChecksumMismatch,

    #[error("Unsupported version: {version}")]
    UnsupportedVersion { version: String },

    #[error("Capacity exceeded: {message}")]
    CapacityExceeded { message: String },

    #[error("Lock acquisition failed: {message}")]
    LockFailed { message: String },

    #[error("Operation cancelled")]
    Cancelled,
}

impl SurgeError {
    /// Get the error code for FFI interop
    pub fn error_code(&self) -> u32 {
        match self {
            SurgeError::DimensionMismatch { .. } => 1001,
            SurgeError::VectorNotFound { .. } => 1002,
            SurgeError::DuplicateId { .. } => 1003,
            SurgeError::EmptyIndex => 1004,
            SurgeError::InvalidConfig { .. } => 1100,
            SurgeError::StorageError { .. } => 1200,
            SurgeError::CollectionNotFound { .. } => 1201,
            SurgeError::DuplicateCollection { .. } => 1202,
            SurgeError::CapacityExceeded { .. } => 1203,
            SurgeError::IoError { .. } => 1300,
            SurgeError::WalCorrupted { .. } => 1301,
            SurgeError::SnapshotCorrupted { .. } => 1302,
            SurgeError::ChecksumMismatch => 1303,
            SurgeError::UnsupportedVersion { .. } => 1304,
            SurgeError::IndexCorrupted { .. } => 1400,
            SurgeError::SerializationError { .. } => 1500,
            SurgeError::LockFailed { .. } => 1600,
            SurgeError::Cancelled => 1601,
        }
    }

    /// Returns true if the error indicates data corruption
    pub fn is_corruption(&self) -> bool {
        matches!(
            self,
            SurgeError::WalCorrupted { .. }
                | SurgeError::SnapshotCorrupted { .. }
                | SurgeError::ChecksumMismatch
                | SurgeError::IndexCorrupted { .. }
        )
    }

    /// Returns true if the operation can be retried
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            SurgeError::EmptyIndex | SurgeError::LockFailed { .. } | SurgeError::Cancelled
        )
    }
}

// Convert from core errors to binding errors
impl From<surgedb_core::Error> for SurgeError {
    fn from(err: surgedb_core::Error) -> Self {
        match err {
            surgedb_core::Error::DimensionMismatch { expected, got } => {
                SurgeError::DimensionMismatch {
                    expected: expected as u32,
                    got: got as u32,
                }
            }
            surgedb_core::Error::VectorNotFound(id) => SurgeError::VectorNotFound { id },
            surgedb_core::Error::DuplicateId(id) => SurgeError::DuplicateId { id },
            surgedb_core::Error::EmptyIndex => SurgeError::EmptyIndex,
            surgedb_core::Error::InvalidConfig(msg) => SurgeError::InvalidConfig { message: msg },
            surgedb_core::Error::InvalidHnswParam {
                param,
                value,
                reason,
            } => SurgeError::InvalidConfig {
                message: format!("{} = {}: {}", param, value, reason),
            },
            surgedb_core::Error::Storage(msg) => SurgeError::StorageError { message: msg },
            surgedb_core::Error::CollectionNotFound(name) => {
                SurgeError::CollectionNotFound { name }
            }
            surgedb_core::Error::DuplicateCollection(name) => {
                SurgeError::DuplicateCollection { name }
            }
            surgedb_core::Error::CapacityExceeded { message } => {
                SurgeError::CapacityExceeded { message }
            }
            surgedb_core::Error::Io(e) => SurgeError::IoError {
                message: e.to_string(),
            },
            surgedb_core::Error::WalCorrupted { message } => SurgeError::WalCorrupted { message },
            surgedb_core::Error::SnapshotCorrupted { message } => {
                SurgeError::SnapshotCorrupted { message }
            }
            surgedb_core::Error::ChecksumMismatch { .. } => {
                // Store in message for FFI (can't expose complex types easily)
                SurgeError::ChecksumMismatch
            }
            surgedb_core::Error::UnsupportedVersion { version, .. } => {
                SurgeError::UnsupportedVersion {
                    version: version.to_string(),
                }
            }
            surgedb_core::Error::IndexCorrupted { message } => {
                SurgeError::IndexCorrupted { message }
            }
            surgedb_core::Error::IdMappingCorrupted {
                internal_id,
                external_id,
            } => SurgeError::IndexCorrupted {
                message: format!(
                    "ID mapping corrupted: internal={}, external={}",
                    internal_id, external_id
                ),
            },
            surgedb_core::Error::Serialization { message } => {
                SurgeError::SerializationError { message }
            }
            surgedb_core::Error::Deserialization { message } => {
                SurgeError::SerializationError { message }
            }
            surgedb_core::Error::LockFailed { message } => SurgeError::LockFailed { message },
            surgedb_core::Error::Cancelled => SurgeError::Cancelled,
        }
    }
}

// =============================================================================
// Enums
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl From<DistanceMetric> for surgedb_core::DistanceMetric {
    fn from(val: DistanceMetric) -> Self {
        match val {
            DistanceMetric::Cosine => surgedb_core::DistanceMetric::Cosine,
            DistanceMetric::Euclidean => surgedb_core::DistanceMetric::Euclidean,
            DistanceMetric::DotProduct => surgedb_core::DistanceMetric::DotProduct,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    None,
    SQ8,
    Binary,
}

impl From<Quantization> for surgedb_core::QuantizationType {
    fn from(val: Quantization) -> Self {
        match val {
            Quantization::None => surgedb_core::QuantizationType::None,
            Quantization::SQ8 => surgedb_core::QuantizationType::SQ8,
            Quantization::Binary => surgedb_core::QuantizationType::Binary,
        }
    }
}

// =============================================================================
// Data Types
// =============================================================================

#[derive(Debug, Clone)]
pub struct SurgeConfig {
    pub dimensions: u32,
    pub distance_metric: DistanceMetric,
    pub quantization: Quantization,
    pub persistent: bool,
    pub data_path: Option<String>,
}

impl Default for SurgeConfig {
    fn default() -> Self {
        Self {
            dimensions: 384,
            distance_metric: DistanceMetric::Cosine,
            quantization: Quantization::None,
            persistent: false,
            data_path: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub vector_count: u64,
    pub dimensions: u32,
    pub memory_usage_bytes: u64,
    pub compression_ratio: f32,
}

// =============================================================================
// Filter Types
// =============================================================================

#[derive(Debug, Clone)]
pub enum SearchFilter {
    Exact {
        field: String,
        value_json: String,
    },
    OneOf {
        field: String,
        values_json: Vec<String>,
    },
    And {
        filters: Vec<SearchFilter>,
    },
    Or {
        filters: Vec<SearchFilter>,
    },
}

impl SearchFilter {
    fn to_core_filter(&self) -> Result<surgedb_core::filter::Filter, SurgeError> {
        match self {
            SearchFilter::Exact { field, value_json } => {
                let value: serde_json::Value = serde_json::from_str(value_json).map_err(|e| {
                    SurgeError::SerializationError {
                        message: e.to_string(),
                    }
                })?;
                Ok(surgedb_core::filter::Filter::Exact(field.clone(), value))
            }
            SearchFilter::OneOf { field, values_json } => {
                let values: Result<Vec<serde_json::Value>, _> = values_json
                    .iter()
                    .map(|v| serde_json::from_str(v))
                    .collect();
                let values = values.map_err(|e| SurgeError::SerializationError {
                    message: e.to_string(),
                })?;
                Ok(surgedb_core::filter::Filter::OneOf(field.clone(), values))
            }
            SearchFilter::And { filters } => {
                let core_filters: Result<Vec<_>, _> =
                    filters.iter().map(|f| f.to_core_filter()).collect();
                Ok(surgedb_core::filter::Filter::And(core_filters?))
            }
            SearchFilter::Or { filters } => {
                let core_filters: Result<Vec<_>, _> =
                    filters.iter().map(|f| f.to_core_filter()).collect();
                Ok(surgedb_core::filter::Filter::Or(core_filters?))
            }
        }
    }
}

// =============================================================================
// Internal Database Wrapper
// =============================================================================

type BatchItems = Vec<(surgedb_core::VectorId, Vec<f32>, Option<serde_json::Value>)>;

/// Internal enum to hold different database types
enum DbInner {
    InMemory(surgedb_core::VectorDb),
    Quantized(surgedb_core::QuantizedVectorDb),
    Persistent(surgedb_core::PersistentVectorDb),
}

// =============================================================================
// Main Client (Public API)
// =============================================================================

/// The main SurgeDB client - thread-safe wrapper around the core database.
///
/// This is the STABLE public API exposed to Python/Swift/Kotlin.
/// Internal changes to surgedb-core won't affect this interface.
pub struct SurgeClient {
    inner: Arc<RwLock<DbInner>>,
    #[allow(dead_code)]
    config: SurgeConfig,
}

impl SurgeClient {
    /// Create a new in-memory database with default settings
    pub fn new_in_memory(dimensions: u32) -> Result<Self, SurgeError> {
        let core_config = surgedb_core::Config {
            dimensions: dimensions as usize,
            distance_metric: surgedb_core::DistanceMetric::Cosine,
            ..Default::default()
        };

        let db = surgedb_core::VectorDb::new(core_config)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(DbInner::InMemory(db))),
            config: SurgeConfig {
                dimensions,
                ..Default::default()
            },
        })
    }

    /// Open a database with the given configuration
    pub fn open(path: String, config: SurgeConfig) -> Result<Self, SurgeError> {
        let inner = if config.persistent {
            // Persistent database
            let core_config = surgedb_core::PersistentConfig {
                dimensions: config.dimensions as usize,
                distance_metric: config.distance_metric.into(),
                ..Default::default()
            };
            let db = surgedb_core::PersistentVectorDb::open(&path, core_config)?;
            DbInner::Persistent(db)
        } else if config.quantization != Quantization::None {
            // Quantized in-memory database
            let core_config = surgedb_core::QuantizedConfig {
                dimensions: config.dimensions as usize,
                distance_metric: config.distance_metric.into(),
                quantization: config.quantization.into(),
                ..Default::default()
            };
            let db = surgedb_core::QuantizedVectorDb::new(core_config)?;
            DbInner::Quantized(db)
        } else {
            // Regular in-memory database
            let core_config = surgedb_core::Config {
                dimensions: config.dimensions as usize,
                distance_metric: config.distance_metric.into(),
                ..Default::default()
            };
            let db = surgedb_core::VectorDb::new(core_config)?;
            DbInner::InMemory(db)
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            config,
        })
    }

    /// Insert a vector with optional metadata
    pub fn insert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata_json: Option<String>,
    ) -> Result<(), SurgeError> {
        let metadata = parse_metadata(&metadata_json)?;
        let mut inner = self.inner.write();

        match &mut *inner {
            DbInner::InMemory(db) => db.insert(id, &vector, metadata)?,
            DbInner::Quantized(db) => db.insert(id, &vector, metadata)?,
            DbInner::Persistent(db) => db.insert(id, &vector, metadata)?,
        }

        Ok(())
    }

    /// Insert or update a vector
    pub fn upsert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata_json: Option<String>,
    ) -> Result<(), SurgeError> {
        let metadata = parse_metadata(&metadata_json)?;
        let mut inner = self.inner.write();

        match &mut *inner {
            DbInner::InMemory(db) => db.upsert(id, &vector, metadata)?,
            DbInner::Quantized(db) => db.upsert(id, &vector, metadata)?,
            DbInner::Persistent(db) => {
                // PersistentVectorDb doesn't have upsert, use delete + insert
                let _ = db.delete(id.clone());
                db.insert(id, &vector, metadata)?;
            }
        }

        Ok(())
    }

    /// Batch insert/upsert multiple vectors
    pub fn upsert_batch(&self, entries: Vec<VectorEntry>) -> Result<(), SurgeError> {
        let items: Result<BatchItems, SurgeError> = entries
            .into_iter()
            .map(|e| {
                let metadata = parse_metadata(&e.metadata_json)?;
                Ok((surgedb_core::VectorId::from(e.id), e.vector, metadata))
            })
            .collect();
        let items = items?;

        let mut inner = self.inner.write();

        match &mut *inner {
            DbInner::InMemory(db) => db.upsert_batch(items)?,
            DbInner::Quantized(db) => db.upsert_batch(items)?,
            DbInner::Persistent(db) => {
                // Persistent doesn't have batch, fall back to individual inserts
                for (id, vector, metadata) in items {
                    let _ = db.delete(id.clone());
                    db.insert(id, &vector, metadata)?;
                }
            }
        }

        Ok(())
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: String) -> Result<bool, SurgeError> {
        let mut inner = self.inner.write();

        let deleted = match &mut *inner {
            DbInner::InMemory(db) => db.delete(id.clone())?,
            DbInner::Quantized(db) => db.delete(id.clone())?,
            DbInner::Persistent(db) => db.delete(id.clone())?,
        };

        Ok(deleted)
    }

    /// Get a vector by ID
    pub fn get(&self, id: String) -> Result<Option<VectorEntry>, SurgeError> {
        let inner = self.inner.read();

        let result = match &*inner {
            DbInner::InMemory(db) => db.get(&id)?,
            DbInner::Quantized(db) => db.get(&id)?,
            DbInner::Persistent(_) => {
                // PersistentVectorDb doesn't have get, return None for now
                // TODO: Add get method to PersistentVectorDb
                None
            }
        };

        Ok(result.map(|(vector, metadata)| VectorEntry {
            id,
            vector,
            metadata_json: metadata.map(|m| m.to_string()),
        }))
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: Vec<f32>, k: u32) -> Result<Vec<SearchResult>, SurgeError> {
        let inner = self.inner.read();

        let results = match &*inner {
            DbInner::InMemory(db) => db.search(&query, k as usize, None)?,
            DbInner::Quantized(db) => db.search(&query, k as usize, None)?,
            DbInner::Persistent(db) => db.search(&query, k as usize, None)?,
        };

        Ok(results
            .into_iter()
            .map(|(id, score, metadata)| SearchResult {
                id: id.to_string(),
                score,
                metadata_json: metadata.map(|m| m.to_string()),
            })
            .collect())
    }

    /// Search with metadata filter
    pub fn search_with_filter(
        &self,
        query: Vec<f32>,
        k: u32,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, SurgeError> {
        let core_filter = filter.to_core_filter()?;
        let inner = self.inner.read();

        let results = match &*inner {
            DbInner::InMemory(db) => db.search(&query, k as usize, Some(&core_filter))?,
            DbInner::Quantized(db) => db.search(&query, k as usize, Some(&core_filter))?,
            DbInner::Persistent(db) => db.search(&query, k as usize, Some(&core_filter))?,
        };

        Ok(results
            .into_iter()
            .map(|(id, score, metadata)| SearchResult {
                id: id.to_string(),
                score,
                metadata_json: metadata.map(|m| m.to_string()),
            })
            .collect())
    }

    /// List vector IDs with pagination
    pub fn list(&self, offset: u32, limit: u32) -> Vec<String> {
        let inner = self.inner.read();

        let ids = match &*inner {
            DbInner::InMemory(db) => db.list(offset as usize, limit as usize),
            DbInner::Quantized(db) => db.list(offset as usize, limit as usize),
            DbInner::Persistent(_) => vec![], // TODO: Add list to PersistentVectorDb
        };

        ids.into_iter().map(|id| id.to_string()).collect()
    }

    /// Get number of vectors
    pub fn len(&self) -> u64 {
        let inner = self.inner.read();

        match &*inner {
            DbInner::InMemory(db) => db.len() as u64,
            DbInner::Quantized(db) => db.len() as u64,
            DbInner::Persistent(db) => db.len() as u64,
        }
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.read();

        match &*inner {
            DbInner::InMemory(db) => db.is_empty(),
            DbInner::Quantized(db) => db.is_empty(),
            DbInner::Persistent(db) => db.is_empty(),
        }
    }

    /// Get database statistics
    pub fn stats(&self) -> DatabaseStats {
        let inner = self.inner.read();

        match &*inner {
            DbInner::InMemory(db) => DatabaseStats {
                vector_count: db.len() as u64,
                dimensions: db.config().dimensions as u32,
                memory_usage_bytes: db.memory_usage() as u64,
                compression_ratio: 1.0,
            },
            DbInner::Quantized(db) => DatabaseStats {
                vector_count: db.len() as u64,
                dimensions: db.config().dimensions as u32,
                memory_usage_bytes: db.memory_usage() as u64,
                compression_ratio: db.compression_ratio(),
            },
            DbInner::Persistent(db) => DatabaseStats {
                vector_count: db.len() as u64,
                dimensions: db.config().dimensions as u32,
                memory_usage_bytes: 0, // TODO: Add memory_usage to PersistentVectorDb
                compression_ratio: 1.0,
            },
        }
    }

    /// Force checkpoint (for persistent databases)
    pub fn checkpoint(&self) -> Result<(), SurgeError> {
        let mut inner = self.inner.write();

        if let DbInner::Persistent(db) = &mut *inner {
            db.checkpoint()?;
        }

        Ok(())
    }

    /// Force sync to disk (for persistent databases)
    pub fn sync(&self) -> Result<(), SurgeError> {
        let mut inner = self.inner.write();

        if let DbInner::Persistent(db) = &mut *inner {
            db.sync()?;
        }

        Ok(())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn parse_metadata(json: &Option<String>) -> Result<Option<serde_json::Value>, SurgeError> {
    match json {
        Some(s) => {
            let value: serde_json::Value =
                serde_json::from_str(s).map_err(|e| SurgeError::SerializationError {
                    message: e.to_string(),
                })?;
            Ok(Some(value))
        }
        None => Ok(None),
    }
}

// =============================================================================
// Module-level Functions
// =============================================================================

/// Get library version
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get system info for debugging
pub fn system_info() -> String {
    format!(
        "SurgeDB Bindings v{}\nTarget: {}-{}\nFeatures: SIMD-optimized",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_basic() {
        let client = SurgeClient::new_in_memory(4).unwrap();

        // Insert
        client
            .insert("vec1".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        client
            .insert("vec2".to_string(), vec![0.0, 1.0, 0.0, 0.0], None)
            .unwrap();

        // Search
        let results = client.search(vec![1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "vec1");
    }

    #[test]
    fn test_with_metadata() {
        let client = SurgeClient::new_in_memory(4).unwrap();

        client
            .insert(
                "vec1".to_string(),
                vec![1.0, 0.0, 0.0, 0.0],
                Some(r#"{"category": "test"}"#.to_string()),
            )
            .unwrap();

        let results = client.search(vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert!(results[0].metadata_json.is_some());
    }

    #[test]
    fn test_delete() {
        let client = SurgeClient::new_in_memory(4).unwrap();

        client
            .insert("vec1".to_string(), vec![1.0, 0.0, 0.0, 0.0], None)
            .unwrap();

        assert_eq!(client.len(), 1);
        assert!(client.delete("vec1".to_string()).unwrap());
        // Note: len() might still return 1 due to soft delete
    }

    #[test]
    fn test_stats() {
        let client = SurgeClient::new_in_memory(128).unwrap();

        for i in 0..100 {
            let vector: Vec<f32> = (0..128).map(|j| ((i * j) as f32).sin()).collect();
            client.insert(format!("v{}", i), vector, None).unwrap();
        }

        let stats = client.stats();
        assert_eq!(stats.vector_count, 100);
        assert_eq!(stats.dimensions, 128);
        assert!(stats.memory_usage_bytes > 0);
    }
}
