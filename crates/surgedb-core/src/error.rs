//! Error types for SurgeDB
//!
//! This module provides strongly-typed error variants for all SurgeDB operations.
//! Errors are categorized by domain to make error handling and recovery straightforward.

use thiserror::Error;

/// Result type alias for SurgeDB operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for SurgeDB operations
///
/// These errors are designed to be:
/// - Strongly typed for precise error handling
/// - FFI/WASM friendly for clean cross-language propagation
/// - Actionable with clear recovery paths
#[derive(Debug, Error)]
pub enum Error {
    // =========================================================================
    // Vector/Dimension Errors
    // =========================================================================
    /// Vector dimensions don't match the database configuration
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Attempted to retrieve a vector that doesn't exist
    #[error("Vector not found: {0}")]
    VectorNotFound(String),

    /// Attempted to insert a vector with an ID that already exists
    #[error("Duplicate vector ID: {0}")]
    DuplicateId(String),

    /// Search was attempted on an empty index
    #[error("Index is empty, cannot search")]
    EmptyIndex,

    // =========================================================================
    // Configuration Errors
    // =========================================================================
    /// Invalid configuration parameter
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Invalid HNSW parameters (M, ef_construction, etc.)
    #[error("Invalid HNSW parameter: {param} = {value}, {reason}")]
    InvalidHnswParam {
        param: &'static str,
        value: String,
        reason: &'static str,
    },

    // =========================================================================
    // Storage/Collection Errors
    // =========================================================================
    /// Generic storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Collection not found in multi-collection database
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    /// Attempted to create a collection that already exists
    #[error("Duplicate collection: {0}")]
    DuplicateCollection(String),

    /// Storage capacity exceeded
    #[error("Storage capacity exceeded: {message}")]
    CapacityExceeded { message: String },

    // =========================================================================
    // Persistence/WAL Errors
    // =========================================================================
    /// Generic I/O error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// WAL (Write-Ahead Log) corruption detected
    #[error("WAL corrupted: {message}")]
    WalCorrupted { message: String },

    /// Snapshot file is corrupted or invalid
    #[error("Snapshot corrupted: {message}")]
    SnapshotCorrupted { message: String },

    /// Checksum verification failed
    #[error("Checksum mismatch: expected {expected:#010x}, got {actual:#010x}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    /// Unsupported file format version
    #[error("Unsupported format version: {version}, supported: {supported}")]
    UnsupportedVersion {
        version: u8,
        supported: &'static str,
    },

    // =========================================================================
    // Index Errors
    // =========================================================================
    /// HNSW index is corrupted (graph invariants violated)
    #[error("Index corrupted: {message}")]
    IndexCorrupted { message: String },

    /// Internal ID mapping is inconsistent
    #[error("ID mapping corrupted: internal_id={internal_id}, external_id={external_id}")]
    IdMappingCorrupted {
        internal_id: usize,
        external_id: String,
    },

    // =========================================================================
    // Serialization Errors
    // =========================================================================
    /// Serialization failed
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// Deserialization failed
    #[error("Deserialization error: {message}")]
    Deserialization { message: String },

    // =========================================================================
    // Concurrency Errors
    // =========================================================================
    /// Lock acquisition failed (timeout or poisoned)
    #[error("Lock acquisition failed: {message}")]
    LockFailed { message: String },

    /// Operation was cancelled
    #[error("Operation cancelled")]
    Cancelled,
}

impl Error {
    /// Returns true if this error is recoverable (can retry the operation)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Error::EmptyIndex | Error::LockFailed { .. } | Error::Cancelled
        )
    }

    /// Returns true if this error indicates data corruption
    pub fn is_corruption(&self) -> bool {
        matches!(
            self,
            Error::WalCorrupted { .. }
                | Error::SnapshotCorrupted { .. }
                | Error::ChecksumMismatch { .. }
                | Error::IndexCorrupted { .. }
                | Error::IdMappingCorrupted { .. }
        )
    }

    /// Returns true if this error is a user/input error
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            Error::DimensionMismatch { .. }
                | Error::VectorNotFound(_)
                | Error::DuplicateId(_)
                | Error::InvalidConfig(_)
                | Error::InvalidHnswParam { .. }
                | Error::CollectionNotFound(_)
                | Error::DuplicateCollection(_)
        )
    }

    /// Get an error code for FFI/WASM interop
    pub fn error_code(&self) -> u32 {
        match self {
            // Vector errors: 1000-1099
            Error::DimensionMismatch { .. } => 1001,
            Error::VectorNotFound(_) => 1002,
            Error::DuplicateId(_) => 1003,
            Error::EmptyIndex => 1004,

            // Config errors: 1100-1199
            Error::InvalidConfig(_) => 1100,
            Error::InvalidHnswParam { .. } => 1101,

            // Storage errors: 1200-1299
            Error::Storage(_) => 1200,
            Error::CollectionNotFound(_) => 1201,
            Error::DuplicateCollection(_) => 1202,
            Error::CapacityExceeded { .. } => 1203,

            // Persistence errors: 1300-1399
            Error::Io(_) => 1300,
            Error::WalCorrupted { .. } => 1301,
            Error::SnapshotCorrupted { .. } => 1302,
            Error::ChecksumMismatch { .. } => 1303,
            Error::UnsupportedVersion { .. } => 1304,

            // Index errors: 1400-1499
            Error::IndexCorrupted { .. } => 1400,
            Error::IdMappingCorrupted { .. } => 1401,

            // Serialization errors: 1500-1599
            Error::Serialization { .. } => 1500,
            Error::Deserialization { .. } => 1501,

            // Concurrency errors: 1600-1699
            Error::LockFailed { .. } => 1600,
            Error::Cancelled => 1601,
        }
    }
}

// Convenient conversion from bincode errors
impl From<Box<bincode::ErrorKind>> for Error {
    fn from(err: Box<bincode::ErrorKind>) -> Self {
        Error::Serialization {
            message: err.to_string(),
        }
    }
}

// Convenient conversion from serde_json errors
impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes_unique() {
        // Ensure error codes don't overlap
        let errors = vec![
            Error::DimensionMismatch {
                expected: 384,
                got: 128,
            },
            Error::VectorNotFound("test".into()),
            Error::DuplicateId("test".into()),
            Error::EmptyIndex,
            Error::InvalidConfig("test".into()),
            Error::Storage("test".into()),
            Error::CollectionNotFound("test".into()),
            Error::DuplicateCollection("test".into()),
            Error::WalCorrupted {
                message: "test".into(),
            },
            Error::IndexCorrupted {
                message: "test".into(),
            },
        ];

        let codes: Vec<u32> = errors.iter().map(|e| e.error_code()).collect();
        let unique: std::collections::HashSet<u32> = codes.iter().cloned().collect();
        assert_eq!(codes.len(), unique.len(), "Error codes must be unique");
    }

    #[test]
    fn test_error_classification() {
        assert!(Error::EmptyIndex.is_recoverable());
        assert!(!Error::DimensionMismatch {
            expected: 1,
            got: 2
        }
        .is_recoverable());

        assert!(Error::WalCorrupted {
            message: "test".into()
        }
        .is_corruption());
        assert!(!Error::EmptyIndex.is_corruption());

        assert!(Error::DimensionMismatch {
            expected: 1,
            got: 2
        }
        .is_user_error());
        assert!(!Error::WalCorrupted {
            message: "test".into()
        }
        .is_user_error());
    }
}
