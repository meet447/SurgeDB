//! SurgeDB WebAssembly Bindings
//!
//! This crate provides JavaScript/TypeScript bindings for SurgeDB,
//! enabling in-browser vector search with sub-millisecond latency.
//!
//! # Example (JavaScript)
//! ```javascript
//! import init, { SurgeDB } from 'surgedb-wasm';
//!
//! await init();
//!
//! const db = new SurgeDB(384);  // 384 dimensions
//! db.insert("doc1", new Float32Array([...]), { title: "Hello" });
//!
//! const results = db.search(new Float32Array([...]), 10);
//! console.log(results);  // [{ id: "doc1", score: 0.95, metadata: {...} }, ...]
//! ```

use serde::{Deserialize, Serialize};
use surgedb_core::VectorId;
use wasm_bindgen::prelude::*;

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// =============================================================================
// Error Handling
// =============================================================================

/// Error codes matching the core error types
#[wasm_bindgen]
#[allow(non_snake_case)]
pub struct SurgeErrorCode;

#[wasm_bindgen]
#[allow(non_snake_case)]
impl SurgeErrorCode {
    #[wasm_bindgen(getter)]
    pub fn DIMENSION_MISMATCH() -> u32 {
        1001
    }
    #[wasm_bindgen(getter)]
    pub fn VECTOR_NOT_FOUND() -> u32 {
        1002
    }
    #[wasm_bindgen(getter)]
    pub fn DUPLICATE_ID() -> u32 {
        1003
    }
    #[wasm_bindgen(getter)]
    pub fn EMPTY_INDEX() -> u32 {
        1004
    }
    #[wasm_bindgen(getter)]
    pub fn INVALID_CONFIG() -> u32 {
        1100
    }
    #[wasm_bindgen(getter)]
    pub fn STORAGE_ERROR() -> u32 {
        1200
    }
    #[wasm_bindgen(getter)]
    pub fn IO_ERROR() -> u32 {
        1300
    }
    #[wasm_bindgen(getter)]
    pub fn SERIALIZATION_ERROR() -> u32 {
        1500
    }
}

#[derive(Debug, Serialize)]
struct SurgeErrorInfo {
    code: u32,
    name: String,
    message: String,
    recoverable: bool,
    corruption: bool,
}

#[derive(Debug)]
pub struct SurgeError {
    code: u32,
    name: String,
    message: String,
    recoverable: bool,
    corruption: bool,
}

impl SurgeError {
    fn new(code: u32, name: &str, message: String, recoverable: bool, corruption: bool) -> Self {
        Self {
            code,
            name: name.to_string(),
            message,
            recoverable,
            corruption,
        }
    }
}

impl From<surgedb_core::Error> for SurgeError {
    fn from(err: surgedb_core::Error) -> Self {
        let code = err.error_code();
        let recoverable = err.is_recoverable();
        let corruption = err.is_corruption();
        let message = err.to_string();

        let name = match &err {
            surgedb_core::Error::DimensionMismatch { .. } => "DimensionMismatch",
            surgedb_core::Error::VectorNotFound(_) => "VectorNotFound",
            surgedb_core::Error::DuplicateId(_) => "DuplicateId",
            surgedb_core::Error::EmptyIndex => "EmptyIndex",
            surgedb_core::Error::InvalidConfig(_) => "InvalidConfig",
            surgedb_core::Error::InvalidHnswParam { .. } => "InvalidHnswParam",
            surgedb_core::Error::Storage(_) => "StorageError",
            surgedb_core::Error::CollectionNotFound(_) => "CollectionNotFound",
            surgedb_core::Error::DuplicateCollection(_) => "DuplicateCollection",
            surgedb_core::Error::CapacityExceeded { .. } => "CapacityExceeded",
            surgedb_core::Error::Io(_) => "IoError",
            surgedb_core::Error::WalCorrupted { .. } => "WalCorrupted",
            surgedb_core::Error::SnapshotCorrupted { .. } => "SnapshotCorrupted",
            surgedb_core::Error::ChecksumMismatch { .. } => "ChecksumMismatch",
            surgedb_core::Error::UnsupportedVersion { .. } => "UnsupportedVersion",
            surgedb_core::Error::IndexCorrupted { .. } => "IndexCorrupted",
            surgedb_core::Error::IdMappingCorrupted { .. } => "IdMappingCorrupted",
            surgedb_core::Error::Serialization { .. } => "SerializationError",
            surgedb_core::Error::Deserialization { .. } => "DeserializationError",
            surgedb_core::Error::LockFailed { .. } => "LockFailed",
            surgedb_core::Error::Cancelled => "Cancelled",
        };

        SurgeError::new(code, name, message, recoverable, corruption)
    }
}

impl From<SurgeError> for JsValue {
    fn from(err: SurgeError) -> Self {
        // Create a proper JavaScript Error object with additional properties
        let error_info = SurgeErrorInfo {
            code: err.code,
            name: err.name.clone(),
            message: err.message.clone(),
            recoverable: err.recoverable,
            corruption: err.corruption,
        };

        // Try to create a structured error, fall back to string
        match serde_wasm_bindgen::to_value(&error_info) {
            Ok(obj) => obj,
            Err(_) => JsValue::from_str(&err.message),
        }
    }
}

impl From<serde_json::Error> for SurgeError {
    fn from(err: serde_json::Error) -> Self {
        SurgeError::new(1500, "SerializationError", err.to_string(), false, false)
    }
}

// =============================================================================
// Data Types
// =============================================================================

#[derive(Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct Stats {
    pub vector_count: usize,
    pub dimensions: usize,
    pub memory_usage_bytes: usize,
}

// =============================================================================
// Main Database Class
// =============================================================================

/// SurgeDB - High-performance vector database for the browser
///
/// Create an in-memory vector database optimized for semantic search.
#[wasm_bindgen]
pub struct SurgeDB {
    inner: surgedb_core::VectorDb,
}

#[wasm_bindgen]
impl SurgeDB {
    /// Create a new in-memory vector database
    ///
    /// @param dimensions - The dimensionality of vectors (e.g., 384 for MiniLM, 768 for BERT)
    #[wasm_bindgen(constructor)]
    pub fn new(dimensions: u32) -> Result<SurgeDB, JsValue> {
        let config = surgedb_core::Config {
            dimensions: dimensions as usize,
            distance_metric: surgedb_core::DistanceMetric::Cosine,
            ..Default::default()
        };

        let inner = surgedb_core::VectorDb::new(config).map_err(SurgeError::from)?;

        Ok(SurgeDB { inner })
    }

    /// Insert a vector with optional metadata
    ///
    /// @param id - Unique identifier for the vector
    /// @param vector - Float32Array of the embedding
    /// @param metadata - Optional JSON metadata object
    #[wasm_bindgen]
    pub fn insert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: JsValue,
    ) -> Result<(), JsValue> {
        // Debug logging
        web_sys::console::log_2(&"Inserting metadata:".into(), &metadata);

        let meta = if metadata.is_undefined() || metadata.is_null() {
            None
        } else {
            let parsed: serde_json::Value = serde_wasm_bindgen::from_value(metadata.clone())
                .map_err(|e| JsValue::from_str(&format!("Metadata parse error: {}", e)))?;

            // Log what we parsed
            let s = serde_json::to_string(&parsed).unwrap_or_default();
            web_sys::console::log_1(&format!("Parsed metadata: {}", s).into());

            Some(parsed)
        };

        self.inner
            .insert(id, &vector, meta)
            .map_err(SurgeError::from)?;

        Ok(())
    }

    /// Insert or update a vector
    ///
    /// @param id - Unique identifier for the vector
    /// @param vector - Float32Array of the embedding
    /// @param metadata - Optional JSON metadata object
    #[wasm_bindgen]
    pub fn upsert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: JsValue,
    ) -> Result<(), JsValue> {
        let meta = if metadata.is_undefined() || metadata.is_null() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value(metadata)?)
        };

        self.inner
            .upsert(id, &vector, meta)
            .map_err(SurgeError::from)?;

        Ok(())
    }

    /// Insert or update multiple vectors in a single operation
    ///
    /// @param entries - Array of { id, vector, metadata } objects
    #[wasm_bindgen(js_name = upsertBatch)]
    pub fn upsert_batch(&mut self, entries: JsValue) -> Result<(), JsValue> {
        let items: Vec<VectorEntry> = serde_wasm_bindgen::from_value(entries)?;

        let core_items: Vec<(VectorId, Vec<f32>, Option<serde_json::Value>)> = items
            .into_iter()
            .map(|e| (VectorId::from(e.id), e.vector, e.metadata))
            .collect();

        self.inner
            .upsert_batch(core_items)
            .map_err(SurgeError::from)?;

        Ok(())
    }

    /// Delete a vector by ID
    ///
    /// @param id - The ID of the vector to delete
    /// @returns true if the vector was found and deleted
    #[wasm_bindgen]
    pub fn delete(&mut self, id: String) -> Result<bool, JsValue> {
        self.inner
            .delete(id)
            .map_err(|e| SurgeError::from(e).into())
    }

    /// Get a vector by ID
    ///
    /// @param id - The ID of the vector
    /// @returns { id, vector, metadata } or undefined if not found
    #[wasm_bindgen]
    pub fn get(&self, id: String) -> Result<JsValue, JsValue> {
        let result = self.inner.get(&id).map_err(SurgeError::from)?;

        match result {
            Some((vector, metadata)) => {
                let entry = VectorEntry {
                    id,
                    vector,
                    metadata,
                };
                serde_wasm_bindgen::to_value(&entry).map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Search for the k nearest neighbors
    ///
    /// @param query - Float32Array query vector
    /// @param k - Number of results to return
    /// @returns Array of { id, score, metadata } objects
    #[wasm_bindgen]
    pub fn search(&self, query: Vec<f32>, k: u32) -> Result<JsValue, JsValue> {
        let results = self
            .inner
            .search(&query, k as usize, None)
            .map_err(SurgeError::from)?;

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(id, score, metadata)| {
                // Debug log removed
                SearchResult {
                    id: id.to_string(),
                    score,
                    metadata,
                }
            })
            .collect();

        // Configure serializer to convert Maps to plain Objects (fixes JSON.stringify issue)
        let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);

        // We use serialize(&serializer) instead of to_value
        use serde::Serialize;
        search_results
            .serialize(&serializer)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the number of vectors in the database
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the database is empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get database statistics
    #[wasm_bindgen]
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let stats = Stats {
            vector_count: self.inner.len(),
            dimensions: self.inner.config().dimensions,
            memory_usage_bytes: self.inner.memory_usage(),
        };

        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// =============================================================================
// Quantized Database (for larger datasets)
// =============================================================================

/// SurgeDB with SQ8 quantization for 4x memory reduction
#[wasm_bindgen]
pub struct SurgeDBQuantized {
    inner: surgedb_core::QuantizedVectorDb,
}

#[wasm_bindgen]
impl SurgeDBQuantized {
    /// Create a new quantized vector database (4x memory reduction)
    ///
    /// @param dimensions - The dimensionality of vectors
    #[wasm_bindgen(constructor)]
    pub fn new(dimensions: u32) -> Result<SurgeDBQuantized, JsValue> {
        let config = surgedb_core::QuantizedConfig {
            dimensions: dimensions as usize,
            distance_metric: surgedb_core::DistanceMetric::Cosine,
            quantization: surgedb_core::QuantizationType::SQ8,
            ..Default::default()
        };

        let inner = surgedb_core::QuantizedVectorDb::new(config).map_err(SurgeError::from)?;

        Ok(SurgeDBQuantized { inner })
    }

    /// Insert a vector with optional metadata
    #[wasm_bindgen]
    pub fn insert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: JsValue,
    ) -> Result<(), JsValue> {
        let meta = if metadata.is_undefined() || metadata.is_null() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value(metadata)?)
        };

        self.inner
            .insert(id, &vector, meta)
            .map_err(SurgeError::from)?;

        Ok(())
    }

    /// Insert or update a vector
    #[wasm_bindgen]
    pub fn upsert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: JsValue,
    ) -> Result<(), JsValue> {
        let meta = if metadata.is_undefined() || metadata.is_null() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value(metadata)?)
        };

        self.inner
            .upsert(id, &vector, meta)
            .map_err(SurgeError::from)?;

        Ok(())
    }

    /// Insert or update multiple vectors in a single operation
    #[wasm_bindgen(js_name = upsertBatch)]
    pub fn upsert_batch(&mut self, entries: JsValue) -> Result<(), JsValue> {
        let items: Vec<VectorEntry> = serde_wasm_bindgen::from_value(entries)?;

        let core_items: Vec<(VectorId, Vec<f32>, Option<serde_json::Value>)> = items
            .into_iter()
            .map(|e| (VectorId::from(e.id), e.vector, e.metadata))
            .collect();

        self.inner
            .upsert_batch(core_items)
            .map_err(SurgeError::from)?;

        Ok(())
    }

    /// Delete a vector by ID

    #[wasm_bindgen]
    pub fn delete(&mut self, id: String) -> Result<bool, JsValue> {
        self.inner
            .delete(id)
            .map_err(|e| SurgeError::from(e).into())
    }

    /// Get a vector by ID
    #[wasm_bindgen]
    pub fn get(&self, id: String) -> Result<JsValue, JsValue> {
        let result = self.inner.get(&id).map_err(SurgeError::from)?;

        match result {
            Some((vector, metadata)) => {
                let entry = VectorEntry {
                    id,
                    vector,
                    metadata,
                };
                serde_wasm_bindgen::to_value(&entry).map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Search for the k nearest neighbors
    #[wasm_bindgen]
    pub fn search(&self, query: Vec<f32>, k: u32) -> Result<JsValue, JsValue> {
        let results = self
            .inner
            .search(&query, k as usize, None)
            .map_err(SurgeError::from)?;

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(id, score, metadata)| SearchResult {
                id: id.to_string(),
                score,
                metadata,
            })
            .collect();

        serde_wasm_bindgen::to_value(&search_results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the number of vectors
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get compression ratio
    #[wasm_bindgen(js_name = compressionRatio)]
    pub fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio()
    }

    /// Get statistics
    #[wasm_bindgen]
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let stats = Stats {
            vector_count: self.inner.len(),
            dimensions: self.inner.config().dimensions,
            memory_usage_bytes: self.inner.memory_usage(),
        };

        serde_wasm_bindgen::to_value(&stats).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get the SurgeDB version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if SIMD acceleration is enabled in this build
#[wasm_bindgen(js_name = hasSimd)]
pub fn has_simd() -> bool {
    cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn log(message: &str) {
    web_sys::console::log_1(&JsValue::from_str(message));
}
