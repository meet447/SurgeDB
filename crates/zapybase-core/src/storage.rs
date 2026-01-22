//! Vector storage implementation
//!
//! Provides efficient storage and retrieval of vectors with ID mapping.

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::types::{InternalId, VectorId};
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::HashMap;

/// Trait for vector storage backends
pub trait VectorStorageTrait {
    /// Get vector data for distance calculations
    /// Note: This copies the vector data, which is slow for repeated use.
    fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>>;

    /// Calculate distance between a stored vector and a query vector
    /// This is optimized to avoid allocation.
    fn distance(
        &self,
        internal_id: InternalId,
        query: &[f32],
        metric: DistanceMetric,
    ) -> Option<f32>;
}

/// In-memory vector storage with ID mapping
pub struct VectorStorage {
    /// Dimensionality of stored vectors
    dimensions: usize,

    /// Flat storage of all vectors (contiguous memory for cache efficiency)
    vectors: RwLock<Vec<f32>>,

    /// Map from external ID to internal ID
    id_to_internal: RwLock<HashMap<VectorId, InternalId>>,

    /// Map from internal ID to external ID
    internal_to_id: RwLock<Vec<VectorId>>,

    /// Optional metadata for each vector
    metadata: RwLock<HashMap<InternalId, Value>>,
}

impl VectorStorage {
    /// Create a new vector storage with the given dimensionality
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            vectors: RwLock::new(Vec::new()),
            id_to_internal: RwLock::new(HashMap::new()),
            internal_to_id: RwLock::new(Vec::new()),
            metadata: RwLock::new(HashMap::new()),
        }
    }

    /// Insert a vector and return its internal ID
    pub fn insert(
        &self,
        id: VectorId,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<InternalId> {
        self.insert_internal(id, vector, metadata, false)
    }

    /// Insert or update a vector (upsert)
    /// If the ID exists, it creates a new internal record and updates the mapping.
    /// The old internal record becomes inaccessible via ID lookup (stale).
    pub fn upsert(
        &self,
        id: VectorId,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<InternalId> {
        self.insert_internal(id, vector, metadata, true)
    }

    fn insert_internal(
        &self,
        id: VectorId,
        vector: &[f32],
        metadata: Option<Value>,
        allow_update: bool,
    ) -> Result<InternalId> {
        if vector.len() != self.dimensions {
            return Err(Error::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        let mut id_to_internal = self.id_to_internal.write();
        if !allow_update && id_to_internal.contains_key(&id) {
            return Err(Error::DuplicateId(id.to_string()));
        }

        let mut vectors = self.vectors.write();
        let mut internal_to_id = self.internal_to_id.write();
        let mut metadata_store = self.metadata.write();

        let internal_id = InternalId::from(internal_to_id.len());

        // Append vector to flat storage
        vectors.extend_from_slice(vector);

        // Update mappings
        id_to_internal.insert(id.clone(), internal_id);
        internal_to_id.push(id);

        // Store metadata if present
        if let Some(meta) = metadata {
            metadata_store.insert(internal_id, meta);
        }

        Ok(internal_id)
    }

    /// Get a vector by its internal ID
    #[inline]
    pub fn get(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        let vectors = self.vectors.read();
        let start = internal_id.as_usize() * self.dimensions;
        let end = start + self.dimensions;

        if end <= vectors.len() {
            Some(vectors[start..end].to_vec())
        } else {
            None
        }
    }

    /// Get a reference to vector data (for distance calculations)
    /// Returns a copy to avoid holding locks during computation
    #[inline]
    pub fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        self.get(internal_id)
    }

    /// Get metadata for a vector
    pub fn get_metadata(&self, internal_id: InternalId) -> Option<Value> {
        self.metadata.read().get(&internal_id).cloned()
    }

    /// Get internal ID from external ID
    pub fn get_internal_id(&self, id: &VectorId) -> Option<InternalId> {
        self.id_to_internal.read().get(id).copied()
    }

    /// Get external ID from internal ID
    pub fn get_external_id(&self, internal_id: InternalId) -> Option<VectorId> {
        let internal_to_id = self.internal_to_id.read();
        internal_to_id.get(internal_id.as_usize()).cloned()
    }

    /// Get the number of stored vectors
    pub fn len(&self) -> usize {
        self.internal_to_id.read().len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all internal IDs
    pub fn all_internal_ids(&self) -> Vec<InternalId> {
        let internal_to_id = self.internal_to_id.read();
        (0..internal_to_id.len()).map(InternalId::from).collect()
    }

    /// Get dimensionality
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get approximate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let vectors_size = self.vectors.read().capacity() * 4;
        // Approximation for maps:
        // id_to_internal: capacity * (size_of<String> + heap_overhead + size_of<InternalId> + map_overhead)
        // internal_to_id: capacity * (size_of<String> + heap_overhead)
        // Assuming avg string len 16 + overhead
        let count = self.len();
        let map_overhead = count * (64 + 4);
        let rev_map_overhead = count * 64;

        vectors_size + map_overhead + rev_map_overhead
    }

    /// Create a view of the storage that holds a read lock
    /// This is optimized for bulk operations like search
    pub fn view(&self) -> VectorStorageView<'_> {
        VectorStorageView {
            guard: self.vectors.read(),
            dimensions: self.dimensions,
        }
    }
}

/// A view into VectorStorage that holds a read lock on the data
/// This avoids repeated locking during search
pub struct VectorStorageView<'a> {
    guard: parking_lot::RwLockReadGuard<'a, Vec<f32>>,
    dimensions: usize,
}

impl<'a> VectorStorageTrait for VectorStorageView<'a> {
    fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        let start = internal_id.as_usize() * self.dimensions;
        let end = start + self.dimensions;

        if end <= self.guard.len() {
            Some(self.guard[start..end].to_vec())
        } else {
            None
        }
    }

    fn distance(
        &self,
        internal_id: InternalId,
        query: &[f32],
        metric: DistanceMetric,
    ) -> Option<f32> {
        let start = internal_id.as_usize() * self.dimensions;
        let end = start + self.dimensions;

        if end <= self.guard.len() {
            let vector_slice = &self.guard[start..end];
            Some(metric.distance(query, vector_slice))
        } else {
            None
        }
    }
}

/// Implement the trait for VectorStorage
impl VectorStorageTrait for VectorStorage {
    fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        self.get(internal_id)
    }

    fn distance(
        &self,
        internal_id: InternalId,
        query: &[f32],
        metric: DistanceMetric,
    ) -> Option<f32> {
        let vectors = self.vectors.read();
        let start = internal_id.as_usize() * self.dimensions;
        let end = start + self.dimensions;

        if end <= vectors.len() {
            let vector_slice = &vectors[start..end];
            Some(metric.distance(query, vector_slice))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let storage = VectorStorage::new(4);

        let id = VectorId::from("test");
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        let internal_id = storage.insert(id.clone(), &vector, None).unwrap();

        let retrieved = storage.get(internal_id).unwrap();
        assert_eq!(retrieved, vector);
    }

    #[test]
    fn test_duplicate_id() {
        let storage = VectorStorage::new(4);

        let id = VectorId::from("test");
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        storage.insert(id.clone(), &vector, None).unwrap();

        let result = storage.insert(id, &vector, None);
        assert!(matches!(result, Err(Error::DuplicateId(_))));
    }

    #[test]
    fn test_dimension_mismatch() {
        let storage = VectorStorage::new(4);

        let id = VectorId::from("test");
        let vector = vec![1.0, 2.0, 3.0]; // Only 3 dimensions

        let result = storage.insert(id, &vector, None);
        assert!(matches!(result, Err(Error::DimensionMismatch { .. })));
    }

    #[test]
    fn test_id_mapping() {
        let storage = VectorStorage::new(4);

        let id = VectorId::from("my-vector");
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        let internal_id = storage.insert(id.clone(), &vector, None).unwrap();

        assert_eq!(storage.get_internal_id(&id), Some(internal_id));
        assert_eq!(storage.get_external_id(internal_id), Some(id));
    }

    #[test]
    fn test_metadata() {
        let storage = VectorStorage::new(4);
        let id = VectorId::from("meta-test");
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let meta = serde_json::json!({"key": "value"});

        let internal_id = storage.insert(id, &vector, Some(meta.clone())).unwrap();
        assert_eq!(storage.get_metadata(internal_id), Some(meta));
    }
}
