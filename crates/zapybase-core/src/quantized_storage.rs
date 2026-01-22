//! Quantized vector storage implementation
//!
//! Provides memory-efficient storage using SQ8 or Binary quantization.

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::quantization::{BinaryQuantizer, QuantizationType, SQ8Metadata, SQ8Quantizer};
use crate::storage::VectorStorageTrait;
use crate::types::{InternalId, VectorId};
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::HashMap;

/// Quantized vector storage with configurable compression
pub struct QuantizedStorage {
    /// Dimensionality of stored vectors
    dimensions: usize,

    /// Quantization type
    quantization: QuantizationType,

    /// SQ8 quantizer (if using SQ8)
    sq8_quantizer: Option<SQ8Quantizer>,

    /// Binary quantizer (if using Binary)
    binary_quantizer: Option<BinaryQuantizer>,

    /// SQ8: Quantized vectors (contiguous u8 storage)
    sq8_vectors: RwLock<Vec<u8>>,

    /// SQ8: Metadata for each vector
    sq8_metadata: RwLock<Vec<SQ8Metadata>>,

    /// Binary: Quantized vectors
    binary_vectors: RwLock<Vec<u8>>,

    /// Original f32 vectors (for re-ranking if needed)
    /// Only stored if keep_originals is true
    original_vectors: RwLock<Option<Vec<f32>>>,

    /// Whether to keep original vectors for re-ranking
    keep_originals: bool,

    /// Map from external ID to internal ID
    id_to_internal: RwLock<HashMap<VectorId, InternalId>>,

    /// Map from internal ID to external ID
    internal_to_id: RwLock<Vec<VectorId>>,

    /// Optional metadata for each vector
    metadata: RwLock<HashMap<InternalId, Value>>,
}

impl QuantizedStorage {
    /// Create a new quantized storage
    pub fn new(dimensions: usize, quantization: QuantizationType, keep_originals: bool) -> Self {
        let sq8_quantizer = match quantization {
            QuantizationType::SQ8 => Some(SQ8Quantizer::new(dimensions)),
            _ => None,
        };

        let binary_quantizer = match quantization {
            QuantizationType::Binary => Some(BinaryQuantizer::new(dimensions)),
            _ => None,
        };

        // For None quantization, we always need to store originals
        let needs_originals = keep_originals || quantization == QuantizationType::None;
        let original_vectors = if needs_originals {
            Some(Vec::new())
        } else {
            None
        };

        Self {
            dimensions,
            quantization,
            sq8_quantizer,
            binary_quantizer,
            sq8_vectors: RwLock::new(Vec::new()),
            sq8_metadata: RwLock::new(Vec::new()),
            binary_vectors: RwLock::new(Vec::new()),
            original_vectors: RwLock::new(original_vectors),
            keep_originals,
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

    /// Insert or update a vector
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

        let mut internal_to_id = self.internal_to_id.write();
        let internal_id = InternalId::from(internal_to_id.len());

        // Store quantized version
        match self.quantization {
            QuantizationType::None => {
                // For None quantization, store in original_vectors
                let mut originals = self.original_vectors.write();
                if let Some(ref mut vecs) = *originals {
                    vecs.extend_from_slice(vector);
                }
            }
            QuantizationType::SQ8 => {
                let quantizer = self.sq8_quantizer.as_ref().unwrap();
                let (quantized, sq8_meta) = quantizer.quantize(vector);

                let mut sq8_vectors = self.sq8_vectors.write();
                let mut sq8_metadata = self.sq8_metadata.write();

                sq8_vectors.extend_from_slice(&quantized);
                sq8_metadata.push(sq8_meta);
            }
            QuantizationType::Binary => {
                let quantizer = self.binary_quantizer.as_ref().unwrap();
                let quantized = quantizer.quantize(vector);

                let mut binary_vectors = self.binary_vectors.write();
                binary_vectors.extend_from_slice(&quantized);
            }
        }

        // Store original if requested
        if self.keep_originals && self.quantization != QuantizationType::None {
            let mut originals = self.original_vectors.write();
            if let Some(ref mut vecs) = *originals {
                vecs.extend_from_slice(vector);
            }
        }

        // Update mappings
        id_to_internal.insert(id.clone(), internal_id);
        internal_to_id.push(id);

        // Store metadata if present
        if let Some(meta) = metadata {
            self.metadata.write().insert(internal_id, meta);
        }

        Ok(internal_id)
    }

    /// Calculate distance from query to stored vector
    #[inline]
    pub fn distance(
        &self,
        query: &[f32],
        internal_id: InternalId,
        metric: DistanceMetric,
    ) -> Option<f32> {
        match self.quantization {
            QuantizationType::None => {
                let originals = self.original_vectors.read();
                if let Some(ref vecs) = *originals {
                    let start = internal_id.as_usize() * self.dimensions;
                    let end = start + self.dimensions;
                    if end <= vecs.len() {
                        return Some(metric.distance(query, &vecs[start..end]));
                    }
                }
                None
            }
            QuantizationType::SQ8 => {
                let quantizer = self.sq8_quantizer.as_ref()?;
                let sq8_vectors = self.sq8_vectors.read();
                let sq8_metadata = self.sq8_metadata.read();

                let idx = internal_id.as_usize();
                if idx >= sq8_metadata.len() {
                    return None;
                }

                let start = idx * self.dimensions;
                let end = start + self.dimensions;
                if end > sq8_vectors.len() {
                    return None;
                }

                let quantized = &sq8_vectors[start..end];
                let metadata = &sq8_metadata[idx];

                Some(quantizer.asymmetric_distance(query, quantized, metadata, metric))
            }
            QuantizationType::Binary => {
                let quantizer = self.binary_quantizer.as_ref()?;
                let binary_vectors = self.binary_vectors.read();

                let byte_size = quantizer.byte_size();
                let start = internal_id.as_usize() * byte_size;
                let end = start + byte_size;

                if end > binary_vectors.len() {
                    return None;
                }

                // Quantize query on the fly
                let query_binary = quantizer.quantize(query);
                let stored = &binary_vectors[start..end];

                let hamming = quantizer.hamming_distance(&query_binary, stored);
                Some(quantizer.hamming_to_cosine(hamming))
            }
        }
    }

    /// Get original vector (for re-ranking)
    pub fn get_original(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        let originals = self.original_vectors.read();
        if let Some(ref vecs) = *originals {
            let start = internal_id.as_usize() * self.dimensions;
            let end = start + self.dimensions;
            if end <= vecs.len() {
                return Some(vecs[start..end].to_vec());
            }
        }
        None
    }

    /// Get metadata for a vector
    pub fn get_metadata(&self, internal_id: InternalId) -> Option<Value> {
        self.metadata.read().get(&internal_id).cloned()
    }

    /// Get external ID from internal ID
    pub fn get_external_id(&self, internal_id: InternalId) -> Option<VectorId> {
        let internal_to_id = self.internal_to_id.read();
        internal_to_id.get(internal_id.as_usize()).cloned()
    }

    /// Get internal ID from external ID
    pub fn get_internal_id(&self, id: &VectorId) -> Option<InternalId> {
        self.id_to_internal.read().get(id).copied()
    }

    /// Get all internal IDs
    pub fn all_internal_ids(&self) -> Vec<InternalId> {
        let internal_to_id = self.internal_to_id.read();
        (0..internal_to_id.len()).map(InternalId::from).collect()
    }

    /// Get the number of stored vectors
    pub fn len(&self) -> usize {
        self.internal_to_id.read().len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let quantized_size = match self.quantization {
            QuantizationType::None => 0,
            QuantizationType::SQ8 => {
                self.sq8_vectors.read().len()
                    + self.sq8_metadata.read().len() * std::mem::size_of::<SQ8Metadata>()
            }
            QuantizationType::Binary => self.binary_vectors.read().len(),
        };

        let original_size = self
            .original_vectors
            .read()
            .as_ref()
            .map(|v| v.len() * 4)
            .unwrap_or(0);

        quantized_size + original_size
    }

    /// Get compression ratio compared to f32 storage
    pub fn compression_ratio(&self) -> f32 {
        let count = self.len();
        if count == 0 {
            return 1.0;
        }

        let f32_size = count * self.dimensions * 4;
        let actual_size = self.memory_usage();

        if actual_size == 0 {
            return 1.0;
        }

        f32_size as f32 / actual_size as f32
    }

    /// Get dimensionality
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get quantization type
    pub fn quantization_type(&self) -> QuantizationType {
        self.quantization
    }

    /// Create a view of the storage that holds read locks
    pub fn view(&self) -> QuantizedStorageView<'_> {
        let (sq8_vectors, sq8_metadata) = if self.quantization == QuantizationType::SQ8 {
            (
                Some(self.sq8_vectors.read()),
                Some(self.sq8_metadata.read()),
            )
        } else {
            (None, None)
        };

        let binary_vectors = if self.quantization == QuantizationType::Binary {
            Some(self.binary_vectors.read())
        } else {
            None
        };

        let original_vectors = if self.quantization == QuantizationType::None {
            Some(self.original_vectors.read())
        } else {
            None
        };

        QuantizedStorageView {
            dimensions: self.dimensions,
            quantization: self.quantization,
            sq8_quantizer: self.sq8_quantizer.as_ref(),
            binary_quantizer: self.binary_quantizer.as_ref(),
            sq8_vectors,
            sq8_metadata,
            binary_vectors,
            original_vectors,
        }
    }

    /// Quantize a query vector for fast distance calculations
    pub fn quantize_query(&self, query: &[f32]) -> QuantizedQuery {
        match self.quantization {
            QuantizationType::None => QuantizedQuery::None,
            QuantizationType::SQ8 => {
                // For SQ8 asymmetric search, we don't necessarily quantize the query
                // because we compare f32 query vs u8 stored.
                // However, we might want to pre-process it if possible?
                // Actually, the asymmetric_distance function takes &[f32].
                // So no quantization needed for query.
                QuantizedQuery::SQ8
            }
            QuantizationType::Binary => {
                if let Some(quantizer) = &self.binary_quantizer {
                    let binary = quantizer.quantize(query);
                    QuantizedQuery::Binary(binary)
                } else {
                    QuantizedQuery::None
                }
            }
        }
    }
}

impl VectorStorageTrait for QuantizedStorage {
    fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        // If we have originals, return them
        if let Some(orig) = self.get_original(internal_id) {
            return Some(orig);
        }

        // Dequantize on demand
        match self.quantization {
            QuantizationType::None => None, // Should have been handled by get_original
            QuantizationType::SQ8 => {
                let quantizer = self.sq8_quantizer.as_ref()?;
                let sq8_vectors = self.sq8_vectors.read();
                let sq8_metadata = self.sq8_metadata.read();

                let idx = internal_id.as_usize();
                if idx >= sq8_metadata.len() {
                    return None;
                }

                let start = idx * self.dimensions;
                let end = start + self.dimensions;
                if end > sq8_vectors.len() {
                    return None;
                }

                Some(quantizer.dequantize(&sq8_vectors[start..end], &sq8_metadata[idx]))
            }
            QuantizationType::Binary => {
                // Binary cannot be easily dequantized to f32 without massive loss
                None
            }
        }
    }

    fn distance(
        &self,
        internal_id: InternalId,
        query: &[f32],
        metric: DistanceMetric,
    ) -> Option<f32> {
        self.distance(query, internal_id, metric)
    }
}

/// A pre-quantized query vector to avoid reallocation during search
#[derive(Debug, Clone)]
pub enum QuantizedQuery {
    None,
    SQ8, // Placeholder as we use asymmetric distance (f32 query)
    Binary(Vec<u8>),
}

/// A view into QuantizedStorage that holds read locks
pub struct QuantizedStorageView<'a> {
    dimensions: usize,
    quantization: QuantizationType,
    sq8_quantizer: Option<&'a SQ8Quantizer>,
    binary_quantizer: Option<&'a BinaryQuantizer>,
    sq8_vectors: Option<parking_lot::RwLockReadGuard<'a, Vec<u8>>>,
    sq8_metadata: Option<parking_lot::RwLockReadGuard<'a, Vec<SQ8Metadata>>>,
    binary_vectors: Option<parking_lot::RwLockReadGuard<'a, Vec<u8>>>,
    original_vectors: Option<parking_lot::RwLockReadGuard<'a, Option<Vec<f32>>>>,
}

impl<'a> QuantizedStorageView<'a> {
    /// Calculate distance using a pre-quantized query
    pub fn distance_quantized(
        &self,
        query: &[f32],
        quantized_query: &QuantizedQuery,
        internal_id: InternalId,
        metric: DistanceMetric,
    ) -> Option<f32> {
        match self.quantization {
            QuantizationType::None => {
                let originals = self.original_vectors.as_ref()?;
                if let Some(ref vecs) = **originals {
                    let start = internal_id.as_usize() * self.dimensions;
                    let end = start + self.dimensions;
                    if end <= vecs.len() {
                        return Some(metric.distance(query, &vecs[start..end]));
                    }
                }
                None
            }
            QuantizationType::SQ8 => {
                let quantizer = self.sq8_quantizer?;
                let sq8_vectors = self.sq8_vectors.as_ref()?;
                let sq8_metadata = self.sq8_metadata.as_ref()?;

                let idx = internal_id.as_usize();
                if idx >= sq8_metadata.len() {
                    return None;
                }

                let start = idx * self.dimensions;
                let end = start + self.dimensions;
                if end > sq8_vectors.len() {
                    return None;
                }

                let quantized = &sq8_vectors[start..end];
                let metadata = &sq8_metadata[idx];

                Some(quantizer.asymmetric_distance(query, quantized, metadata, metric))
            }
            QuantizationType::Binary => {
                let quantizer = self.binary_quantizer?;
                let binary_vectors = self.binary_vectors.as_ref()?;

                // Use pre-quantized query if available
                let query_binary = match quantized_query {
                    QuantizedQuery::Binary(b) => b,
                    _ => return None, // Should not happen if logic is correct
                };

                let byte_size = quantizer.byte_size();
                let start = internal_id.as_usize() * byte_size;
                let end = start + byte_size;

                if end > binary_vectors.len() {
                    return None;
                }

                let stored = &binary_vectors[start..end];
                let hamming = quantizer.hamming_distance(query_binary, stored);
                Some(quantizer.hamming_to_cosine(hamming))
            }
        }
    }

    /// Calculate distance from query to stored vector (legacy/slow path)
    pub fn distance(
        &self,
        query: &[f32],
        internal_id: InternalId,
        metric: DistanceMetric,
    ) -> Option<f32> {
        // Fallback to non-pre-quantized distance
        self.distance_quantized(query, &QuantizedQuery::None, internal_id, metric)
    }
}

impl<'a> VectorStorageTrait for QuantizedStorageView<'a> {
    fn get_vector_data(&self, internal_id: InternalId) -> Option<Vec<f32>> {
        // If we have originals, return them
        if let Some(originals) = &self.original_vectors {
            if let Some(ref vecs) = **originals {
                let start = internal_id.as_usize() * self.dimensions;
                let end = start + self.dimensions;
                if end <= vecs.len() {
                    return Some(vecs[start..end].to_vec());
                }
            }
        }

        match self.quantization {
            QuantizationType::None => None,
            QuantizationType::SQ8 => {
                let quantizer = self.sq8_quantizer?;
                let sq8_vectors = self.sq8_vectors.as_ref()?;
                let sq8_metadata = self.sq8_metadata.as_ref()?;

                let idx = internal_id.as_usize();
                if idx >= sq8_metadata.len() {
                    return None;
                }

                let start = idx * self.dimensions;
                let end = start + self.dimensions;
                if end > sq8_vectors.len() {
                    return None;
                }

                Some(quantizer.dequantize(&sq8_vectors[start..end], &sq8_metadata[idx]))
            }
            QuantizationType::Binary => None,
        }
    }

    fn distance(
        &self,
        internal_id: InternalId,
        query: &[f32],
        metric: DistanceMetric,
    ) -> Option<f32> {
        // Fallback to non-pre-quantized distance
        self.distance_quantized(query, &QuantizedQuery::None, internal_id, metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_storage() {
        let storage = QuantizedStorage::new(4, QuantizationType::SQ8, false);

        let id = VectorId::from("test");
        let vector = vec![1.0, 0.0, 0.0, 0.0];

        let internal_id = storage.insert(id, &vector, None).unwrap();

        let dist = storage
            .distance(&vector, internal_id, DistanceMetric::Cosine)
            .unwrap();

        // Distance to self should be ~0
        assert!(dist < 0.01, "dist={}", dist);
    }

    #[test]
    fn test_binary_storage() {
        let storage = QuantizedStorage::new(8, QuantizationType::Binary, false);

        let v1 = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let v2 = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

        let id1 = storage.insert("v1".into(), &v1, None).unwrap();
        let id2 = storage.insert("v2".into(), &v2, None).unwrap();

        let dist1 = storage.distance(&v1, id1, DistanceMetric::Cosine).unwrap();
        let dist2 = storage.distance(&v1, id2, DistanceMetric::Cosine).unwrap();

        // Distance to self should be 0
        assert!(dist1 < 0.01, "dist1={}", dist1);
        // Distance to different vector should be > 0
        assert!(dist2 > 0.0, "dist2={}", dist2);
    }

    #[test]
    fn test_keep_originals() {
        let storage = QuantizedStorage::new(4, QuantizationType::SQ8, true);

        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let internal_id = storage.insert("test".into(), &vector, None).unwrap();

        let original = storage.get_original(internal_id).unwrap();
        assert_eq!(original, vector);
    }

    #[test]
    fn test_compression_ratio() {
        let storage = QuantizedStorage::new(384, QuantizationType::SQ8, false);

        // Insert 100 vectors
        for i in 0..100 {
            let vector: Vec<f32> = (0..384).map(|j| (i * j) as f32 / 1000.0).collect();
            storage
                .insert(format!("v{}", i).into(), &vector, None)
                .unwrap();
        }

        let ratio = storage.compression_ratio();
        // SQ8 should give ~4x compression (minus metadata overhead)
        assert!(ratio > 3.5, "compression ratio: {}", ratio);
    }
}
