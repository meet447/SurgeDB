//! Quantization module for vector compression
//!
//! Provides SQ8 (Scalar Quantization to 8-bit) and Binary Quantization
//! for significant memory reduction with minimal accuracy loss.
//!
//! ## SQ8 (Scalar Quantization)
//! - Converts f32 (4 bytes) to u8 (1 byte) = **4x compression**
//! - Uses min-max scaling per vector
//! - Typical recall loss: < 5% for most embedding models
//!
//! ## Binary Quantization (BQ)
//! - Converts f32 to single bit = **32x compression**
//! - Uses sign of each dimension
//! - Best for first-pass retrieval with re-ranking

use crate::distance::DistanceMetric;
use serde::{Deserialize, Serialize};

/// Quantization method to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization - full f32 precision
    #[default]
    None,
    /// Scalar quantization to 8-bit (4x compression)
    SQ8,
    /// Binary quantization (32x compression)
    Binary,
}

/// Metadata for reconstructing quantized vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SQ8Metadata {
    /// Minimum value per vector (for denormalization)
    pub min: f32,
    /// Scale factor per vector (max - min) / 255
    pub scale: f32,
}

impl SQ8Metadata {
    /// Create metadata from a vector's min/max values
    pub fn from_vector(vector: &[f32]) -> Self {
        let min = vector.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        Self { min, scale }
    }

    /// Quantize a single f32 value to u8
    #[inline]
    pub fn quantize_value(&self, value: f32) -> u8 {
        let normalized = (value - self.min) / self.scale;
        normalized.clamp(0.0, 255.0) as u8
    }

    /// Dequantize a single u8 value to f32
    #[inline]
    pub fn dequantize_value(&self, value: u8) -> f32 {
        (value as f32) * self.scale + self.min
    }
}

/// SQ8 Quantizer - handles encoding and distance calculations
#[derive(Debug, Clone)]
pub struct SQ8Quantizer {
    dimensions: usize,
}

impl SQ8Quantizer {
    /// Create a new SQ8 quantizer
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    /// Quantize a f32 vector to u8 with metadata
    pub fn quantize(&self, vector: &[f32]) -> (Vec<u8>, SQ8Metadata) {
        let metadata = SQ8Metadata::from_vector(vector);
        let quantized: Vec<u8> = vector.iter().map(|&v| metadata.quantize_value(v)).collect();
        (quantized, metadata)
    }

    /// Dequantize a u8 vector back to f32
    pub fn dequantize(&self, quantized: &[u8], metadata: &SQ8Metadata) -> Vec<f32> {
        quantized
            .iter()
            .map(|&v| metadata.dequantize_value(v))
            .collect()
    }

    /// Calculate asymmetric distance: query (f32) vs stored (u8)
    /// This is faster than symmetric and typically used for search
    #[inline]
    pub fn asymmetric_distance(
        &self,
        query: &[f32],
        quantized: &[u8],
        metadata: &SQ8Metadata,
        metric: DistanceMetric,
    ) -> f32 {
        match metric {
            DistanceMetric::Cosine => self.asymmetric_cosine_distance(query, quantized, metadata),
            DistanceMetric::Euclidean => {
                self.asymmetric_euclidean_distance(query, quantized, metadata)
            }
            DistanceMetric::DotProduct => {
                self.asymmetric_dot_product_distance(query, quantized, metadata)
            }
        }
    }

    /// Asymmetric cosine distance with SIMD optimization
    #[inline]
    fn asymmetric_cosine_distance(
        &self,
        query: &[f32],
        quantized: &[u8],
        metadata: &SQ8Metadata,
    ) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            self.asymmetric_cosine_neon(query, quantized, metadata)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            self.asymmetric_cosine_scalar(query, quantized, metadata)
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn asymmetric_cosine_scalar(
        &self,
        query: &[f32],
        quantized: &[u8],
        metadata: &SQ8Metadata,
    ) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_q = 0.0f32;
        let mut norm_v = 0.0f32;

        for i in 0..query.len() {
            let q = query[i];
            let v = metadata.dequantize_value(quantized[i]);
            dot += q * v;
            norm_q += q * q;
            norm_v += v * v;
        }

        let denom = (norm_q * norm_v).sqrt();
        if denom == 0.0 {
            return 1.0;
        }
        1.0 - (dot / denom)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn asymmetric_cosine_neon(
        &self,
        query: &[f32],
        quantized: &[u8],
        metadata: &SQ8Metadata,
    ) -> f32 {
        use std::arch::aarch64::*;

        let n = query.len();
        let chunks = n / 4;

        unsafe {
            let scale_vec = vdupq_n_f32(metadata.scale);
            let min_vec = vdupq_n_f32(metadata.min);

            let mut dot_acc = vdupq_n_f32(0.0);
            let mut norm_q_acc = vdupq_n_f32(0.0);
            let mut norm_v_acc = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;

                // Load query vector (f32)
                let q = vld1q_f32(query.as_ptr().add(offset));

                // Load and convert quantized values (u8 -> f32)
                let q_bytes: [u8; 4] = [
                    quantized[offset],
                    quantized[offset + 1],
                    quantized[offset + 2],
                    quantized[offset + 3],
                ];
                let v_u8 = vld1_u8(q_bytes.as_ptr());
                let v_u16 = vmovl_u8(v_u8);
                let v_u32 = vmovl_u16(vget_low_u16(v_u16));
                let v_f32 = vcvtq_f32_u32(v_u32);

                // Dequantize: v = v_f32 * scale + min
                let v = vfmaq_f32(min_vec, v_f32, scale_vec);

                // Accumulate
                dot_acc = vfmaq_f32(dot_acc, q, v);
                norm_q_acc = vfmaq_f32(norm_q_acc, q, q);
                norm_v_acc = vfmaq_f32(norm_v_acc, v, v);
            }

            // Horizontal sum
            let mut dot = vaddvq_f32(dot_acc);
            let mut norm_q = vaddvq_f32(norm_q_acc);
            let mut norm_v = vaddvq_f32(norm_v_acc);

            // Handle remainder
            for i in (chunks * 4)..n {
                let q = query[i];
                let v = metadata.dequantize_value(quantized[i]);
                dot += q * v;
                norm_q += q * q;
                norm_v += v * v;
            }

            let denom = (norm_q * norm_v).sqrt();
            if denom == 0.0 {
                return 1.0;
            }
            1.0 - (dot / denom)
        }
    }

    #[inline]
    fn asymmetric_euclidean_distance(
        &self,
        query: &[f32],
        quantized: &[u8],
        metadata: &SQ8Metadata,
    ) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..query.len() {
            let diff = query[i] - metadata.dequantize_value(quantized[i]);
            sum += diff * diff;
        }
        sum.sqrt()
    }

    #[inline]
    fn asymmetric_dot_product_distance(
        &self,
        query: &[f32],
        quantized: &[u8],
        metadata: &SQ8Metadata,
    ) -> f32 {
        let mut dot = 0.0f32;
        for i in 0..query.len() {
            dot += query[i] * metadata.dequantize_value(quantized[i]);
        }
        1.0 - dot
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Binary Quantizer - extreme compression (32x)
#[derive(Debug, Clone)]
pub struct BinaryQuantizer {
    dimensions: usize,
    /// Number of bytes needed to store the binary vector
    byte_size: usize,
}

impl BinaryQuantizer {
    /// Create a new binary quantizer
    pub fn new(dimensions: usize) -> Self {
        let byte_size = dimensions.div_ceil(8); // Round up to nearest byte
        Self {
            dimensions,
            byte_size,
        }
    }

    /// Quantize a f32 vector to binary (1 bit per dimension)
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        let mut result = vec![0u8; self.byte_size];

        for (i, &value) in vector.iter().enumerate() {
            if value > 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                result[byte_idx] |= 1 << bit_idx;
            }
        }

        result
    }

    /// Calculate Hamming distance between two binary vectors
    #[inline]
    pub fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        #[cfg(target_arch = "aarch64")]
        {
            self.hamming_distance_neon(a, b)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            self.hamming_distance_scalar(a, b)
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn hamming_distance_scalar(&self, a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn hamming_distance_neon(&self, a: &[u8], b: &[u8]) -> u32 {
        use std::arch::aarch64::*;

        let n = a.len();
        let chunks = n / 16;

        unsafe {
            let mut total: u32 = 0;

            for i in 0..chunks {
                let offset = i * 16;
                let va = vld1q_u8(a.as_ptr().add(offset));
                let vb = vld1q_u8(b.as_ptr().add(offset));
                let xor = veorq_u8(va, vb);
                let cnt = vcntq_u8(xor);
                total += vaddlvq_u8(cnt) as u32;
            }

            // Handle remainder
            for i in (chunks * 16)..n {
                total += (a[i] ^ b[i]).count_ones();
            }

            total
        }
    }

    /// Convert Hamming distance to approximate cosine distance
    /// Useful for ranking compatibility with non-quantized results
    #[inline]
    pub fn hamming_to_cosine(&self, hamming: u32) -> f32 {
        // Approximate: cosine_dist ≈ hamming / dimensions
        hamming as f32 / self.dimensions as f32
    }

    /// Get byte size
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_quantize_dequantize() {
        let quantizer = SQ8Quantizer::new(4);
        let vector = vec![0.0, 0.5, 1.0, 0.25];

        let (quantized, metadata) = quantizer.quantize(&vector);
        let dequantized = quantizer.dequantize(&quantized, &metadata);

        // Check that dequantized values are close to original
        for (orig, deq) in vector.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.01, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn test_sq8_metadata() {
        let vector = vec![-1.0, 0.0, 1.0, 2.0];
        let metadata = SQ8Metadata::from_vector(&vector);

        assert_eq!(metadata.min, -1.0);
        assert!((metadata.scale - 3.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_sq8_asymmetric_distance() {
        let quantizer = SQ8Quantizer::new(4);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0, 0.0];

        let (q2, meta2) = quantizer.quantize(&v2);

        let dist = quantizer.asymmetric_distance(&v1, &q2, &meta2, DistanceMetric::Cosine);

        // Should be very close to 0 (identical vectors)
        assert!(dist < 0.01, "dist={}", dist);
    }

    #[test]
    fn test_binary_quantize() {
        let quantizer = BinaryQuantizer::new(8);
        let vector = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.1, -0.1, 0.0];

        let binary = quantizer.quantize(&vector);

        // Expected: bits set for positive values at positions 0, 2, 5
        // Binary: 0010_0101 = 0x25
        assert_eq!(binary.len(), 1);
        assert_eq!(binary[0], 0b00100101);
    }

    #[test]
    fn test_binary_hamming_distance() {
        let quantizer = BinaryQuantizer::new(8);

        let v1 = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let v2 = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

        let b1 = quantizer.quantize(&v1);
        let b2 = quantizer.quantize(&v2);

        let dist = quantizer.hamming_distance(&b1, &b2);

        // 4 bits should differ
        assert_eq!(dist, 4);
    }

    #[test]
    fn test_compression_ratio() {
        // SQ8: 4 bytes -> 1 byte = 4x compression
        let dims = 384;
        let f32_size = dims * 4;
        let sq8_size = dims + 8; // +8 for metadata (min + scale)

        let sq8_ratio = f32_size as f32 / sq8_size as f32;
        assert!(sq8_ratio > 3.9, "SQ8 compression ratio: {}", sq8_ratio);

        // Binary: 4 bytes -> 1/8 byte = 32x compression
        let binary_size = (dims + 7) / 8;
        let binary_ratio = f32_size as f32 / binary_size as f32;
        assert!(
            binary_ratio > 31.0,
            "Binary compression ratio: {}",
            binary_ratio
        );
    }
}
