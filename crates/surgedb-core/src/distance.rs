//! SIMD-optimized distance calculations
//!
//! This module provides highly optimized distance functions using platform-specific
//! SIMD instructions (NEON on ARM, AVX on x86).

use serde::{Deserialize, Serialize};

/// Distance metric to use for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cos(a, b))
    /// Best for normalized embeddings (OpenAI, sentence-transformers)
    #[default]
    Cosine,

    /// Euclidean distance (L2 norm)
    /// Good for geometric similarity
    Euclidean,

    /// Dot product (inner product)
    /// Fast but requires normalized vectors for proper similarity
    DotProduct,
}

impl DistanceMetric {
    /// Calculate distance between two vectors
    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => dot_product_distance(a, b),
        }
    }
}

/// Cosine distance: 1 - cosine_similarity
/// Returns 0 for identical vectors, 2 for opposite vectors
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        cosine_distance_neon(a, b)
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        cosine_distance_avx(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        cosine_distance_scalar(a, b)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        cosine_distance_wasm(a, b)
    }

    #[cfg(all(
        feature = "simd",
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        not(all(target_arch = "wasm32", target_feature = "simd128"))
    ))]
    {
        cosine_distance_scalar(a, b)
    }
}

/// Euclidean distance (L2)
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        euclidean_distance_neon(a, b)
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        euclidean_distance_avx(a, b)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        euclidean_distance_wasm(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        euclidean_distance_scalar(a, b)
    }

    #[cfg(all(
        feature = "simd",
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        not(all(target_arch = "wasm32", target_feature = "simd128"))
    ))]
    {
        euclidean_distance_scalar(a, b)
    }
}

/// Dot product distance (1 - dot_product for normalized vectors)
#[inline]
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        1.0 - dot_product_neon(a, b)
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        1.0 - dot_product_avx(a, b)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        1.0 - dot_product_wasm(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        1.0 - dot_product_scalar(a, b)
    }

    #[cfg(all(
        feature = "simd",
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        not(all(target_arch = "wasm32", target_feature = "simd128"))
    ))]
    {
        1.0 - dot_product_scalar(a, b)
    }
}

// =============================================================================
// Scalar implementations (fallback / used on non-SIMD platforms)
// =============================================================================

#[inline]
#[allow(dead_code)]
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 1.0;
    }

    1.0 - (dot / denom)
}

#[inline]
#[allow(dead_code)]
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

#[inline]
#[allow(dead_code)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

// =============================================================================
// ARM NEON implementations (Apple Silicon M1/M2/M3)
// =============================================================================

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let mut dot_acc = vdupq_n_f32(0.0);
        let mut norm_a_acc = vdupq_n_f32(0.0);
        let mut norm_b_acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));

            dot_acc = vfmaq_f32(dot_acc, va, vb);
            norm_a_acc = vfmaq_f32(norm_a_acc, va, va);
            norm_b_acc = vfmaq_f32(norm_b_acc, vb, vb);
        }

        // Horizontal sum
        let dot = vaddvq_f32(dot_acc);
        let norm_a = vaddvq_f32(norm_a_acc);
        let norm_b = vaddvq_f32(norm_b_acc);

        // Handle remainder
        let mut dot_rem = 0.0f32;
        let mut norm_a_rem = 0.0f32;
        let mut norm_b_rem = 0.0f32;

        for i in (chunks * 4)..n {
            dot_rem += a[i] * b[i];
            norm_a_rem += a[i] * a[i];
            norm_b_rem += b[i] * b[i];
        }

        let total_dot = dot + dot_rem;
        let total_norm_a = norm_a + norm_a_rem;
        let total_norm_b = norm_b + norm_b_rem;

        let denom = (total_norm_a * total_norm_b).sqrt();
        if denom == 0.0 {
            return 1.0;
        }

        1.0 - (total_dot / denom)
    }
}

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let mut sum_acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));

            let diff = vsubq_f32(va, vb);
            sum_acc = vfmaq_f32(sum_acc, diff, diff);
        }

        let mut sum = vaddvq_f32(sum_acc);

        // Handle remainder
        for i in (chunks * 4)..n {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }

        sum.sqrt()
    }
}

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            acc = vfmaq_f32(acc, va, vb);
        }

        let mut sum = vaddvq_f32(acc);

        // Handle remainder
        for i in (chunks * 4)..n {
            sum += a[i] * b[i];
        }

        sum
    }
}

// =============================================================================
// x86_64 AVX implementations
// =============================================================================

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
fn cosine_distance_avx(a: &[f32], b: &[f32]) -> f32 {
    // Check for AVX support at runtime
    if is_x86_feature_detected!("avx") {
        unsafe { cosine_distance_avx_inner(a, b) }
    } else {
        cosine_distance_scalar(a, b)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn cosine_distance_avx_inner(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    let mut dot_acc = _mm256_setzero_ps();
    let mut norm_a_acc = _mm256_setzero_ps();
    let mut norm_b_acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
    }

    // Horizontal sum for AVX (256-bit -> 128-bit -> scalar)
    fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
        unsafe {
            let high = _mm256_extractf128_ps(v, 1);
            let low = _mm256_castps256_ps128(v);
            let sum128 = _mm_add_ps(high, low);
            let high64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, high64);
            let high32 = _mm_shuffle_ps(sum64, sum64, 1);
            _mm_cvtss_f32(_mm_add_ss(sum64, high32))
        }
    }

    let mut dot = hsum_avx(dot_acc);
    let mut norm_a = hsum_avx(norm_a_acc);
    let mut norm_b = hsum_avx(norm_b_acc);

    // Handle remainder
    for i in (chunks * 8)..n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 1.0;
    }

    1.0 - (dot / denom)
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
fn euclidean_distance_avx(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { euclidean_distance_avx_inner(a, b) }
    } else {
        euclidean_distance_scalar(a, b)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn euclidean_distance_avx_inner(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    let mut sum_acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum_acc = _mm256_fmadd_ps(diff, diff, sum_acc);
    }

    // Horizontal sum
    let high = _mm256_extractf128_ps(sum_acc, 1);
    let low = _mm256_castps256_ps128(sum_acc);
    let sum128 = _mm_add_ps(high, low);
    let high64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, high64);
    let high32 = _mm_shuffle_ps(sum64, sum64, 1);
    let mut sum = _mm_cvtss_f32(_mm_add_ss(sum64, high32));

    // Handle remainder
    for i in (chunks * 8)..n {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum.sqrt()
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { dot_product_avx_inner(a, b) }
    } else {
        dot_product_scalar(a, b)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn dot_product_avx_inner(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    // Horizontal sum
    let high = _mm256_extractf128_ps(acc, 1);
    let low = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(high, low);
    let high64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, high64);
    let high32 = _mm_shuffle_ps(sum64, sum64, 1);
    let mut sum = _mm_cvtss_f32(_mm_add_ss(sum64, high32));

    // Handle remainder
    for i in (chunks * 8)..n {
        sum += a[i] * b[i];
    }

    sum
}

// =============================================================================
// WASM SIMD128 implementations
// =============================================================================

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn cosine_distance_wasm(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    // Accumulators
    let mut dot_acc = f32x4_splat(0.0);
    let mut norm_a_acc = f32x4_splat(0.0);
    let mut norm_b_acc = f32x4_splat(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = unsafe { v128_load(a.as_ptr().add(offset) as *const v128) };
        let vb = unsafe { v128_load(b.as_ptr().add(offset) as *const v128) };

        dot_acc = f32x4_add(dot_acc, f32x4_mul(va, vb));
        norm_a_acc = f32x4_add(norm_a_acc, f32x4_mul(va, va));
        norm_b_acc = f32x4_add(norm_b_acc, f32x4_mul(vb, vb));
    }

    // Horizontal sum
    let dot = f32x4_extract_lane::<0>(dot_acc)
        + f32x4_extract_lane::<1>(dot_acc)
        + f32x4_extract_lane::<2>(dot_acc)
        + f32x4_extract_lane::<3>(dot_acc);
    let norm_a = f32x4_extract_lane::<0>(norm_a_acc)
        + f32x4_extract_lane::<1>(norm_a_acc)
        + f32x4_extract_lane::<2>(norm_a_acc)
        + f32x4_extract_lane::<3>(norm_a_acc);
    let norm_b = f32x4_extract_lane::<0>(norm_b_acc)
        + f32x4_extract_lane::<1>(norm_b_acc)
        + f32x4_extract_lane::<2>(norm_b_acc)
        + f32x4_extract_lane::<3>(norm_b_acc);

    // Handle remainder
    let mut dot_rem = 0.0f32;
    let mut norm_a_rem = 0.0f32;
    let mut norm_b_rem = 0.0f32;

    for i in (chunks * 4)..n {
        dot_rem += a[i] * b[i];
        norm_a_rem += a[i] * a[i];
        norm_b_rem += b[i] * b[i];
    }

    let total_dot = dot + dot_rem;
    let total_norm_a = norm_a + norm_a_rem;
    let total_norm_b = norm_b + norm_b_rem;

    let denom = (total_norm_a * total_norm_b).sqrt();
    if denom == 0.0 {
        return 1.0;
    }

    1.0 - (total_dot / denom)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn euclidean_distance_wasm(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    let mut sum_acc = f32x4_splat(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = unsafe { v128_load(a.as_ptr().add(offset) as *const v128) };
        let vb = unsafe { v128_load(b.as_ptr().add(offset) as *const v128) };

        let diff = f32x4_sub(va, vb);
        sum_acc = f32x4_add(sum_acc, f32x4_mul(diff, diff));
    }

    let mut sum = f32x4_extract_lane::<0>(sum_acc)
        + f32x4_extract_lane::<1>(sum_acc)
        + f32x4_extract_lane::<2>(sum_acc)
        + f32x4_extract_lane::<3>(sum_acc);

    // Handle remainder
    for i in (chunks * 4)..n {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum.sqrt()
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn dot_product_wasm(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    let mut acc = f32x4_splat(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = unsafe { v128_load(a.as_ptr().add(offset) as *const v128) };
        let vb = unsafe { v128_load(b.as_ptr().add(offset) as *const v128) };

        acc = f32x4_add(acc, f32x4_mul(va, vb));
    }

    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);

    // Handle remainder
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn assert_float_eq(a: f32, b: f32) {
        assert!(
            (a - b).abs() < EPSILON,
            "Expected {} to equal {} (within {})",
            a,
            b,
            EPSILON
        );
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert_float_eq(cosine_distance(&a, &a), 0.0);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert_float_eq(cosine_distance(&a, &b), 1.0);
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0, 0.0];
        assert_float_eq(cosine_distance(&a, &b), 2.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        assert_float_eq(euclidean_distance(&a, &b), 1.0);
    }

    #[test]
    fn test_dot_product_distance() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        // Dot product = 1, distance = 1 - 1 = 0
        assert_float_eq(dot_product_distance(&a, &b), 0.0);
    }

    #[test]
    fn test_large_vectors() {
        // Test with 384-dimensional vectors (MiniLM size)
        let a: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        let b: Vec<f32> = (0..384).map(|i| ((i + 1) as f32) / 384.0).collect();

        let dist = cosine_distance(&a, &b);
        assert!((0.0..=2.0).contains(&dist));
    }
}
