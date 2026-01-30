//! Product Quantization (PQ) implementation
//!
//! PQ splits vectors into sub-vectors and quantizes each sub-vector independently
//! using k-means clustering. This allows for high compression ratios with
//! efficient asymmetric distance calculation (ADC).

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// Product Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqConfig {
    /// Number of sub-vectors (m)
    pub num_subvectors: usize,
    /// Number of centroids per sub-space (k) - usually 256 for u8 indices
    pub num_centroids: usize,
    /// Training sample size
    pub sample_size: usize,
    /// Max k-means iterations
    pub max_iterations: usize,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            num_subvectors: 8,
            num_centroids: 256,
            sample_size: 10000,
            max_iterations: 20,
        }
    }
}

/// Trained PQ Codebook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodebook {
    /// Configuration used
    pub config: PqConfig,
    /// Dimensions of original vectors
    pub dimensions: usize,
    /// Dimensions of each sub-vector
    pub subvector_dim: usize,
    /// Centroids: [m][k][sub_dim] - flattened as [m * k * sub_dim]
    pub centroids: Vec<f32>,
}

impl PqCodebook {
    /// Train a new codebook from a set of vectors
    pub fn train(vectors: &[Vec<f32>], config: PqConfig) -> Result<Self> {
        if vectors.is_empty() {
            return Err(Error::InvalidConfig("Training set cannot be empty".into()));
        }

        let dimensions = vectors[0].len();
        if !dimensions.is_multiple_of(config.num_subvectors) {
            return Err(Error::InvalidConfig(format!(
                "Dimensions ({}) must be divisible by num_subvectors ({})",
                dimensions, config.num_subvectors
            )));
        }

        let subvector_dim = dimensions / config.num_subvectors;
        let mut rng = rand::thread_rng();

        // Sample vectors if we have too many
        let sample_indices: Vec<usize> = if vectors.len() > config.sample_size {
            (0..vectors.len())
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, config.sample_size)
                .cloned()
                .collect()
        } else {
            (0..vectors.len()).collect()
        };

        let mut centroids =
            Vec::with_capacity(config.num_subvectors * config.num_centroids * subvector_dim);

        // Train k-means for each sub-space
        for m in 0..config.num_subvectors {
            // Extract sub-vectors
            let start = m * subvector_dim;
            let end = start + subvector_dim;

            let sub_vectors: Vec<Vec<f32>> = sample_indices
                .iter()
                .map(|&i| vectors[i][start..end].to_vec())
                .collect();

            // Run k-means
            let sub_centroids = kmeans(&sub_vectors, config.num_centroids, config.max_iterations);

            // Append flattened centroids
            for centroid in sub_centroids {
                centroids.extend(centroid);
            }
        }

        Ok(Self {
            config,
            dimensions,
            subvector_dim,
            centroids,
        })
    }

    /// Encode a vector into PQ codes (byte array)
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dimensions);
        let mut codes = Vec::with_capacity(self.config.num_subvectors);

        for m in 0..self.config.num_subvectors {
            let start = m * self.subvector_dim;
            let end = start + self.subvector_dim;
            let sub_vector = &vector[start..end];

            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut nearest_idx = 0;

            for k in 0..self.config.num_centroids {
                let centroid_start = (m * self.config.num_centroids + k) * self.subvector_dim;
                let centroid = &self.centroids[centroid_start..centroid_start + self.subvector_dim];

                let dist = crate::distance::euclidean_distance(sub_vector, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = k;
                }
            }

            codes.push(nearest_idx as u8);
        }

        codes
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.config.num_subvectors);
        let mut vector = Vec::with_capacity(self.dimensions);

        for (m, &code) in codes.iter().enumerate() {
            let k = code as usize;
            let centroid_start = (m * self.config.num_centroids + k) * self.subvector_dim;
            let centroid = &self.centroids[centroid_start..centroid_start + self.subvector_dim];
            vector.extend_from_slice(centroid);
        }

        vector
    }

    /// Pre-compute distance table for a query (ADC - Asymmetric Distance Computation)
    /// Returns a table of size [num_subvectors * num_centroids] containing distances
    pub fn precompute_adc(&self, query: &[f32], metric: DistanceMetric) -> Vec<f32> {
        assert_eq!(query.len(), self.dimensions);
        let mut table = Vec::with_capacity(self.config.num_subvectors * self.config.num_centroids);

        for m in 0..self.config.num_subvectors {
            let start = m * self.subvector_dim;
            let end = start + self.subvector_dim;
            let sub_query = &query[start..end];

            for k in 0..self.config.num_centroids {
                let centroid_start = (m * self.config.num_centroids + k) * self.subvector_dim;
                let centroid = &self.centroids[centroid_start..centroid_start + self.subvector_dim];

                // For ADC, we usually use squared Euclidean distance for efficiency
                // But we support multiple metrics.
                // Note: For Cosine, we assume normalized vectors, so DotProduct is close enough
                // provided we normalize sub-vectors? No, sub-vectors aren't normalized.
                // Standard PQ ADC uses Euclidean.

                let dist = match metric {
                    DistanceMetric::Cosine => {
                        // Cosine on sub-vectors is weird. Usually we use DotProduct or L2.
                        // Let's use L2 for now as standard PQ.
                        crate::distance::euclidean_distance(sub_query, centroid).powi(2)
                    }
                    DistanceMetric::Euclidean => {
                        crate::distance::euclidean_distance(sub_query, centroid).powi(2)
                    }
                    DistanceMetric::DotProduct => {
                        crate::distance::dot_product_distance(sub_query, centroid)
                    }
                };

                table.push(dist);
            }
        }

        table
    }

    /// Calculate distance using pre-computed ADC table
    #[inline]
    pub fn distance_adc(&self, codes: &[u8], adc_table: &[f32]) -> f32 {
        let mut dist = 0.0;
        for (m, &code) in codes.iter().enumerate() {
            let table_idx = m * self.config.num_centroids + (code as usize);
            dist += unsafe { *adc_table.get_unchecked(table_idx) };
        }
        dist
    }
}

/// Simple k-means clustering
fn kmeans(vectors: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let dim = vectors[0].len();
    let mut rng = rand::thread_rng();

    // Initialize centroids randomly
    let mut centroids: Vec<Vec<f32>> = vectors.choose_multiple(&mut rng, k).cloned().collect();

    // Fallback if not enough vectors
    while centroids.len() < k {
        centroids.push(vec![0.0; dim]);
    }

    for _ in 0..max_iter {
        let mut sums = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        let mut changed = false;

        // Assign vectors to nearest centroid
        for vec in vectors {
            let mut min_dist = f32::MAX;
            let mut nearest = 0;

            for (i, centroid) in centroids.iter().enumerate() {
                let dist = crate::distance::euclidean_distance(vec, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    nearest = i;
                }
            }

            for j in 0..dim {
                sums[nearest][j] += vec[j];
            }
            counts[nearest] += 1;
        }

        // Update centroids
        for i in 0..k {
            if counts[i] > 0 {
                let new_centroid: Vec<f32> =
                    sums[i].iter().map(|&s| s / counts[i] as f32).collect();

                // Check if changed significantly (simple check)
                for j in 0..dim {
                    if (new_centroid[j] - centroids[i][j]).abs() > 1e-5 {
                        changed = true;
                    }
                }
                centroids[i] = new_centroid;
            } else {
                // Re-initialize empty cluster with random vector
                if let Some(v) = vectors.choose(&mut rng) {
                    centroids[i] = v.clone();
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_training_and_encoding() {
        // Generate random training data
        let dim = 16;
        let mut vectors = Vec::new();
        for i in 0..100 {
            let vec: Vec<f32> = (0..dim).map(|j| (i + j) as f32).collect();
            vectors.push(vec);
        }

        let config = PqConfig {
            num_subvectors: 4, // 4 subvectors of 4 dims
            num_centroids: 16,
            sample_size: 100,
            max_iterations: 5,
        };

        let codebook = PqCodebook::train(&vectors, config).unwrap();

        // Test encoding
        let vec = &vectors[0];
        let codes = codebook.encode(vec);
        assert_eq!(codes.len(), 4);

        // Test decoding
        let decoded = codebook.decode(&codes);
        assert_eq!(decoded.len(), dim);

        // Decoded should be somewhat close to original (but lossy)
        // Just check it doesn't crash and returns valid floats
        for x in decoded {
            assert!(!x.is_nan());
        }
    }
}
