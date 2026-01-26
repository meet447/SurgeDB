//! Fuzz test for vector search
//!
//! Tests that searching with arbitrary queries never panics.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use surgedb_core::{Config, DistanceMetric, VectorDb};

/// Arbitrary input for search operations
#[derive(Debug, Arbitrary)]
struct SearchInput {
    /// Number of vectors to insert (clamped)
    num_vectors: u8,
    /// Vector dimensions (clamped)
    dimensions: u8,
    /// Random seed data for vectors
    seed_data: Vec<f32>,
    /// Query vector data
    query_data: Vec<f32>,
    /// Number of results to request
    k: u16,
}

fuzz_target!(|input: SearchInput| {
    // Clamp dimensions and num_vectors
    let dimensions = (input.dimensions as usize).clamp(4, 256);
    let num_vectors = (input.num_vectors as usize).clamp(1, 100);
    let k = (input.k as usize).clamp(1, 50);

    // Skip if not enough seed data
    if input.seed_data.len() < dimensions * num_vectors {
        return;
    }
    if input.query_data.len() < dimensions {
        return;
    }

    // Create database
    let config = Config {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let db = match VectorDb::new(config) {
        Ok(db) => db,
        Err(_) => return,
    };

    // Insert vectors
    for i in 0..num_vectors {
        let start = i * dimensions;
        let end = start + dimensions;
        let vector: Vec<f32> = input.seed_data[start..end]
            .iter()
            .map(|&x| {
                if x.is_nan() || x.is_infinite() {
                    0.0
                } else {
                    x
                }
            })
            .collect();

        let _ = db.insert(format!("vec_{}", i), &vector, None);
    }

    // Create query vector
    let query: Vec<f32> = input
        .query_data
        .iter()
        .take(dimensions)
        .map(|&x| {
            if x.is_nan() || x.is_infinite() {
                0.0
            } else {
                x
            }
        })
        .collect();

    // Search - should never panic
    let _ = db.search(&query, k, None);
});
