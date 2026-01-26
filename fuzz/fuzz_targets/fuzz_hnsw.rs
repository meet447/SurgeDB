//! Fuzz test for HNSW index operations
//!
//! Tests the robustness of the HNSW graph under arbitrary operations.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use surgedb_core::{Config, DistanceMetric, VectorDb};

/// Operation type for the fuzzer
#[derive(Debug, Arbitrary)]
enum Operation {
    Insert { id: u16, vector_seed: u32 },
    Search { query_seed: u32, k: u8 },
    Delete { id: u16 },
    Upsert { id: u16, vector_seed: u32 },
}

/// Arbitrary input for HNSW operations
#[derive(Debug, Arbitrary)]
struct HnswInput {
    /// Dimensions (clamped)
    dimensions: u8,
    /// Sequence of operations to perform
    operations: Vec<Operation>,
}

/// Generate a deterministic vector from a seed
fn generate_vector(seed: u32, dimensions: usize) -> Vec<f32> {
    let mut vector = Vec::with_capacity(dimensions);
    let mut state = seed;
    for _ in 0..dimensions {
        // Simple LCG for reproducible randomness
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let value = ((state >> 16) as f32) / 32768.0 - 1.0;
        vector.push(value);
    }
    vector
}

fuzz_target!(|input: HnswInput| {
    // Clamp dimensions
    let dimensions = (input.dimensions as usize).clamp(4, 128);

    // Limit operations
    if input.operations.len() > 1000 {
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

    // Execute operations
    for op in input.operations {
        match op {
            Operation::Insert { id, vector_seed } => {
                let vector = generate_vector(vector_seed, dimensions);
                let _ = db.insert(format!("vec_{}", id), &vector, None);
            }
            Operation::Search { query_seed, k } => {
                let query = generate_vector(query_seed, dimensions);
                let k = (k as usize).clamp(1, 20);
                let _ = db.search(&query, k, None);
            }
            Operation::Delete { id } => {
                let _ = db.delete(format!("vec_{}", id));
            }
            Operation::Upsert { id, vector_seed } => {
                let vector = generate_vector(vector_seed, dimensions);
                let _ = db.upsert(format!("vec_{}", id), &vector, None);
            }
        }
    }

    // Final search to ensure graph is still valid
    let query = generate_vector(12345, dimensions);
    let _ = db.search(&query, 10, None);
});
