//! Fuzz test for vector insertion
//!
//! Tests that inserting arbitrary data never panics or causes undefined behavior.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use surgedb_core::{Config, DistanceMetric, VectorDb};

/// Arbitrary input for insert operations
#[derive(Debug, Arbitrary)]
struct InsertInput {
    /// Vector ID (will be truncated to reasonable length)
    id: String,
    /// Vector dimensions (clamped to reasonable range)
    dimensions: u8,
    /// Raw vector data
    vector_data: Vec<f32>,
    /// Whether to include metadata
    include_metadata: bool,
    /// Simple metadata value
    metadata_value: Option<String>,
}

fuzz_target!(|input: InsertInput| {
    // Clamp dimensions to reasonable range (4-512)
    let dimensions = (input.dimensions as usize).clamp(4, 512);

    // Skip if vector data is too short
    if input.vector_data.len() < dimensions {
        return;
    }

    // Limit ID length
    let id = if input.id.len() > 256 {
        input.id[..256].to_string()
    } else if input.id.is_empty() {
        "default_id".to_string()
    } else {
        input.id
    };

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

    // Extract vector (take first `dimensions` elements)
    let vector: Vec<f32> = input
        .vector_data
        .iter()
        .take(dimensions)
        .map(|&x| {
            // Replace NaN/Inf with 0.0 to avoid issues
            if x.is_nan() || x.is_infinite() {
                0.0
            } else {
                x
            }
        })
        .collect();

    // Create metadata
    let metadata = if input.include_metadata {
        input
            .metadata_value
            .as_ref()
            .map(|s| serde_json::json!({ "value": s }))
    } else {
        None
    };

    // Attempt insert - should never panic
    let _ = db.insert(id.clone(), &vector, metadata.clone());

    // Try upsert as well
    let _ = db.upsert(id.clone(), &vector, metadata);

    // Try delete
    let _ = db.delete(id);
});
