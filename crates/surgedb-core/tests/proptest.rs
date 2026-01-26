//! Property-based tests for SurgeDB
//!
//! These tests verify important invariants:
//! - Search never crashes regardless of input
//! - Deleted items never appear in search results
//! - Insert followed by search finds the vector
//! - Dimension mismatches are properly rejected

use proptest::prelude::*;
use surgedb_core::{Config, DistanceMetric, QuantizedConfig, QuantizedVectorDb, VectorDb};

/// Generate a valid vector ID
fn arb_id() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_]{1,64}".prop_map(|s| s)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // =========================================================================
    // Invariant: Search never panics
    // =========================================================================
    #[test]
    fn prop_search_never_crashes(
        dims in 4usize..128,
        num_vectors in 1usize..50,
        k in 1usize..20,
    ) {
        let config = Config {
            dimensions: dims,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut db = VectorDb::new(config).unwrap();

        // Insert some vectors
        for i in 0..num_vectors {
            let vector: Vec<f32> = (0..dims).map(|j| ((i * j) as f32).sin()).collect();
            let _ = db.insert(format!("v{}", i), &vector, None);
        }

        // Search with a random query
        let query: Vec<f32> = (0..dims).map(|i| (i as f32).cos()).collect();
        let result = db.search(&query, k, None);

        // Should always succeed (may return fewer than k if db has fewer vectors)
        prop_assert!(result.is_ok());
        let results = result.unwrap();
        prop_assert!(results.len() <= k);
        prop_assert!(results.len() <= num_vectors);
    }

    // =========================================================================
    // Invariant: Deleted items never appear in search results
    // =========================================================================
    #[test]
    fn prop_deleted_items_never_in_results(
        dims in 4usize..64,
        num_vectors in 5usize..30,
        delete_idx in 0usize..5,
    ) {
        let config = Config {
            dimensions: dims,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut db = VectorDb::new(config).unwrap();

        // Insert vectors
        for i in 0..num_vectors {
            let vector: Vec<f32> = (0..dims).map(|j| ((i * j) as f32).sin()).collect();
            db.insert(format!("v{}", i), &vector, None).unwrap();
        }

        // Delete one vector
        let delete_id = format!("v{}", delete_idx % num_vectors);
        let deleted = db.delete(delete_id.clone());
        prop_assert!(deleted.is_ok());

        // Search
        let query: Vec<f32> = (0..dims).map(|i| (i as f32).cos()).collect();
        let results = db.search(&query, num_vectors, None).unwrap();

        // Deleted item should never appear
        for (id, _, _) in &results {
            prop_assert_ne!(id.to_string(), delete_id.as_str(), "Deleted item appeared in results");
        }
    }

    // =========================================================================
    // Invariant: Insert then search finds the vector (with high probability)
    // =========================================================================
    #[test]
    fn prop_insert_then_search_finds_vector(
        dims in 4usize..64,
        id in arb_id(),
    ) {
        let config = Config {
            dimensions: dims,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut db = VectorDb::new(config).unwrap();

        // Create a specific vector
        let vector: Vec<f32> = (0..dims).map(|i| (i as f32 * 0.1).sin()).collect();

        // Insert it
        db.insert(id.clone(), &vector, None).unwrap();

        // Search with the exact same vector
        let results = db.search(&vector, 1, None).unwrap();

        // Should find exactly the inserted vector
        prop_assert_eq!(results.len(), 1);
        prop_assert_eq!(results[0].0.to_string(), id);
        // Distance should be 0 (or very close) for cosine similarity
        prop_assert!(results[0].1 < 0.001, "Distance too large: {}", results[0].1);
    }

    // =========================================================================
    // Invariant: Dimension mismatch is properly rejected
    // =========================================================================
    #[test]
    fn prop_dimension_mismatch_rejected(
        dims in 4usize..64,
        wrong_dims in 4usize..64,
    ) {
        prop_assume!(dims != wrong_dims);

        let config = Config {
            dimensions: dims,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut db = VectorDb::new(config).unwrap();

        // Try to insert with wrong dimensions
        let wrong_vector: Vec<f32> = (0..wrong_dims).map(|i| i as f32).collect();
        let result = db.insert("wrong".to_string(), &wrong_vector, None);

        prop_assert!(result.is_err());
        match result {
            Err(surgedb_core::Error::DimensionMismatch { expected, got }) => {
                prop_assert_eq!(expected, dims);
                prop_assert_eq!(got, wrong_dims);
            }
            Err(e) => prop_assert!(false, "Wrong error type: {:?}", e),
            Ok(_) => prop_assert!(false, "Should have failed"),
        }
    }

    // =========================================================================
    // Invariant: Duplicate IDs are rejected
    // =========================================================================
    #[test]
    fn prop_duplicate_ids_rejected(
        dims in 4usize..32,
        id in arb_id(),
    ) {
        let config = Config {
            dimensions: dims,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut db = VectorDb::new(config).unwrap();

        let vector1: Vec<f32> = (0..dims).map(|i| i as f32).collect();
        let vector2: Vec<f32> = (0..dims).map(|i| (i * 2) as f32).collect();

        // First insert should succeed
        let result1 = db.insert(id.clone(), &vector1, None);
        prop_assert!(result1.is_ok());

        // Second insert with same ID should fail
        let result2 = db.insert(id.clone(), &vector2, None);
        prop_assert!(result2.is_err());
        match result2 {
            Err(surgedb_core::Error::DuplicateId(_)) => {}
            Err(e) => prop_assert!(false, "Wrong error type: {:?}", e),
            Ok(_) => prop_assert!(false, "Should have failed"),
        }
    }

    // =========================================================================
    // Invariant: Upsert always succeeds
    // =========================================================================
    #[test]
    fn prop_upsert_always_succeeds(
        dims in 4usize..32,
        id in arb_id(),
    ) {
        let config = Config {
            dimensions: dims,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut db = VectorDb::new(config).unwrap();

        let vector1: Vec<f32> = (0..dims).map(|i| i as f32).collect();
        let vector2: Vec<f32> = (0..dims).map(|i| (i * 2) as f32).collect();

        // First upsert
        let result1 = db.upsert(id.clone(), &vector1, None);
        prop_assert!(result1.is_ok());

        // Second upsert with same ID should also succeed
        let result2 = db.upsert(id.clone(), &vector2, None);
        prop_assert!(result2.is_ok());
    }

    // =========================================================================
    // Invariant: Quantized DB maintains search quality
    // =========================================================================
    #[test]
    fn prop_quantized_db_finds_vectors(
        dims in 32usize..128,
        num_vectors in 10usize..50,
    ) {
        let config = QuantizedConfig {
            dimensions: dims,
            distance_metric: DistanceMetric::Cosine,
            quantization: surgedb_core::QuantizationType::SQ8,
            ..Default::default()
        };

        let mut db = QuantizedVectorDb::new(config).unwrap();

        // Insert vectors
        for i in 0..num_vectors {
            let vector: Vec<f32> = (0..dims).map(|j| ((i * j) as f32 * 0.1).sin()).collect();
            db.insert(format!("v{}", i), &vector, None).unwrap();
        }

        // Search with a known vector
        let target_idx = num_vectors / 2;
        let query: Vec<f32> = (0..dims).map(|j| ((target_idx * j) as f32 * 0.1).sin()).collect();

        let results = db.search(&query, 5, None).unwrap();

        // Should find some results
        prop_assert!(!results.is_empty());

        // The exact match should be in top results (allowing for quantization error)
        let found = results.iter().any(|(id, _, _)| id.to_string() == format!("v{}", target_idx));
        prop_assert!(found, "Expected vector not found in top 5 results");
    }
}
