use serde_json::json;
use surgedb_core::filter::Filter;
use surgedb_core::{Config, VectorDb};

#[test]
fn test_metadata_filtering() {
    let config = Config {
        dimensions: 4,
        ..Default::default()
    };
    let mut db = VectorDb::new(config).unwrap();

    // Insert data
    // vec1: "books"
    db.insert(
        "vec1",
        &[1.0, 0.0, 0.0, 0.0],
        Some(json!({"category": "books"})),
    )
    .unwrap();
    // vec2: "movies" (similar to vec1)
    db.insert(
        "vec2",
        &[1.0, 0.1, 0.0, 0.0],
        Some(json!({"category": "movies"})),
    )
    .unwrap();
    // vec3: "books" (orthogonal)
    db.insert(
        "vec3",
        &[0.0, 1.0, 0.0, 0.0],
        Some(json!({"category": "books"})),
    )
    .unwrap();

    // Search without filter (query close to vec1/vec2)
    let results = db.search(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0.as_str(), "vec1");
    assert_eq!(results[1].0.as_str(), "vec2");

    // Search WITH filter (category="books")
    let filter = Filter::Exact("category".to_string(), json!("books"));
    let results_filtered = db.search(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter)).unwrap();

    // Should only return vec1 and vec3
    assert_eq!(results_filtered.len(), 2);
    assert_eq!(results_filtered[0].0.as_str(), "vec1");
    assert_eq!(results_filtered[1].0.as_str(), "vec3");

    // Ensure vec2 is NOT in results
    assert!(results_filtered
        .iter()
        .all(|(id, _, _)| id.as_str() != "vec2"));
}

#[test]
fn test_complex_filtering() {
    let config = Config {
        dimensions: 2,
        ..Default::default()
    };
    let mut db = VectorDb::new(config).unwrap();

    db.insert("v1", &[1.0, 0.0], Some(json!({"tag": "A", "val": 10})))
        .unwrap();
    db.insert("v2", &[1.0, 0.0], Some(json!({"tag": "B", "val": 10})))
        .unwrap();
    db.insert("v3", &[1.0, 0.0], Some(json!({"tag": "A", "val": 20})))
        .unwrap();

    // Filter: tag="A" AND val=10
    let filter = Filter::And(vec![
        Filter::Exact("tag".to_string(), json!("A")),
        Filter::Exact("val".to_string(), json!(10)),
    ]);

    let results = db.search(&[1.0, 0.0], 10, Some(&filter)).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.as_str(), "v1");
}
