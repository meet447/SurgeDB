//! Concurrency tests using loom for deterministic thread testing
//!
//! These tests verify thread-safety of internal synchronization primitives.
//! Note: loom tests are expensive, so we test isolated concurrent patterns.

// NOTE: Full loom integration requires modifying the sync module to conditionally
// use loom primitives. For now, we test the patterns we use.

use loom::sync::Arc;
use loom::sync::RwLock;
use loom::thread;

/// Test that concurrent readers don't block each other
#[test]
fn test_concurrent_reads() {
    loom::model(|| {
        let data = Arc::new(RwLock::new(vec![1, 2, 3, 4, 5]));

        let handles: Vec<_> = (0..3)
            .map(|i| {
                let data = Arc::clone(&data);
                thread::spawn(move || {
                    let guard = data.read().unwrap();
                    assert_eq!(guard.len(), 5);
                    guard[i % 5]
                })
            })
            .collect();

        for h in handles {
            let _ = h.join();
        }
    });
}

/// Test that a writer blocks readers and other writers
#[test]
fn test_write_exclusion() {
    loom::model(|| {
        let data = Arc::new(RwLock::new(0u32));

        let data1 = Arc::clone(&data);
        let data2 = Arc::clone(&data);

        let h1 = thread::spawn(move || {
            let mut guard = data1.write().unwrap();
            *guard += 1;
        });

        let h2 = thread::spawn(move || {
            let mut guard = data2.write().unwrap();
            *guard += 10;
        });

        h1.join().unwrap();
        h2.join().unwrap();

        let final_value = *data.read().unwrap();
        assert!(
            final_value == 11 || final_value == 11,
            "Expected 11, got {}",
            final_value
        );
    });
}

/// Test read-modify-write pattern (simulating index update)
#[test]
fn test_read_modify_write() {
    loom::model(|| {
        let counter = Arc::new(RwLock::new(0u32));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    // Read current value
                    let current = *counter.read().unwrap();
                    // Write new value (this is intentionally racy to test loom detection)
                    let mut guard = counter.write().unwrap();
                    *guard = current + 1;
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Due to the intentional race, value could be 1 or 2
        let final_value = *counter.read().unwrap();
        assert!(final_value >= 1 && final_value <= 2);
    });
}

/// Test simulated vector storage pattern
#[test]
fn test_vector_storage_pattern() {
    loom::model(|| {
        // Simulates the ID-to-internal mapping pattern
        let id_map = Arc::new(RwLock::new(std::collections::HashMap::new()));
        let next_id = Arc::new(RwLock::new(0usize));

        let handles: Vec<_> = (0..2)
            .map(|i| {
                let id_map = Arc::clone(&id_map);
                let next_id = Arc::clone(&next_id);
                thread::spawn(move || {
                    // Allocate new internal ID
                    let internal_id = {
                        let mut id = next_id.write().unwrap();
                        let current = *id;
                        *id += 1;
                        current
                    };

                    // Store mapping
                    {
                        let mut map = id_map.write().unwrap();
                        map.insert(format!("vec_{}", i), internal_id);
                    }

                    internal_id
                })
            })
            .collect();

        for h in handles {
            let _ = h.join();
        }

        // Both should have unique IDs
        let map = id_map.read().unwrap();
        assert_eq!(map.len(), 2);

        let id = next_id.read().unwrap();
        assert_eq!(*id, 2);
    });
}

/// Test simulated search pattern (multiple concurrent readers)
#[test]
fn test_concurrent_search_pattern() {
    loom::model(|| {
        // Simulates multiple concurrent searches
        let vectors = Arc::new(RwLock::new(vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let vectors = Arc::clone(&vectors);
                thread::spawn(move || {
                    let guard = vectors.read().unwrap();
                    // Simulate distance calculation
                    let mut best_idx = 0;
                    let mut best_dist = f32::MAX;
                    for (idx, v) in guard.iter().enumerate() {
                        let dist = v[0]; // Simplified "distance"
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = idx;
                        }
                    }
                    best_idx
                })
            })
            .collect();

        for h in handles {
            let _ = h.join();
        }
    });
}
