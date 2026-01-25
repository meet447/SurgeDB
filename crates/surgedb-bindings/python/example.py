"""
SurgeDB Python Example

This example shows how to use SurgeDB from Python via UniFFI bindings.
"""

# NOTE: After generating bindings, import like this:
# from surgedb import SurgeClient, SearchFilter, DistanceMetric, Quantization, SurgeConfig

def main():
    """Demonstrate SurgeDB Python usage."""
    
    # Import the generated bindings
    try:
        from surgedb import (
            SurgeClient, 
            SurgeConfig, 
            SearchFilter,
            DistanceMetric, 
            Quantization,
            VectorEntry,
            version,
            system_info,
        )
    except ImportError:
        print("SurgeDB bindings not found!")
        print("Generate them first with: make generate-python")
        print("\nShowing what the API will look like:\n")
        show_api_preview()
        return

    # Print version info
    print(f"SurgeDB Version: {version()}")
    print(f"System Info: {system_info()}")
    print()

    # ==========================================================================
    # Example 1: Simple In-Memory Database
    # ==========================================================================
    print("=" * 60)
    print("Example 1: In-Memory Vector Database")
    print("=" * 60)

    # Create a new in-memory database with 384 dimensions (MiniLM embeddings)
    db = SurgeClient.new_in_memory(dimensions=384)

    # Insert some vectors with metadata
    for i in range(100):
        vector = [float(i * j % 100) / 100.0 for j in range(384)]
        metadata = f'{{"id": {i}, "category": "{["tech", "science", "art"][i % 3]}"}}'
        db.insert(f"doc_{i}", vector, metadata)

    print(f"Inserted {db.len()} vectors")
    print(f"Stats: {db.stats()}")

    # Search for similar vectors
    query = [0.5] * 384
    results = db.search(query, k=5)

    print("\nTop 5 similar vectors:")
    for result in results:
        print(f"  ID: {result.id}, Score: {result.score:.4f}")

    # ==========================================================================
    # Example 2: Persistent Database with Quantization
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Persistent Database with SQ8 Quantization")
    print("=" * 60)

    config = SurgeConfig(
        dimensions=768,  # BERT-sized embeddings
        distance_metric=DistanceMetric.COSINE,
        quantization=Quantization.SQ8,  # 4x memory reduction
        persistent=True,
        data_path="./my_surgedb_data"
    )

    db = SurgeClient.open("./my_surgedb_data", config)

    # Batch insert
    entries = [
        VectorEntry(
            id=f"batch_{i}",
            vector=[float(i) / 100.0] * 768,
            metadata_json=f'{{"batch": true, "index": {i}}}'
        )
        for i in range(50)
    ]
    db.upsert_batch(entries)

    print(f"Batch inserted {len(entries)} vectors")
    print(f"Compression ratio: {db.stats().compression_ratio:.2f}x")

    # Checkpoint to disk
    db.checkpoint()
    print("Checkpointed to disk")

    # ==========================================================================
    # Example 3: Filtered Search
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Example 3: Filtered Search")
    print("=" * 60)

    # Create filter: category == "tech"
    filter = SearchFilter.Exact(field="category", value_json='"tech"')

    results = db.search_with_filter(query, k=5, filter=filter)
    print(f"Found {len(results)} results matching filter")

    # Complex filter: (category == "tech" OR category == "science")
    complex_filter = SearchFilter.Or(filters=[
        SearchFilter.Exact(field="category", value_json='"tech"'),
        SearchFilter.Exact(field="category", value_json='"science"'),
    ])

    results = db.search_with_filter(query, k=10, filter=complex_filter)
    print(f"Found {len(results)} results with complex filter")


def show_api_preview():
    """Show what the API will look like when bindings are available."""
    
    api_preview = '''
    # SurgeDB Python API Preview
    # ==========================

    from surgedb import SurgeClient, SurgeConfig, DistanceMetric, Quantization

    # Create in-memory database
    db = SurgeClient.new_in_memory(dimensions=384)

    # Or with full config
    config = SurgeConfig(
        dimensions=768,
        distance_metric=DistanceMetric.COSINE,
        quantization=Quantization.SQ8,
        persistent=True,
        data_path="./data"
    )
    db = SurgeClient.open("./data", config)

    # Insert vectors
    db.insert("vec1", [0.1, 0.2, ...], '{"category": "tech"}')
    db.upsert("vec1", [0.15, 0.25, ...], '{"category": "tech", "updated": true}')

    # Batch insert
    from surgedb import VectorEntry
    entries = [
        VectorEntry(id="v1", vector=[...], metadata_json='{"key": "value"}'),
        VectorEntry(id="v2", vector=[...], metadata_json=None),
    ]
    db.upsert_batch(entries)

    # Search
    results = db.search([0.1, 0.2, ...], k=10)
    for r in results:
        print(f"{r.id}: {r.score} - {r.metadata_json}")

    # Filtered search
    from surgedb import SearchFilter
    filter = SearchFilter.Exact(field="category", value_json='"tech"')
    results = db.search_with_filter(query, k=10, filter=filter)

    # Get stats
    stats = db.stats()
    print(f"Vectors: {stats.vector_count}")
    print(f"Memory: {stats.memory_usage_bytes} bytes")
    print(f"Compression: {stats.compression_ratio}x")

    # Persistence
    db.checkpoint()  # Create snapshot
    db.sync()        # Force sync to disk
    '''
    print(api_preview)


if __name__ == "__main__":
    main()
