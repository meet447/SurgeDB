# SurgeDB Bindings

UniFFI-based bindings for SurgeDB, enabling usage from Python, Swift, and Kotlin.

## Architecture

```
Python / Swift / Kotlin
         │
         ▼
┌─────────────────────────┐
│   surgedb-bindings      │  ← Stable API (this crate)
│   (SurgeClient)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│     surgedb-core        │  ← Can change freely (internal)
│   (internal engine)     │
└─────────────────────────┘
```

## Stable API

The binding layer provides a **stable API** that won't change even when `surgedb-core` internals are optimized. This means:

- Internal SIMD optimizations → No binding changes needed
- HNSW algorithm improvements → No binding changes needed
- New quantization methods → Add new enum variant (backward compatible)
- Bug fixes → No binding changes needed

## Building

```bash
# Build the Rust library
cargo build --release -p surgedb-bindings
```

## Generating Python Bindings

```bash
cd crates/surgedb-bindings

# On macOS
make generate-python

# On Linux
make generate-python-linux
```

This will create Python bindings in the `python/` directory.

## Python Usage

```python
from surgedb import SurgeClient, SurgeConfig, DistanceMetric, Quantization

# Simple in-memory database
db = SurgeClient.new_in_memory(dimensions=384)

# Insert vectors
db.insert("doc1", [0.1, 0.2, ...], '{"title": "Hello World"}')
db.insert("doc2", [0.3, 0.4, ...], '{"title": "Goodbye World"}')

# Search
results = db.search([0.1, 0.2, ...], k=5)
for r in results:
    print(f"{r.id}: {r.score}")

# With quantization and persistence
config = SurgeConfig(
    dimensions=768,
    distance_metric=DistanceMetric.COSINE,
    quantization=Quantization.SQ8,
    persistent=True,
    data_path="./my_db"
)
db = SurgeClient.open("./my_db", config)
```

## API Reference

### SurgeClient

| Method | Description |
|--------|-------------|
| `new_in_memory(dimensions)` | Create in-memory database |
| `open(path, config)` | Open persistent database |
| `insert(id, vector, metadata)` | Insert a vector |
| `upsert(id, vector, metadata)` | Insert or update |
| `upsert_batch(entries)` | Batch insert/update |
| `delete(id)` | Delete by ID |
| `get(id)` | Get vector by ID |
| `search(query, k)` | Find k nearest neighbors |
| `search_with_filter(query, k, filter)` | Filtered search |
| `list(offset, limit)` | List vector IDs |
| `len()` | Get vector count |
| `is_empty()` | Check if empty |
| `stats()` | Get database statistics |
| `checkpoint()` | Create snapshot |
| `sync()` | Force sync to disk |

### Enums

```python
# Distance metrics
DistanceMetric.COSINE      # Cosine similarity (default)
DistanceMetric.EUCLIDEAN   # Euclidean distance
DistanceMetric.DOT_PRODUCT # Dot product

# Quantization types
Quantization.NONE    # No quantization (full precision)
Quantization.SQ8     # 4x compression
Quantization.BINARY  # 32x compression
```

### Filters

```python
from surgedb import SearchFilter

# Exact match
f = SearchFilter.Exact(field="category", value_json='"tech"')

# One of many values
f = SearchFilter.OneOf(field="tag", values_json=['"ai"', '"ml"'])

# Logical AND
f = SearchFilter.And(filters=[filter1, filter2])

# Logical OR
f = SearchFilter.Or(filters=[filter1, filter2])
```

## Swift / Kotlin

Swift and Kotlin bindings can be generated similarly:

```bash
# Generate Swift bindings
cargo run --release -p surgedb-bindings --bin uniffi-bindgen -- \
    generate \
    --library ../target/release/libsurgedb_bindings.dylib \
    --language swift \
    --out-dir swift/

# Generate Kotlin bindings
cargo run --release -p surgedb-bindings --bin uniffi-bindgen -- \
    generate \
    --library ../target/release/libsurgedb_bindings.so \
    --language kotlin \
    --out-dir kotlin/
```

## Development

### Adding New Methods

1. Add method to `src/surgedb.udl`
2. Implement in `src/lib.rs`
3. Regenerate bindings
4. Update language-specific tests

### Testing

```bash
# Rust tests
cargo test -p surgedb-bindings

# Python tests (after generating bindings)
make test-python
```
