# SurgeDB Python

[![PyPI](https://img.shields.io/pypi/v/surgedb.svg)](https://pypi.org/project/surgedb/)
[![Python versions](https://img.shields.io/pypi/pyversions/surgedb.svg)](https://pypi.org/project/surgedb/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/meet447/SurgeDB/blob/main/LICENSE)

**SurgeDB** is a high-performance, embedded vector database for Python. It runs entirely locally—no Docker containers, no external APIs, and no complex setup required.

Built in **Rust** and powered by SIMD-accelerated HNSW indices, SurgeDB offers **millisecond-latency** vector search with a minimal memory footprint.

---

## Key Features

* 🚀 **Blazing Fast**: Hand-tuned AVX-512 and NEON kernels for maximum throughput.
* 🧠 **Memory Efficient**: Built-in SQ8 (4x) and Binary (32x) quantization. Index millions of vectors on a laptop.
* 📦 **Embedded**: Runs in-process. Just `pip install` and go.
* 💾 **Persistent**: ACID-compliant storage with Write-Ahead Logs (WAL) and crash-safe snapshots.
* 🔍 **Rich Filtering**: Filter search results by metadata (exact match, comparison, logical operators).

---

## Installation

```bash
pip install surgedb
```

---

## Quick Start

```python
from surgedb import SurgeClient, SurgeConfig, DistanceMetric, Quantization

# 1. Initialize a persistent database
config = SurgeConfig(
    dimensions=384,                     # e.g., for all-MiniLM-L6-v2
    distance_metric=DistanceMetric.COSINE,
    quantization=Quantization.SQ8,      # 4x compression
    persistent=True,
    data_path="./my_vector_db"
)
db = SurgeClient.open("./my_vector_db", config)

# 2. Insert data (ID, Vector, Metadata)
db.insert(
    "doc_1", 
    [0.1, 0.2, 0.3, ...],               # 384-dim list or numpy array
    '{"title": "How to train your dragon", "tag": "movie"}'
)

# 3. Search with metadata filtering
results = db.search_with_filter(
    query=[0.1, 0.2, 0.3, ...], 
    k=5, 
    filter={'Exact': {'field': 'tag', 'value': 'movie'}}
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score:.4f}")
```

---

## Advanced Usage

### Batch Operations

For maximum write throughput, use `upsert_batch`.

```python
vectors = []
for i in range(1000):
    vectors.append({
        "id": f"vec_{i}",
        "vector": [0.1] * 384,
        "metadata": {"index": i}
    })

db.upsert_batch(vectors)
```

### Metadata Filtering

SurgeDB supports a structured query language for filtering.

```python
from surgedb import SearchFilter

# Find movies released after 2020 OR in the "Sci-Fi" genre
filter_query = SearchFilter.Or([
    SearchFilter.Comparison(field="year", operator="gt", value=2020),
    SearchFilter.Exact(field="genre", value="Sci-Fi")
])

results = db.search_with_filter(query_vec, 10, filter_query)
```

---

## Performance

SurgeDB is designed to outperform pure-Python solutions and compete with heavy C++ vector stores, while remaining lightweight.

| Metric | Performance |
| :--- | :--- |
| **Search Latency** | < 1ms (1M vectors, SQ8) |
| **Indexing Speed** | ~20k vectors/sec |
| **Memory (1M vectors)** | ~120MB (SQ8) vs ~4GB (Float32) |
| **Cold Start** | < 50ms |

---

## Development

If you want to contribute to the bindings or build from source:

### Prerequisites

* Rust toolchain (stable)
* Python 3.7+
* `maturin` build tool

### Building from Source

```bash
# Install maturin
pip install maturin

# Build and install locally
maturin develop --release -m crates/surgedb-bindings/Cargo.toml
```

### Generating Bindings (UniFFI)

The bindings are automatically generated using UniFFI.

```bash
cd crates/surgedb-bindings
make generate-python
```
