# SurgeDB

![Rust CI](https://github.com/meet447/SurgeDB/actions/workflows/rust.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/surgedb.svg)](https://pypi.org/project/surgedb/)
[![npm](https://img.shields.io/npm/v/surgedb-wasm.svg)](https://www.npmjs.com/package/surgedb-wasm)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)

**The SIMD-powered, ultra-lightweight vector database for the Edge.**

---

## Why SurgeDB?

Most vector databases are designed for massive cloud clusters. SurgeDB is designed for **efficiency**.

* **Ultra-Low Footprint**: While production databases like **Qdrant** or **Milvus** often require 500MB+ just to idle, SurgeDB can index **100k vectors using only ~39MB of RAM** (with SQ8).
* **Edge Ready**: Optimized specifically for Apple Silicon (NEON) and modern x86_64 (AVX-512).
* **Zero Dependencies**: Written in pure Rust. No Python runtime, no Docker containers required.

---

## Features

* **Adaptive HNSW Indexing**: High-speed approximate nearest neighbor search.
* **SIMD Optimized**: Hand-tuned kernels for NEON (Apple Silicon) and AVX-512 (x86).
* **Plug-and-Play Quantization**:
  * **SQ8**: 4x compression with <1% accuracy loss.
  * **Binary**: 32x compression for massive datasets.
* **ACID-Compliant Persistence**: Write-Ahead Log (WAL) and Snapshots for crash-safe data.
* **Mmap Support**: Disk-resident vectors for datasets larger than RAM.
* **Collections & Metadata**: Manage multiple collections with rich JSON metadata.
* **Metadata Filtering**: Filter search results using structured queries (e.g., `category == "books"`).
* **HTTP Server**: Built-in high-performance Axum server for easy deployment.

---

## Architecture

SurgeDB uses a hybrid storage engine to balance speed and durability.

```mermaid
graph TD
    Client["Client / HTTP"] --> API["Axum API Layer"]
    API --> Engine["Core Vector Engine"]
    Engine --> HNSW["HNSW Index (RAM)"]
    Engine --> Storage{"Storage Backend"}
    Storage -->|Hot Data| WAL["Write-Ahead Log"]
    Storage -->|Cold Data| Mmap["Mmap Vectors (Disk)"]
    Storage -->|Recovery| Snap["Snapshots"]
```

---

## Performance Snapshot

We validate every build for **Recall** (accuracy) and **Latency** across different vector sizes.

### Heavy Workload (768 dim - SigLIP)

*Comparison vs Qdrant on 5,000 points with heavy metadata.*

| Operation | Qdrant (Local) | SurgeDB (Local) | Comparison |
| :--- | :--- | :--- | :--- |
| **Create Collection** | 64.58 ms | **2.08 ms** | **SurgeDB ~31x faster** |
| **Search Avg** | 3.52 ms | **0.64 ms** | **SurgeDB ~5.5x faster** |
| **Retrieve by ID** | 7.03 ms | **0.68 ms** | **SurgeDB ~10x faster** |
| **Bulk Upsert (5k)** | **2,384 ms** | 5,257 ms | Qdrant ~2.2x faster |

### HNSW Accuracy & Recall

*Measured on 2k vectors (128 dim) against exact brute-force truth.*

| Metric | Full Precision HNSW | SQ8 Quantized HNSW | Result |
| :--- | :--- | :--- | :--- |
| **Top-10 Recall** | **99.20%** | **98.80%** | **Near Perfect** |
| **Rank Consistency**| **86.00%** | 76.50% | **Very High** |

### Standard Workload (128 dim)

| Mode | Recall @ 10 | Latency (Avg) | Compression |
| :--- | :--- | :--- | :--- |
| **HNSW (In-Memory)** | **99.2%** | **0.15 ms** | 1x |
| **SQ8 (Quantized)** | **98.8%** | **0.22 ms** | **3.76x** |
| **Binary (1-bit)** | 25.7% | 0.02 ms | 32.0x |

### Scaling Benchmarks (384 dim)

| Dataset Size | Mode | Latency (Avg) | Throughput | Memory Usage |
| :--- | :--- | :--- | :--- | :--- |
| **50,000 Vectors** | **HNSW** | **0.78 ms** | 1,282 QPS | ~87 MB |
| **100,000 Vectors** | **SQ8 (Indexed)** | **1.04 ms** | 964 QPS | **~96 MB** |

*Performance measured on M2 Mac. HNSW provides sub-millisecond search at scale, while SQ8 offers massive memory savings.*

---

## Installation

### 🦀 Rust (Core)

Add SurgeDB to your `Cargo.toml`:

```toml
[dependencies]
surgedb-core = { git = "https://github.com/meet447/SurgeDB" }
```

### 📚 Clients & Bindings

For detailed installation, usage, and API references, please read the specific package guides:

* **🐍 Python**: [Read the Python Guide & Docs](./crates/surgedb-bindings/README.md)
* **🌐 JavaScript / WASM**: [Read the WASM Guide & Docs](./crates/surgedb-wasm/README.md)

---

## Examples

Standalone examples for different SDKs can be found in the [examples/](./examples) directory:

* [Rust Basic Example](./examples/rust/basic.rs)
* [Python Basic Example](./examples/python/basic.py)

---

## Quick Start (WASM / Browser)

SurgeDB runs entirely in the browser using WebAssembly.

```javascript
import init, { SurgeDB } from './pkg/surgedb_wasm.js';

async function run() {
    await init();
    
    // 1. Create DB
    const db = new SurgeDB(384);
    
    // 2. Insert
    db.insert("doc_1", new Float32Array([0.1, ...]), { title: "Hello" });
    
    // 3. Search
    const results = db.search(new Float32Array([0.1, ...]), 5);
    console.log(results);
}
```

---

## Quick Start (Rust)

```rust
use surgedb_core::{PersistentVectorDb, PersistentConfig, DistanceMetric};

fn main() {
    // 1. Setup Persistent Database
    let config = PersistentConfig {
        dimensions: 384, // MiniLM size
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut db = PersistentVectorDb::open("./surgedb_data", config).unwrap();

    // 2. Insert Vector with Metadata
    let vec = vec![0.1; 384];
    let meta = serde_json::json!({"title": "SurgeDB Guide"});
    db.insert("doc_1", &vec, Some(meta)).unwrap();

    // 3. Search
    let query = vec![0.1; 384];
    let results = db.search(&query, 5).unwrap();
    
    println!("Found match: {} (meta: {:?})", results[0].0, results[0].2);
}
```

---

## HTTP Server

SurgeDB includes a high-performance HTTP server powered by **Axum**.

### Start the Server

```bash
cargo run --release -p surgedb-server
# Server listening on 0.0.0.0:3000
```

### API Usage

**Create Collection**

```bash
curl -X POST http://localhost:3000/collections \
  -H "Content-Type: application/json" \
  -d '{ 
    "name": "docs", 
    "dimensions": 384,
    "quantization": "SQ8" 
  }'
```

**Upsert Vector (Insert or Update)**

```bash
curl -X POST http://localhost:3000/collections/docs/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "id": "vec1",
    "vector": [0.1, 0.2, 0.3, ...],
    "metadata": { "category": "AI", "tags": ["fast"] }
  }'
```

**Batch Upsert (Bulk)**

```bash
curl -X POST http://localhost:3000/collections/docs/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      { "id": "vec1", "vector": [...], "metadata": {...} },
      { "id": "vec2", "vector": [...], "metadata": {...} }
    ]
  }'
```

**Get Vector by ID**

```bash
curl http://localhost:3000/collections/docs/vectors/vec1
```

**List Vectors (Pagination)**

```bash
curl "http://localhost:3000/collections/docs/vectors?offset=0&limit=10"
```

**Delete Vector by ID**

```bash
curl -X DELETE http://localhost:3000/collections/docs/vectors/vec1
```

**Search**

```bash
curl -X POST http://localhost:3000/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{ 
    "vector": [0.1, 0.2, 0.3, ...], 
    "k": 5,
    "filter": { "Exact": ["category", "AI"] }
  }'
```

**Delete Collection**

```bash
curl -X DELETE http://localhost:3000/collections/docs
```

---

## CLI Usage

SurgeDB comes with a powerful CLI for benchmarking and validation.

```bash
# Run the validation suite (Recall & Latency)
cargo run --release -- validate

# Benchmark with 10k vectors + SQ8 compression
cargo run --release -- bench -c 10000 -q sq8

# Test persistence and recovery
cargo run --release -- persist
```

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the detailed path to production.

* [x] SIMD Distance Kernels (NEON/AVX)
* [x] HNSW Algorithm
* [x] SQ8 & Binary Quantization
* [x] WAL & Snapshot Persistence
* [x] Mmap Storage Backend
* [x] Collections & Metadata Support
* [x] Metadata Filtering (Exact, And, Or, Not)
* [x] HTTP Server (Axum)
* [x] UniFFI Bindings (Python, Swift, Kotlin)
* [x] WASM / Browser Support (Edge)
* [ ] Zero-Config RAG Pipeline (Candle Integration)

## License

Distributed under the MIT License. See `LICENSE` for more information.
