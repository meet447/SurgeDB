# ⚡ ZappyBase

**ZappyBase** is a high-performance, super-fast, and ultra-lightweight vector database built from scratch in Rust. It is optimized for edge devices (like Mac M-series) and resource-constrained environments (like 1GB RAM cloud instances).

[![Rust CI](https://github.com/meetsonawane/zapybase/actions/workflows/rust.yml/badge.svg)](https://github.com/meetsonawane/zapybase/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Performance Snapshot (M2 Mac)

We validate every build for **Recall** (accuracy) and **Latency**.

| Mode | Recall @ 10 | Latency (Avg) | Compression |
| :--- | :--- | :--- | :--- |
| **HNSW (In-Memory)** | **99.4%** | 0.21 ms | 1x |
| **SQ8 (Quantized)** | **99.0%** | **0.15 ms** | **4x** |
| **Binary (1-bit)** | 28.4% | 0.23 ms | 32x |

---

## ✨ Features

- **Adaptive HNSW Indexing**: High-speed approximate nearest neighbor search.
- **SIMD Optimized**: Hand-tuned kernels for NEON (Apple Silicon) and AVX-512 (x86).
- **Plug-and-Play Quantization**: 
  - **SQ8**: 4x compression with <1% accuracy loss.
  - **Binary**: 32x compression for massive datasets.
- **ACID-Compliant Persistence**: Write-Ahead Log (WAL) and Snapshots for crash-safe data.
- **Mmap Support**: Disk-resident vectors for datasets larger than RAM.

---

## 📦 Installation

Add ZappyBase to your `Cargo.toml`:

```toml
[dependencies]
zapybase-core = { git = "https://github.com/meet447/zapybase" }
```

---

## 🛠️ Quick Start (Rust)

```rust
use zapybase_core::{PersistentVectorDb, PersistentConfig, DistanceMetric};

fn main() {
    // 1. Setup Persistent Database
    let config = PersistentConfig {
        dimensions: 384, // MiniLM size
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut db = PersistentVectorDb::open("./zapybase_data", config).unwrap();

    // 2. Insert Vector
    let vec = vec![0.1; 384];
    db.insert("doc_1", &vec).unwrap();

    // 3. Search
    let query = vec![0.1; 384];
    let results = db.search(&query, 5).unwrap();
    
    println!("Found match: {}", results[0].0);
}
```

---

## 🖥️ CLI Usage

ZappyBase comes with a powerful CLI for benchmarking and validation.

```bash
# Run the validation suite (Recall & Latency)
cargo run --release -- validate

# Benchmark with 10k vectors + SQ8 compression
cargo run --release -- bench -c 10000 -q sq8

# Test persistence and recovery
cargo run --release -- persist
```

---

## 🗺️ Roadmap

- [x] SIMD Distance Kernels (NEON/AVX)
- [x] HNSW Algorithm
- [x] SQ8 & Binary Quantization
- [x] WAL & Snapshot Persistence
- [x] Mmap Storage Backend
- [ ] UniFFI Bindings (Python, Swift, Kotlin)
- [ ] Zero-Config RAG Pipeline (Candle Integration)
- [ ] HTTP/gRPC Server Layer

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
