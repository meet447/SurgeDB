# SurgeDB WASM

[![npm version](https://img.shields.io/npm/v/surgedb-wasm.svg)](https://www.npmjs.com/package/surgedb-wasm)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/meet447/SurgeDB/blob/main/LICENSE)
[![Size](https://img.shields.io/bundlephobia/minzip/surgedb-wasm)](https://bundlephobia.com/package/surgedb-wasm)

**SurgeDB WASM** brings high-performance, persistent vector search directly to the browser.

It is a lightweight **WebAssembly** binding for the SurgeDB engine, optimized for edge devices and client-side applications. Run semantic search, recommendation systems, and RAG pipelines entirely offline without sending user data to a server.

---

## Key Features

* 🔒 **Privacy First**: All data stays on the client. No API calls, no data leaks.
* ⚡ **Sub-millisecond Search**: Powered by SIMD-accelerated HNSW indices (Rust).
* 📦 **Tiny Footprint**: ~100KB gzipped (WASM + JS).
* 💾 **Memory Efficient**: Built-in SQ8 quantization reduces memory usage by 4x.
* 🛠️ **TypeScript**: Fully typed API for seamless development.

---

## Installation

```bash
npm install surgedb-wasm
```

---

## Quick Start

SurgeDB is designed to work seamlessly with modern bundlers (Vite, Webpack, etc.).

```javascript
import init, { SurgeDB } from 'surgedb-wasm';

async function main() {
    // 1. Initialize the WASM module
    await init();
    
    // 2. Create the database (384 dimensions for MiniLM)
    const db = new SurgeDB(384);
    
    // 3. Add documents
    db.insert("doc_1", new Float32Array([0.1, 0.2, 0.3, ...]), { 
        title: "Client-side AI", 
        category: "Web" 
    });

    db.insert("doc_2", new Float32Array([0.4, 0.5, 0.6, ...]), { 
        title: "Rust in Browser", 
        category: "Web" 
    });
    
    // 4. Search
    const query = new Float32Array([0.1, 0.2, 0.3, ...]);
    const results = db.search(query, 5); // Start with top 5
    
    console.log(results);
    // Output: [{ id: "doc_1", score: 1.0, metadata: {...} }, ...]

    // 5. Clean up memory when component unmounts
    db.free();
}

main();
```

---

## Advanced Usage

### Quantization (Save 4x Memory)

For larger datasets (10k+ vectors), use `SurgeDBQuantized` to enable SQ8 compression. This slightly reduces precision but drastically lowers memory usage.

```javascript
import init, { SurgeDBQuantized } from 'surgedb-wasm';

await init();
const db = new SurgeDBQuantized(384);

// Usage is identical to standard SurgeDB
db.insert("id", vector, metadata);
```

### React Example

```tsx
import { useEffect, useState } from 'react';
import init, { SurgeDB } from 'surgedb-wasm';

export function VectorSearch() {
  const [db, setDb] = useState<SurgeDB | null>(null);

  useEffect(() => {
    init().then(() => {
      const database = new SurgeDB(384);
      setDb(database);
    });

    return () => db?.free();
  }, []);

  const handleSearch = (queryVec: Float32Array) => {
    if (!db) return;
    const results = db.search(queryVec, 10);
    console.log(results);
  };

  return <div>Search Component</div>;
}
```

---

## Performance Benchmark

Benchmarks run on an M2 MacBook Air (Chrome):

| Operation | Metric |
| :--- | :--- |
| **Search Latency** | ~0.5ms (1k vectors) |
| **Insert Speed** | ~50ms (1k vectors) |
| **Memory (Quantized)** | ~400KB per 1k vectors |
| **Memory (Standard)** | ~1.5MB per 1k vectors |

---

## Development & Building

If you are contributing to the project or building from source:

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for npm (generates pkg/ directory)
wasm-pack build --target web --release
```
