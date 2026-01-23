# Contributing to SurgeDB

First off, thank you for considering contributing to SurgeDB! It's people like you that make open source such a great community.

## Development Setup

1. **Rust**: You'll need the latest stable Rust toolchain.
2. **SIMD Support**: The core engine uses NEON/AVX. To run with full optimizations:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```

## Workflow

1. Fork the repo.
2. Create a new branch for your feature or bugfix.
3. Ensure all tests pass: `cargo test`
4. Run the validation suite to ensure no regression in recall:
   ```bash
   cargo run --release -- validate
   ```
5. Submit a Pull Request.

## Areas for Contribution
- **Persistence**: Improving the WAL compaction.
- **Quantization**: Adding Product Quantization (PQ).
- **Bindings**: Swift/Kotlin/Python glue code.
- **Documentation**: Examples and tutorials.
