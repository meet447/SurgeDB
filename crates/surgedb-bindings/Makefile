# SurgeDB Python Bindings - Makefile
# This Makefile helps generate and install Python bindings

.PHONY: all build generate-python install-python test clean

# Build the Rust library in release mode
build:
	cargo build --release -p surgedb-bindings

# Generate Python bindings using UniFFI (macOS)
generate-python: build
	cargo run --release -p surgedb-bindings --bin uniffi-bindgen -- \
		generate \
		--library ../../target/release/libsurgedb_bindings.dylib \
		--language python \
		--out-dir python/
	cp ../../target/release/libsurgedb_bindings.dylib python/
	@echo "Python bindings generated in python/"
	@echo "Library copied to python/"

# For Linux, use .so instead of .dylib
generate-python-linux: build
	cargo run --release -p surgedb-bindings --bin uniffi-bindgen -- \
		generate \
		--library ../../target/release/libsurgedb_bindings.so \
		--language python \
		--out-dir python/
	cp ../../target/release/libsurgedb_bindings.so python/
	@echo "Python bindings generated in python/"
	@echo "Library copied to python/"

# Install Python bindings (development mode)
install-python: generate-python
	cd python && pip install -e .

# Run Python tests
test-python:
	cd python && python -m pytest tests/ -v

# Clean generated files
clean:
	rm -rf python/surgedb/
	rm -f python/*.so python/*.dylib

# Show help
help:
	@echo "SurgeDB Bindings Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  build            - Build Rust library in release mode"
	@echo "  generate-python  - Generate Python bindings (macOS)"
	@echo "  generate-python-linux - Generate Python bindings (Linux)"
	@echo "  install-python   - Install Python package in dev mode"
	@echo "  test-python      - Run Python tests"
	@echo "  clean            - Remove generated files"
