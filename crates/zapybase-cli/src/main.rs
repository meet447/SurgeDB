//! ZappyBase CLI - Command-line interface for the vector database

use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Instant;
use zapybase_core::{
    Config, DistanceMetric, MmapConfig, MmapVectorDb, PersistentConfig, PersistentVectorDb,
    QuantizationType, QuantizedConfig, QuantizedVectorDb, VectorDb,
};

#[derive(Parser)]
#[command(name = "zapybase")]
#[command(author = "Meet Sonawane")]
#[command(version)]
#[command(about = "A high-performance, lightweight vector database", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a quick benchmark to test performance
    Bench {
        /// Number of vectors to insert
        #[arg(short, long, default_value = "10000")]
        count: usize,

        /// Vector dimensions
        #[arg(short, long, default_value = "384")]
        dimensions: usize,

        /// Quantization type
        #[arg(short, long, default_value = "none")]
        quantization: QuantizationArg,

        /// Use persistent storage (writes to disk)
        #[arg(short, long)]
        persistent: bool,

        /// Data directory for persistent storage
        #[arg(long, default_value = "./zapybase_data")]
        data_dir: PathBuf,
    },

    /// Compare quantization modes
    Compare {
        /// Number of vectors to insert
        #[arg(short, long, default_value = "5000")]
        count: usize,

        /// Vector dimensions
        #[arg(short, long, default_value = "384")]
        dimensions: usize,
    },

    /// Test persistence and recovery
    Persist {
        /// Data directory
        #[arg(short, long, default_value = "./zapybase_data")]
        data_dir: PathBuf,

        /// Number of vectors to insert
        #[arg(short, long, default_value = "1000")]
        count: usize,

        /// Vector dimensions
        #[arg(long, default_value = "128")]
        dimensions: usize,
    },

    /// Benchmark mmap storage (disk-resident vectors)
    Mmap {
        /// Data directory
        #[arg(short, long, default_value = "./zapybase_mmap")]
        data_dir: PathBuf,

        /// Number of vectors to insert
        #[arg(short, long, default_value = "10000")]
        count: usize,

        /// Vector dimensions
        #[arg(long, default_value = "384")]
        dimensions: usize,
    },

    /// Import vectors from a JSON file
    Import {
        /// Path to JSON file (format: [{"id": "...", "vector": [...]}, ...])
        #[arg(short, long)]
        file: PathBuf,

        /// Data directory for storage
        #[arg(short, long, default_value = "./zapybase_data")]
        data_dir: PathBuf,

        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,

        /// Quantization type
        #[arg(short, long, default_value = "none")]
        quantization: QuantizationArg,
    },

    /// Search the imported database
    Query {
        /// Data directory
        #[arg(short, long, default_value = "./zapybase_data")]
        data_dir: PathBuf,

        /// Vector dimensions
        #[arg(short, long)]
        dimensions: usize,

        /// Query vector as comma-separated floats
        #[arg(short, long)]
        vec: String,

        /// Top K
        #[arg(short, long, default_value = "5")]
        k: usize,
    },

    /// Validate accuracy (Recall) and performance across all modes
    Validate {
        /// Number of vectors to test
        #[arg(short, long, default_value = "2000")]
        count: usize,

        /// Vector dimensions
        #[arg(long, default_value = "128")]
        dimensions: usize,

        /// Top K for recall calculation
        #[arg(short, long, default_value = "10")]
        k: usize,
    },

    /// Heavy stress test with massive scale and concurrency
    Stress {
        /// Number of vectors to insert
        #[arg(short, long, default_value = "100000")]
        count: usize,

        /// Vector dimensions
        #[arg(short, long, default_value = "768")]
        dimensions: usize,

        /// Number of concurrent search threads
        #[arg(short, long, default_value = "8")]
        threads: usize,

        /// Data directory
        #[arg(long, default_value = "./zapybase_stress")]
        data_dir: PathBuf,
    },

    /// Show version and system information
    Info,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuantizationArg {
    None,
    Sq8,
    Binary,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench {
            count,
            dimensions,
            quantization,
            persistent,
            data_dir,
        } => {
            if persistent {
                run_persistent_benchmark(count, dimensions, &data_dir);
            } else {
                run_benchmark(count, dimensions, quantization);
            }
        }
        Commands::Compare { count, dimensions } => run_comparison(count, dimensions),
        Commands::Persist {
            data_dir,
            count,
            dimensions,
        } => run_persistence_test(&data_dir, count, dimensions),
        Commands::Mmap {
            data_dir,
            count,
            dimensions,
        } => run_mmap_benchmark(&data_dir, count, dimensions),
        Commands::Import {
            file,
            data_dir,
            dimensions,
            quantization,
        } => run_import(&file, &data_dir, dimensions, quantization),
        Commands::Query {
            data_dir,
            dimensions,
            vec,
            k,
        } => run_query(&data_dir, dimensions, &vec, k),
        Commands::Validate {
            count,
            dimensions,
            k,
        } => run_validation(count, dimensions, k),
        Commands::Stress {
            count,
            dimensions,
            threads,
            data_dir,
        } => run_stress_test(count, dimensions, threads, &data_dir),
        Commands::Info => show_info(),
    }
}

#[derive(Deserialize)]
struct ImportItem {
    id: String,
    vector: Vec<f32>,
}

fn run_import(
    file: &PathBuf,
    data_dir: &PathBuf,
    dimensions: usize,
    _quantization: QuantizationArg,
) {
    println!("ZappyBase Import");
    println!("===============");
    println!("File: {}", file.display());
    println!("Dimensions: {}", dimensions);
    println!();

    // Read JSON
    let file_content = std::fs::read_to_string(file).expect("Failed to read import file");
    let items: Vec<ImportItem> = serde_json::from_str(&file_content).expect("Failed to parse JSON");

    println!("Importing {} vectors...", items.len());

    let config = PersistentConfig {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let mut db = PersistentVectorDb::open(data_dir, config).expect("Failed to create database");

    let start = Instant::now();
    let mut skip_count = 0;
    for (i, item) in items.iter().enumerate() {
        match db.insert(item.id.clone(), &item.vector) {
            Ok(_) => {}
            Err(zapybase_core::Error::DuplicateId(_)) => {
                skip_count += 1;
            }
            Err(e) => panic!("Failed to insert: {:?}", e),
        }
        if (i + 1) % 100 == 0 {
            print!(
                "\r  Progress: {}/{} (skipped: {})",
                i + 1,
                items.len(),
                skip_count
            );
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }

    db.sync().unwrap();
    println!(
        "\r  Done! Imported {} vectors (skipped {}) in {:?}",
        items.len() - skip_count,
        skip_count,
        start.elapsed()
    );
    println!("Data stored in: {}", data_dir.display());
}

fn run_query(data_dir: &PathBuf, dimensions: usize, vec_str: &str, k: usize) {
    let query_vec: Vec<f32> = vec_str
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid float in query vector"))
        .collect();

    if query_vec.len() != dimensions {
        eprintln!(
            "Error: Query vector has {} dims, expected {}",
            query_vec.len(),
            dimensions
        );
        return;
    }

    let config = PersistentConfig {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let db = PersistentVectorDb::open(data_dir, config).expect("Failed to open database");

    println!("Searching for top {} neighbors...", k);
    let start = Instant::now();
    let results = db.search(&query_vec, k).expect("Search failed");
    let duration = start.elapsed();

    println!("\rFound {} results in {:?}", results.len(), duration);
    println!("{:<20} {:>10}", "ID", "Distance");
    println!("{}", "-".repeat(32));
    for (id, dist) in results {
        println!("{:<20} {:>10.4}", id, dist);
    }
}

fn run_benchmark(count: usize, dimensions: usize, quantization: QuantizationArg) {
    let quant_name = match quantization {
        QuantizationArg::None => "None (f32)",
        QuantizationArg::Sq8 => "SQ8 (u8)",
        QuantizationArg::Binary => "Binary (1-bit)",
    };

    println!("ZappyBase Benchmark");
    println!("==================");
    println!("Vectors: {}", count);
    println!("Dimensions: {}", dimensions);
    println!("Quantization: {}", quant_name);
    println!();

    // Generate random vectors
    println!("Generating {} random vectors...", count);
    let start = Instant::now();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| {
            (0..dimensions)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();
    println!("Generated in {:?}", start.elapsed());
    println!();

    match quantization {
        QuantizationArg::None => run_unquantized_bench(&vectors, dimensions),
        QuantizationArg::Sq8 => run_quantized_bench(&vectors, dimensions, QuantizationType::SQ8),
        QuantizationArg::Binary => {
            run_quantized_bench(&vectors, dimensions, QuantizationType::Binary)
        }
    }
}

fn run_unquantized_bench(vectors: &[Vec<f32>], dimensions: usize) {
    let count = vectors.len();

    let config = Config {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut db = VectorDb::new(config).expect("Failed to create database");

    // Insert vectors
    println!("Inserting vectors (with HNSW indexing)...");
    let start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        db.insert(format!("vec_{}", i), vector)
            .expect("Failed to insert");

        if (i + 1) % 1000 == 0 {
            print!("\r  Inserted: {}/{}", i + 1, count);
        }
    }
    let insert_time = start.elapsed();
    println!("\r  Inserted: {}/{}    ", count, count);
    println!("Insert time: {:?}", insert_time);
    println!(
        "Throughput: {:.0} vectors/sec",
        count as f64 / insert_time.as_secs_f64()
    );
    println!();

    // Search benchmark
    run_search_bench(&db, vectors, "HNSW");

    // Memory estimate
    let vector_bytes = count * dimensions * 4;
    let overhead_estimate = count * 200;
    let total_bytes = vector_bytes + overhead_estimate;
    println!("Memory estimate:");
    println!("  Vector data: {:.2} MB", vector_bytes as f64 / 1_000_000.0);
    println!("  Total (est): {:.2} MB", total_bytes as f64 / 1_000_000.0);
}

fn run_quantized_bench(vectors: &[Vec<f32>], dimensions: usize, quant_type: QuantizationType) {
    let count = vectors.len();

    let config = QuantizedConfig {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        quantization: quant_type,
        keep_originals: false,
        ..Default::default()
    };
    let mut db = QuantizedVectorDb::new(config).expect("Failed to create database");

    // Insert vectors
    println!("Inserting vectors (quantized storage)...");
    let start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        db.insert(format!("vec_{}", i), vector)
            .expect("Failed to insert");

        if (i + 1) % 1000 == 0 {
            print!("\r  Inserted: {}/{}", i + 1, count);
        }
    }
    let insert_time = start.elapsed();
    println!("\r  Inserted: {}/{}    ", count, count);
    println!("Insert time: {:?}", insert_time);
    println!(
        "Throughput: {:.0} vectors/sec",
        count as f64 / insert_time.as_secs_f64()
    );
    println!();

    // Search benchmark
    run_search_bench_quantized(&db, vectors, "Quantized Brute Force");

    // Memory stats
    let memory = db.memory_usage();
    let ratio = db.compression_ratio();
    let uncompressed = count * dimensions * 4;
    println!("Memory usage:");
    println!("  Quantized: {:.2} MB", memory as f64 / 1_000_000.0);
    println!(
        "  Uncompressed would be: {:.2} MB",
        uncompressed as f64 / 1_000_000.0
    );
    println!("  Compression ratio: {:.2}x", ratio);
}

fn run_persistent_benchmark(count: usize, dimensions: usize, data_dir: &PathBuf) {
    println!("ZappyBase Persistent Benchmark");
    println!("==============================");
    println!("Vectors: {}", count);
    println!("Dimensions: {}", dimensions);
    println!("Data dir: {}", data_dir.display());
    println!();

    // Clean up any existing data
    if data_dir.exists() {
        std::fs::remove_dir_all(data_dir).ok();
    }

    // Generate random vectors
    println!("Generating {} random vectors...", count);
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| {
            (0..dimensions)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();
    println!();

    let config = PersistentConfig {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        sync_writes: false,
        checkpoint_threshold: 16 * 1024 * 1024, // 16MB
        ..Default::default()
    };

    let mut db = PersistentVectorDb::open(data_dir, config).expect("Failed to create database");

    // Insert vectors
    println!("Inserting vectors (persistent + HNSW)...");
    let start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        db.insert(format!("vec_{}", i), vector)
            .expect("Failed to insert");

        if (i + 1) % 1000 == 0 {
            print!("\r  Inserted: {}/{}", i + 1, count);
        }
    }
    let insert_time = start.elapsed();
    println!("\r  Inserted: {}/{}    ", count, count);
    println!("Insert time: {:?}", insert_time);
    println!(
        "Throughput: {:.0} vectors/sec",
        count as f64 / insert_time.as_secs_f64()
    );
    println!();

    // Checkpoint
    println!("Creating checkpoint...");
    let start = Instant::now();
    db.checkpoint().expect("Checkpoint failed");
    println!("Checkpoint time: {:?}", start.elapsed());
    println!();

    // Search benchmark
    run_search_bench_persistent(&db, &vectors, "HNSW");

    // Show disk usage
    let disk_usage = dir_size(data_dir).unwrap_or(0);
    println!("Disk usage: {:.2} MB", disk_usage as f64 / 1_000_000.0);
}

fn run_mmap_benchmark(data_dir: &PathBuf, count: usize, dimensions: usize) {
    println!("ZappyBase Mmap Benchmark");
    println!("========================");
    println!("Vectors: {}", count);
    println!("Dimensions: {}", dimensions);
    println!("Data dir: {}", data_dir.display());
    println!();

    // Clean up any existing data
    if data_dir.exists() {
        std::fs::remove_dir_all(data_dir).ok();
    }

    // Generate random vectors
    println!("Generating {} random vectors...", count);
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| {
            (0..dimensions)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();
    println!();

    let config = MmapConfig {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let mut db = MmapVectorDb::open(data_dir, config).expect("Failed to create database");

    // Insert vectors
    println!("Inserting vectors (mmap + HNSW)...");
    let start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        db.insert(format!("vec_{}", i), vector)
            .expect("Failed to insert");

        if (i + 1) % 1000 == 0 {
            print!("\r  Inserted: {}/{}", i + 1, count);
        }
    }
    let insert_time = start.elapsed();
    println!("\r  Inserted: {}/{}    ", count, count);
    println!("Insert time: {:?}", insert_time);
    println!(
        "Throughput: {:.0} vectors/sec",
        count as f64 / insert_time.as_secs_f64()
    );
    println!();

    db.sync().expect("Sync failed");

    // Search benchmark
    run_search_bench_mmap(&db, &vectors, "Mmap HNSW");

    // Show disk usage
    let disk_usage = db.disk_usage();
    println!("Disk usage: {:.2} MB", disk_usage as f64 / 1_000_000.0);
}

fn run_persistence_test(data_dir: &PathBuf, count: usize, dimensions: usize) {
    println!("ZappyBase Persistence Test");
    println!("==========================");
    println!("Data dir: {}", data_dir.display());
    println!();

    // Clean up any existing data
    if data_dir.exists() {
        println!("Cleaning up existing data...");
        std::fs::remove_dir_all(data_dir).ok();
    }

    let config = PersistentConfig {
        dimensions,
        sync_writes: true,
        ..Default::default()
    };

    // Phase 1: Create and populate
    println!(
        "Phase 1: Creating database and inserting {} vectors...",
        count
    );
    {
        let mut db =
            PersistentVectorDb::open(data_dir, config.clone()).expect("Failed to create database");

        for i in 0..count {
            let vector: Vec<f32> = (0..dimensions).map(|j| ((i * j) as f32).sin()).collect();
            db.insert(format!("v{}", i), &vector)
                .expect("Failed to insert");
        }

        println!("  Inserted {} vectors", db.len());
        println!("  Creating checkpoint...");
        db.checkpoint().expect("Checkpoint failed");
        println!("  Done!");
    }
    println!();

    // Phase 2: Reopen and verify
    println!("Phase 2: Reopening database and verifying recovery...");
    {
        let db =
            PersistentVectorDb::open(data_dir, config.clone()).expect("Failed to open database");

        println!("  Recovered {} vectors", db.len());

        if db.len() == count {
            println!("  Recovery successful!");
        } else {
            println!("  ERROR: Expected {} vectors, got {}", count, db.len());
        }

        // Test search
        let query: Vec<f32> = (0..dimensions).map(|j| (j as f32).sin()).collect();
        let results = db.search(&query, 5).expect("Search failed");

        println!("  Search test: found {} results", results.len());
        for (id, dist) in results.iter().take(3) {
            println!("    {} (distance: {:.4})", id, dist);
        }
    }
    println!();

    // Phase 3: Add more data after recovery
    println!("Phase 3: Adding more data after recovery...");
    {
        let mut db = PersistentVectorDb::open(data_dir, config).expect("Failed to open database");

        let additional = 100;
        for i in count..(count + additional) {
            let vector: Vec<f32> = (0..dimensions).map(|j| ((i * j) as f32).cos()).collect();
            db.insert(format!("v{}", i), &vector)
                .expect("Failed to insert");
        }

        println!("  Added {} more vectors", additional);
        println!("  Total: {} vectors", db.len());
        db.sync().expect("Sync failed");
    }
    println!();

    // Show disk usage
    let disk_usage = dir_size(data_dir).unwrap_or(0);
    println!(
        "Final disk usage: {:.2} MB",
        disk_usage as f64 / 1_000_000.0
    );
    println!();
    println!("Persistence test complete!");
}

fn run_search_bench(db: &VectorDb, vectors: &[Vec<f32>], method: &str) {
    println!("Running search benchmark ({})...", method);
    let query_count = 100;
    let k = 10;

    let start = Instant::now();
    for i in 0..query_count {
        let query = &vectors[i % vectors.len()];
        let _ = db.search(query, k).expect("Search failed");
    }
    let search_time = start.elapsed();

    let avg_latency = search_time.as_micros() as f64 / query_count as f64;
    println!("  Queries: {}", query_count);
    println!("  Total time: {:?}", search_time);
    println!(
        "  Avg latency: {:.2} us ({:.2} ms)",
        avg_latency,
        avg_latency / 1000.0
    );
    println!(
        "  Throughput: {:.0} queries/sec",
        query_count as f64 / search_time.as_secs_f64()
    );
    println!();
}

fn run_search_bench_quantized(db: &QuantizedVectorDb, vectors: &[Vec<f32>], method: &str) {
    println!("Running search benchmark ({})...", method);
    let query_count = 100;
    let k = 10;

    let start = Instant::now();
    for i in 0..query_count {
        let query = &vectors[i % vectors.len()];
        let _ = db.search(query, k).expect("Search failed");
    }
    let search_time = start.elapsed();

    let avg_latency = search_time.as_micros() as f64 / query_count as f64;
    println!("  Queries: {}", query_count);
    println!("  Total time: {:?}", search_time);
    println!(
        "  Avg latency: {:.2} us ({:.2} ms)",
        avg_latency,
        avg_latency / 1000.0
    );
    println!(
        "  Throughput: {:.0} queries/sec",
        query_count as f64 / search_time.as_secs_f64()
    );
    println!();
}

fn run_search_bench_persistent(db: &PersistentVectorDb, vectors: &[Vec<f32>], method: &str) {
    println!("Running search benchmark ({})...", method);
    let query_count = 100;
    let k = 10;

    let start = Instant::now();
    for i in 0..query_count {
        let query = &vectors[i % vectors.len()];
        let _ = db.search(query, k).expect("Search failed");
    }
    let search_time = start.elapsed();

    let avg_latency = search_time.as_micros() as f64 / query_count as f64;
    println!("  Queries: {}", query_count);
    println!("  Total time: {:?}", search_time);
    println!(
        "  Avg latency: {:.2} us ({:.2} ms)",
        avg_latency,
        avg_latency / 1000.0
    );
    println!(
        "  Throughput: {:.0} queries/sec",
        query_count as f64 / search_time.as_secs_f64()
    );
    println!();
}

fn run_search_bench_mmap(db: &MmapVectorDb, vectors: &[Vec<f32>], method: &str) {
    println!("Running search benchmark ({})...", method);
    let query_count = 100;
    let k = 10;

    let start = Instant::now();
    for i in 0..query_count {
        let query = &vectors[i % vectors.len()];
        let _ = db.search(query, k).expect("Search failed");
    }
    let search_time = start.elapsed();

    let avg_latency = search_time.as_micros() as f64 / query_count as f64;
    println!("  Queries: {}", query_count);
    println!("  Total time: {:?}", search_time);
    println!(
        "  Avg latency: {:.2} us ({:.2} ms)",
        avg_latency,
        avg_latency / 1000.0
    );
    println!(
        "  Throughput: {:.0} queries/sec",
        query_count as f64 / search_time.as_secs_f64()
    );
    println!();
}

fn run_comparison(count: usize, dimensions: usize) {
    println!("ZappyBase Quantization Comparison");
    println!("==================================");
    println!("Vectors: {}", count);
    println!("Dimensions: {}", dimensions);
    println!();

    // Generate random vectors
    println!("Generating vectors...");
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| {
            (0..dimensions)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();
    println!();

    // Test each quantization mode
    let modes = [
        ("None (f32)", QuantizationType::None),
        ("SQ8 (u8)", QuantizationType::SQ8),
        ("Binary", QuantizationType::Binary),
    ];

    println!(
        "{:<15} {:>12} {:>12} {:>12} {:>10}",
        "Mode", "Insert (ms)", "Search (us)", "Memory (MB)", "Ratio"
    );
    println!("{}", "-".repeat(65));

    for (name, quant_type) in modes {
        let config = QuantizedConfig {
            dimensions,
            distance_metric: DistanceMetric::Cosine,
            quantization: quant_type,
            keep_originals: false,
            ..Default::default()
        };
        let mut db = QuantizedVectorDb::new(config).expect("Failed to create database");

        // Insert
        let start = Instant::now();
        for (i, vector) in vectors.iter().enumerate() {
            db.insert(format!("vec_{}", i), vector).unwrap();
        }
        let insert_time = start.elapsed().as_millis();

        // Search (average of 50 queries)
        let query_count = 50;
        let start = Instant::now();
        for i in 0..query_count {
            let _ = db.search(&vectors[i % vectors.len()], 10);
        }
        let search_time = start.elapsed().as_micros() / query_count as u128;

        // Memory
        let memory = db.memory_usage() as f64 / 1_000_000.0;
        let ratio = db.compression_ratio();

        println!(
            "{:<15} {:>12} {:>12} {:>12.2} {:>10.2}x",
            name, insert_time, search_time, memory, ratio
        );
    }

    println!();
    println!("Note: Binary quantization trades accuracy for 32x compression.");
    println!("      SQ8 is recommended for most use cases (4x compression, <5% recall loss)..");
}

fn show_info() {
    println!("ZappyBase v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("A high-performance, lightweight vector database");
    println!();
    println!("Features:");
    println!("  - SIMD-accelerated distance calculations");
    println!("  - HNSW indexing for fast approximate search");
    println!("  - SQ8 quantization (4x compression)");
    println!("  - Binary quantization (32x compression)");
    println!("  - ACID-compliant persistence (WAL + snapshots)");
    println!("  - Mmap-based disk-resident vector storage");
    println!("  - Cosine, Euclidean, and Dot Product metrics");
    println!();

    #[cfg(target_arch = "aarch64")]
    println!("Platform: ARM64 (Apple Silicon) with NEON SIMD");

    #[cfg(target_arch = "x86_64")]
    {
        println!("Platform: x86_64");
        if is_x86_feature_detected!("avx2") {
            println!("  AVX2: Supported");
        }
        if is_x86_feature_detected!("avx512f") {
            println!("  AVX-512: Supported");
        }
    }

    println!();
    println!("Commands:");
    println!("  zapybase bench                     Run in-memory benchmark");
    println!("  zapybase bench -q sq8              Benchmark with SQ8 quantization");
    println!("  zapybase bench -p                  Benchmark with persistence");
    println!("  zapybase compare                   Compare quantization modes");
    println!("  zapybase persist                   Test persistence & recovery");
    println!("  zapybase mmap                      Benchmark mmap storage");
    println!("  zapybase validate                  Check Recall & Quality");
    println!("  zapybase import                    Import vectors from JSON");
    println!("  zapybase query                     Search imported database");
    println!("  zapybase stress                    Heavy Stress Test (100k+ vectors)");
}

fn run_validation(count: usize, dimensions: usize, k: usize) {
    println!("ZappyBase Validation Suite");
    println!("==========================");
    println!("Testing accuracy and performance across all indexing modes.");
    println!(
        "Vectors: {}, Dimensions: {}, Top K: {}",
        count, dimensions, k
    );
    println!();

    // 1. Generate Data
    println!("Generating random vectors...");
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| {
            (0..dimensions)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();

    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| {
            (0..dimensions)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();

    // 2. Compute Ground Truth (Exact Search)
    println!("Computing Ground Truth (Exact Brute Force)...");
    let mut ground_truth = Vec::new();
    let start = Instant::now();
    for query in &queries {
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, DistanceMetric::Cosine.distance(query, v)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let top_ids: Vec<usize> = distances.iter().take(k).map(|(i, _)| *i).collect();
        ground_truth.push(top_ids);
    }
    println!("Ground Truth computed in {:?}", start.elapsed());
    println!();

    println!(
        "{:<20} {:>10} {:>12} {:>10}",
        "Mode", "Recall@K", "Latency (ms)", "Memory"
    );
    println!("{}", "-".repeat(55));

    // 3. Test In-Memory HNSW (The Gold Standard)
    {
        let config = Config {
            dimensions,
            ..Default::default()
        };
        let mut db = VectorDb::new(config).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            db.insert(format!("{}", i), v).unwrap();
        }

        let (recall, latency) = measure_db_performance(&db, &queries, &ground_truth, k);
        println!(
            "{:<20} {:>10.2}% {:>12.2} {:>10}",
            "HNSW (In-Mem)",
            recall * 100.0,
            latency,
            "N/A"
        );
    }

    // 4. Test SQ8 Quantization
    {
        let config = QuantizedConfig {
            dimensions,
            quantization: QuantizationType::SQ8,
            keep_originals: false,
            ..Default::default()
        };
        let mut db = QuantizedVectorDb::new(config).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            db.insert(format!("{}", i), v).unwrap();
        }

        let (recall, latency) = measure_quantized_db_performance(&db, &queries, &ground_truth, k);
        println!(
            "{:<20} {:>10.2}% {:>12.2} {:>10.2}x",
            "SQ8 (Quantized)",
            recall * 100.0,
            latency,
            db.compression_ratio()
        );
    }

    // 5. Test Binary Quantization
    {
        let config = QuantizedConfig {
            dimensions,
            quantization: QuantizationType::Binary,
            keep_originals: false,
            ..Default::default()
        };
        let mut db = QuantizedVectorDb::new(config).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            db.insert(format!("{}", i), v).unwrap();
        }

        let (recall, latency) = measure_quantized_db_performance(&db, &queries, &ground_truth, k);
        println!(
            "{:<20} {:>10.2}% {:>12.2} {:>10.2}x",
            "Binary (1-bit)",
            recall * 100.0,
            latency,
            db.compression_ratio()
        );
    }

    println!();
    println!("Note: Recall@K compares the top results against an exact search.");
    println!("      Higher is better (100% is perfect match).");
}

fn run_stress_test(count: usize, dimensions: usize, threads: usize, data_dir: &PathBuf) {
    println!("ZappyBase Industrial Stress Test");
    println!("===============================");
    println!("Scale: {} vectors", count);
    println!("Dimensions: {}", dimensions);
    println!("Concurrency: {} search threads", threads);
    println!("Data Dir: {}", data_dir.display());
    println!();

    if data_dir.exists() {
        std::fs::remove_dir_all(data_dir).ok();
    }

    // 1. Ingestion Stress
    println!("Phase 1: High-Speed Ingestion...");
    let config = PersistentConfig {
        dimensions,
        ..Default::default()
    };
    let mut db = PersistentVectorDb::open(data_dir, config).unwrap();

    let start = Instant::now();
    for i in 0..count {
        let vec: Vec<f32> = (0..dimensions).map(|_| rand::random::<f32>()).collect();
        db.insert(format!("v{}", i), &vec).unwrap();
        if (i + 1) % 5000 == 0 {
            print!("\r  Ingested: {}/{}", i + 1, count);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    let ingest_time = start.elapsed();
    println!("\r  Ingested: {} vectors in {:?}", count, ingest_time);
    println!(
        "  Throughput: {:.0} vectors/sec",
        count as f64 / ingest_time.as_secs_f64()
    );
    println!();

    // 2. Memory Footprint
    let disk_size = dir_size(data_dir).unwrap_or(0);
    println!("Phase 2: Resource Usage");
    println!("  Disk Usage: {:.2} MB", disk_size as f64 / 1_000_000.0);
    println!();

    // 3. Concurrency Stress
    println!(
        "Phase 3: Parallel Search Stress (Simulating {} users)...",
        threads
    );
    let query_count = 1000;
    let queries: Vec<Vec<f32>> = (0..query_count)
        .map(|_| (0..dimensions).map(|_| rand::random::<f32>()).collect())
        .collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    let start = Instant::now();
    let latencies: Vec<f64> = pool.install(|| {
        queries
            .par_iter()
            .map(|q| {
                let q_start = Instant::now();
                db.search(q, 10).unwrap();
                q_start.elapsed().as_secs_f64() * 1000.0
            })
            .collect()
    });
    let total_time = start.elapsed();

    // Calculate Percentiles
    let mut latencies = latencies;
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("  Total Queries: {}", query_count);
    println!("  Total Time: {:?}", total_time);
    println!(
        "  Throughput: {:.0} queries/sec",
        query_count as f64 / total_time.as_secs_f64()
    );
    println!("  Latency Percentiles:");
    println!("    p50: {:.2} ms", p50);
    println!("    p95: {:.2} ms", p95);
    println!("    p99: {:.2} ms", p99);
    println!();

    // 4. Recovery Stress
    println!("Phase 4: Cold Start Recovery...");
    drop(db); // Close DB
    let start = Instant::now();
    let db = PersistentVectorDb::open(
        data_dir,
        PersistentConfig {
            dimensions,
            ..Default::default()
        },
    )
    .unwrap();
    println!("  Recovered {} vectors in {:?}", db.len(), start.elapsed());
    println!();

    println!("Stress test complete!");
}

fn measure_db_performance(
    db: &VectorDb,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    k: usize,
) -> (f32, f64) {
    let start = Instant::now();
    let mut total_hits = 0;

    for (i, query) in queries.iter().enumerate() {
        let results = db.search(query, k).unwrap();
        let result_ids: std::collections::HashSet<String> =
            results.into_iter().map(|(id, _)| id.to_string()).collect();

        for &id_idx in &truth[i] {
            if result_ids.contains(&format!("{}", id_idx)) {
                total_hits += 1;
            }
        }
    }

    let avg_recall = total_hits as f32 / (queries.len() * k) as f32;
    let avg_latency = start.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;

    (avg_recall, avg_latency)
}

fn measure_quantized_db_performance(
    db: &QuantizedVectorDb,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    k: usize,
) -> (f32, f64) {
    let start = Instant::now();
    let mut total_hits = 0;

    for (i, query) in queries.iter().enumerate() {
        let results = db.search(query, k).unwrap();
        let result_ids: std::collections::HashSet<String> =
            results.into_iter().map(|(id, _)| id.to_string()).collect();

        for &id_idx in &truth[i] {
            if result_ids.contains(&format!("{}", id_idx)) {
                total_hits += 1;
            }
        }
    }

    let avg_recall = total_hits as f32 / (queries.len() * k) as f32;
    let avg_latency = start.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;

    (avg_recall, avg_latency)
}

/// Calculate total size of a directory
fn dir_size(path: &PathBuf) -> std::io::Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}
