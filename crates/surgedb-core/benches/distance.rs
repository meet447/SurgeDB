//! Benchmarks for distance calculations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use surgedb_core::distance::{cosine_distance, dot_product_distance, euclidean_distance};

fn generate_random_vector(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| rand::random::<f32>() * 2.0 - 1.0)
        .collect()
}

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for dim in [128, 256, 384, 512, 768, 1024, 1536].iter() {
        let a = generate_random_vector(*dim);
        let b = generate_random_vector(*dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bencher, _| {
            bencher.iter(|| cosine_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");

    for dim in [128, 256, 384, 512, 768, 1024, 1536].iter() {
        let a = generate_random_vector(*dim);
        let b = generate_random_vector(*dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bencher, _| {
            bencher.iter(|| euclidean_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_distance");

    for dim in [128, 256, 384, 512, 768, 1024, 1536].iter() {
        let a = generate_random_vector(*dim);
        let b = generate_random_vector(*dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bencher, _| {
            bencher.iter(|| dot_product_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_distance,
    bench_euclidean_distance,
    bench_dot_product
);
criterion_main!(benches);
