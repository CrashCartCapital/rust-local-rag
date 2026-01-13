//! Benchmarks for rag-core search operations.
//!
//! Run with: `cargo bench -p rag-core`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rag_core::IndexSet;

/// Generate a deterministic embedding from an index.
fn make_embedding(dim: usize, seed: usize) -> Vec<f32> {
    let mut emb = vec![0.0f32; dim];
    for (i, v) in emb.iter_mut().enumerate() {
        let h = (seed.wrapping_mul(31).wrapping_add(i)) as f32;
        *v = (h % 1000.0) / 1000.0 - 0.5;
    }
    // Normalize
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut emb {
            *v /= norm;
        }
    }
    emb
}

fn bench_search_candidates(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_candidates");

    for &chunk_count in &[100, 500, 1000, 5000] {
        let dim = 384;
        let mut index = IndexSet::with_dimension(dim);

        // Populate index
        for i in 0..chunk_count {
            let id = format!("chunk-{}", i);
            let text = format!("This is chunk number {} with some sample text content", i);
            let embedding = make_embedding(dim, i);
            index.add_chunk(&id, &text, &embedding).unwrap();
        }

        let query_embedding = make_embedding(dim, 999999);
        let query_text = "sample text content";

        group.bench_with_input(
            BenchmarkId::new("chunks", chunk_count),
            &chunk_count,
            |b, _| {
                b.iter(|| {
                    index.search_candidates(
                        black_box(&query_embedding),
                        black_box(query_text),
                        0.7,
                        0.3,
                        10,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_add_chunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_chunk");
    let dim = 384;

    group.bench_function("single_chunk", |b| {
        let mut index = IndexSet::with_dimension(dim);
        let mut counter = 0usize;

        b.iter(|| {
            let id = format!("chunk-{}", counter);
            let text = "Sample text for benchmarking chunk addition";
            let embedding = make_embedding(dim, counter);
            index.add_chunk(&id, text, &embedding).unwrap();
            counter += 1;
        })
    });

    group.finish();
}

fn bench_apply_batch(c: &mut Criterion) {
    use rag_core::ChunkUpdate;

    let mut group = c.benchmark_group("apply_batch");
    let dim = 384;

    for &batch_size in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &size| {
                let mut counter = 0usize;

                b.iter(|| {
                    let mut index = IndexSet::with_dimension(dim);
                    let updates: Vec<ChunkUpdate> = (0..size)
                        .map(|i| {
                            let idx = counter + i;
                            ChunkUpdate::Add {
                                id: format!("chunk-{}", idx),
                                text: format!("Batch chunk {} content", idx),
                                embedding: make_embedding(dim, idx),
                            }
                        })
                        .collect();
                    counter += size;
                    index.apply_batch(black_box(&updates)).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_search_candidates, bench_add_chunk, bench_apply_batch);
criterion_main!(benches);
