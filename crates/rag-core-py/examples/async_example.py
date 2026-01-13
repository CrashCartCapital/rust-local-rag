#!/usr/bin/env python3
"""Async API example demonstrating concurrent operations.

This example shows how to use the async methods for non-blocking
document indexing and search operations.
"""

import asyncio
import tempfile
import time
from ragcore import RagEngine


async def index_document(engine: RagEngine, name: str, content: str) -> int:
    """Index a single document asynchronously."""
    start = time.perf_counter()
    chunks = await engine.aupsert_document(name, content)
    elapsed = time.perf_counter() - start
    print(f"  Indexed {name}: {chunks} chunks in {elapsed:.3f}s")
    return chunks


async def search_async(engine: RagEngine, query: str, label: str) -> None:
    """Perform an async search."""
    start = time.perf_counter()
    results = await engine.asearch(query, top_k=2)
    elapsed = time.perf_counter() - start
    print(f"\n[{label}] Query: '{query}' ({elapsed:.3f}s)")
    for r in results:
        print(f"  - {r.document}: {r.score:.4f}")


async def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = RagEngine.create_mock(tmp_dir, dimension=64)

        # Sample documents
        documents = {
            "doc1.txt": "Python is excellent for data science and machine learning.",
            "doc2.txt": "Rust provides memory safety without garbage collection.",
            "doc3.txt": "JavaScript powers interactive web applications.",
            "doc4.txt": "Go is designed for concurrent programming at scale.",
            "doc5.txt": "TypeScript adds static typing to JavaScript.",
        }

        # Sequential async indexing
        print("Sequential async indexing:")
        for name, content in documents.items():
            await index_document(engine, name, content)

        print(f"\nTotal: {engine.stats().document_count} documents indexed")

        # Concurrent async indexing (new documents)
        print("\n--- Concurrent Indexing ---")
        more_docs = {
            "doc6.txt": "Machine learning models require large datasets.",
            "doc7.txt": "Web servers handle HTTP requests and responses.",
            "doc8.txt": "Database systems store and retrieve structured data.",
        }

        start = time.perf_counter()
        tasks = [
            index_document(engine, name, content)
            for name, content in more_docs.items()
        ]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        print(f"Concurrent indexing completed in {elapsed:.3f}s total")

        # Concurrent async searches
        print("\n--- Concurrent Searches ---")
        queries = [
            ("programming languages", "search1"),
            ("web development", "search2"),
            ("memory management", "search3"),
        ]

        start = time.perf_counter()
        search_tasks = [
            search_async(engine, query, label)
            for query, label in queries
        ]
        await asyncio.gather(*search_tasks)
        elapsed = time.perf_counter() - start
        print(f"\nAll searches completed in {elapsed:.3f}s total")

        # Mixed operations
        print("\n--- Mixed Concurrent Operations ---")
        start = time.perf_counter()
        await asyncio.gather(
            engine.aupsert_document("doc9.txt", "Redis is an in-memory data store."),
            engine.asearch("data storage", top_k=1),
            engine.aupsert_document("doc10.txt", "Docker containers isolate applications."),
        )
        elapsed = time.perf_counter() - start
        print(f"Mixed operations completed in {elapsed:.3f}s")

        # Final stats
        stats = engine.stats()
        print(f"\nFinal stats: {stats.document_count} documents, {stats.chunk_count} chunks")


if __name__ == "__main__":
    asyncio.run(main())
