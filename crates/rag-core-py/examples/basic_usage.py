#!/usr/bin/env python3
"""Basic usage example using the mock backend.

This example demonstrates core ragcore functionality without
any external dependencies (no embedding service required).
"""

import tempfile
from ragcore import RagEngine, QuerySpec


def main():
    # Create a temporary directory for the index
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create engine with mock backend (deterministic embeddings based on text)
        engine = RagEngine.create_mock(tmp_dir, dimension=128)

        # Index some documents
        print("Indexing documents...")
        chunks1 = engine.upsert_document(
            "python_intro.txt",
            """Python is a high-level, general-purpose programming language.
            Its design philosophy emphasizes code readability with the use
            of significant indentation. Python is dynamically typed and
            garbage-collected. It supports multiple programming paradigms,
            including structured, object-oriented and functional programming."""
        )
        print(f"  python_intro.txt: {chunks1} chunks")

        chunks2 = engine.upsert_document(
            "rust_intro.txt",
            """Rust is a multi-paradigm, general-purpose programming language
            that emphasizes performance, type safety, and concurrency. It
            enforces memory safety without garbage collection. Rust has been
            called a systems programming language and is designed for low-level
            memory management with zero-cost abstractions."""
        )
        print(f"  rust_intro.txt: {chunks2} chunks")

        chunks3 = engine.upsert_document(
            "ml_basics.txt",
            """Machine learning is a subset of artificial intelligence that
            enables systems to learn and improve from experience. It focuses
            on developing algorithms that can access data and use it to learn
            for themselves. Neural networks are a key technique in modern
            machine learning approaches."""
        )
        print(f"  ml_basics.txt: {chunks3} chunks")

        # Check stats
        stats = engine.stats()
        print(f"\nEngine stats:")
        print(f"  Documents: {stats.document_count}")
        print(f"  Chunks: {stats.chunk_count}")
        print(f"  Model: {stats.embedding_model}")
        print(f"  Dimension: {stats.embedding_dimension}")

        # Basic search
        print("\n--- Basic Search ---")
        query = "programming language features"
        results = engine.search(query, top_k=3)
        print(f"Query: '{query}'")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.document} (chunk {r.chunk_index})")
            print(f"   Score: {r.score:.4f}")
            print(f"   Text: {r.text[:80]}...")

        # Search with QuerySpec for diversity
        print("\n--- Search with Diversity ---")
        spec = QuerySpec(
            top_k=3,
            diversity_factor=0.5,  # Higher diversity
        )
        results = engine.search(query, spec=spec)
        print(f"Query: '{query}' (diversity_factor=0.5)")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.document} (chunk {r.chunk_index})")
            print(f"   Score: {r.score:.4f}")

        # Detailed score breakdown
        print("\n--- Score Breakdown ---")
        results = engine.search("memory safety", top_k=1)
        if results:
            r = results[0]
            print(f"Document: {r.document}")
            print(f"Scores:")
            print(f"  Embedding: {r.scores.embedding}")
            print(f"  Lexical:   {r.scores.lexical}")
            print(f"  Initial:   {r.scores.initial}")
            print(f"  Reranker:  {r.scores.reranker}")
            print(f"  Total:     {r.scores.total}")

        # List documents
        print("\n--- Document List ---")
        docs = engine.list_documents()
        for doc in docs:
            print(f"  - {doc}")

        # Persistence
        print("\n--- Persistence ---")
        engine.save()
        print("Index saved to disk")

        # Create new engine and load
        engine2 = RagEngine.create_mock(tmp_dir, dimension=128)
        engine2.load()
        print(f"Index loaded: {engine2.stats().document_count} documents")

        # Remove a document
        print("\n--- Remove Document ---")
        removed = engine.remove_document("ml_basics.txt")
        print(f"Removed {removed} chunks from ml_basics.txt")
        print(f"Remaining documents: {engine.list_documents()}")


if __name__ == "__main__":
    main()
