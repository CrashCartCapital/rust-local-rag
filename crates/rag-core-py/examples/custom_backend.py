#!/usr/bin/env python3
"""Custom embedding backend example.

This example shows how to create a custom embedding backend
that could integrate with any embedding service (OpenAI, Ollama,
HuggingFace, etc.).

Note: This example uses a simple hash-based embedding for demonstration.
In production, replace with actual embedding API calls.
"""

import hashlib
import tempfile
from ragcore import RagEngine, QuerySpec


class SimpleHashEmbedder:
    """A simple hash-based embedder for demonstration.

    In production, replace this with actual embedding API calls
    (e.g., OpenAI, Ollama, HuggingFace Transformers).
    """

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    def model_id(self) -> str:
        """Unique identifier for this embedding model."""
        return f"simple-hash-{self._dimension}d"

    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Generate embedding from text using SHA-256 hash.

        This is a deterministic but NOT semantically meaningful embedding.
        Real embeddings should capture semantic similarity.
        """
        # Hash the text
        text_hash = hashlib.sha256(text.lower().encode()).digest()

        # Convert hash bytes to floats in [-1, 1]
        embedding = []
        for i in range(self._dimension):
            byte_idx = i % len(text_hash)
            # Convert byte to float in [-1, 1]
            value = (text_hash[byte_idx] / 127.5) - 1.0
            embedding.append(value)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding (optional optimization).

        If provided, ragcore will use this for bulk operations.
        """
        return [self.embed(text) for text in texts]


class SimpleReranker:
    """A simple keyword-based reranker for demonstration.

    In production, replace with LLM-based reranking (e.g., Cohere,
    cross-encoder models, or custom LLM prompting).
    """

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates based on keyword overlap.

        Args:
            query: The search query
            candidates: List of candidate chunks with metadata

        Returns:
            List of {chunk_id, relevance} dicts
        """
        query_words = set(query.lower().split())

        results = []
        for candidate in candidates:
            text = candidate["text"].lower()
            text_words = set(text.split())

            # Simple overlap score
            overlap = len(query_words & text_words)
            max_possible = len(query_words)
            relevance = overlap / max_possible if max_possible > 0 else 0.0

            results.append({
                "chunk_id": candidate["chunk_id"],
                "relevance": min(relevance, 1.0),
            })

        return results


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create embedder and reranker
        embedder = SimpleHashEmbedder(dimension=64)
        reranker = SimpleReranker()

        print(f"Using embedder: {embedder.model_id()}")
        print(f"Embedding dimension: {embedder.dimension()}")

        # Create engine with custom backends
        engine = RagEngine.create(
            tmp_dir,
            backend=embedder,
            reranker=reranker,
        )

        # Index documents
        print("\nIndexing documents...")
        docs = [
            ("ai_paper.txt", """
                Artificial intelligence and machine learning have revolutionized
                how we approach complex problems. Deep learning models, particularly
                neural networks, have shown remarkable performance in tasks like
                image recognition and natural language processing.
            """),
            ("systems_paper.txt", """
                Systems programming requires careful attention to memory management
                and performance. Languages like Rust provide compile-time guarantees
                about memory safety while maintaining the performance characteristics
                needed for systems-level work.
            """),
            ("web_paper.txt", """
                Modern web development involves a complex ecosystem of tools and
                frameworks. JavaScript and its typed variant TypeScript dominate
                the frontend, while various backend technologies compete for
                server-side implementation.
            """),
        ]

        for name, content in docs:
            chunks = engine.upsert_document(name, content)
            print(f"  {name}: {chunks} chunks")

        # Search without reranker influence (for comparison)
        print("\n--- Search (embedding-only weights) ---")
        spec = QuerySpec(
            top_k=3,
            reranker_weight=0.0,  # Ignore reranker
            initial_weight=1.0,   # Use only embedding scores
        )
        results = engine.search("machine learning neural networks", spec=spec)
        for r in results:
            print(f"  {r.document}: {r.score:.4f}")
            print(f"    Embedding: {r.scores.embedding}, Reranker: {r.scores.reranker}")

        # Search with reranker influence
        print("\n--- Search (with reranker) ---")
        spec = QuerySpec(
            top_k=3,
            reranker_weight=0.7,  # Weight reranker more
            initial_weight=0.3,
        )
        results = engine.search("machine learning neural networks", spec=spec)
        for r in results:
            print(f"  {r.document}: {r.score:.4f}")
            print(f"    Embedding: {r.scores.embedding}, Reranker: {r.scores.reranker}")

        # Demonstrate persistence
        print("\n--- Persistence ---")
        engine.save()
        print("Saved index")

        # Verify model consistency check
        engine2 = RagEngine.create(tmp_dir, backend=embedder, reranker=reranker)
        engine2.load()
        print(f"Loaded index: {engine2.stats().document_count} docs")
        print(f"Needs reindex: {engine2.needs_reindex()}")

        # What happens if we change the model?
        different_embedder = SimpleHashEmbedder(dimension=128)  # Different dimension!
        engine3 = RagEngine.create(tmp_dir, backend=different_embedder)
        engine3.load()
        print(f"\nWith different model ({different_embedder.model_id()}):")
        print(f"  Needs reindex: {engine3.needs_reindex()}")


if __name__ == "__main__":
    main()
