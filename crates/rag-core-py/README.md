# rag-core-py

Python bindings for [rag-core](../rag-core/), a batteries-included RAG engine.

## Installation

```bash
# From source (development)
cd crates/rag-core-py
pip install maturin
maturin develop

# Run tests
pip install pytest pytest-asyncio
pytest tests/
```

## Quick Start

```python
from ragcore import RagEngine, QuerySpec

# Create engine with custom embedding backend
class MyBackend:
    def model_id(self) -> str:
        return "my-model"

    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        # Your embedding logic here
        return [0.0] * 768

engine = RagEngine.open("./my-index", backend=MyBackend())

# Index documents
engine.upsert_document("doc.txt", "Hello world content here")
engine.save()

# Search
results = engine.search("hello", top_k=5)
for r in results:
    print(f"{r.document}: {r.score:.3f}")
```

## Async Support

```python
import asyncio
from ragcore import RagEngine

async def main():
    engine = RagEngine.open("./index", backend=MyBackend())
    results = await engine.asearch("query")
    print(results)

asyncio.run(main())
```

## License

MIT
