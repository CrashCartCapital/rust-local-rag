"""RAG Client interface to rust-local-rag.

Provides HTTP-based communication with the RAG server for evaluation.
HTTP mode is recommended for simplicity; MCP mode available for integration testing.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result from RAG."""

    chunk_id: str
    document: str
    page: int
    text: str
    score: float
    section: Optional[str] = None


@dataclass
class SearchResponse:
    """Response from RAG search including timing."""

    results: List[SearchResult]
    latency_ms: float


class RAGClient:
    """Abstracts communication with rust-local-rag.

    Recommendation: Use HTTP mode for evaluation (simpler, faster).
    MCP mode available for testing actual production path.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        mode: str = "http",
        timeout: int = 60,
    ):
        """Initialize RAG client.

        Args:
            endpoint: RAG server URL (for HTTP mode)
            mode: "http" (recommended) or "mcp"
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self._session = requests.Session()

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        """Execute search query against RAG.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            SearchResponse with results and latency
        """
        start = time.perf_counter()

        if self.mode == "mcp":
            results = self._search_via_mcp(query, top_k)
        else:
            results = self._search_via_http(query, top_k)

        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(results=results, latency_ms=latency_ms)

    def _search_via_http(self, query: str, top_k: int) -> List[SearchResult]:
        """Direct HTTP call - recommended for evaluation."""
        try:
            response = self._session.post(
                f"{self.endpoint}/search",
                json={"query": query, "top_k": top_k},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("results", []):
                results.append(
                    SearchResult(
                        chunk_id=r.get("chunk_id", ""),
                        document=r.get("document", r.get("document_name", "")),
                        page=r.get("page", r.get("page_number", 0)),
                        text=r.get("text", ""),
                        score=r.get("score", r.get("relevance_score", 0.0)),
                        section=r.get("section"),
                    )
                )
            return results

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to RAG server: {e}") from e

    def _search_via_mcp(self, query: str, top_k: int) -> List[SearchResult]:
        """MCP via JSON-RPC over HTTP.

        Calls the search_documents MCP tool via the rmcp HTTP transport.
        Endpoint should be the MCP HTTP endpoint (default: http://localhost:3046/mcp).
        """
        try:
            # MCP JSON-RPC request format
            request_id = int(time.time() * 1000)
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": "search_documents",
                    "arguments": {
                        "query": query,
                        "top_k": top_k,
                    },
                },
            }

            response = self._session.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Handle JSON-RPC response
            if "error" in data:
                raise ConnectionError(f"MCP error: {data['error']}")

            result = data.get("result", {})
            content = result.get("content", [])

            # Parse MCP tool response (text content with JSON)
            # NOTE: The MCP server currently returns Markdown-formatted text, not JSON.
            # This parser attempts to extract JSON but will likely get empty results.
            # For proper evaluation, use HTTP mode with a server that has /search endpoint.
            results = []
            for item in content:
                if item.get("type") == "text":
                    text_content = item.get("text", "")
                    try:
                        search_results = json.loads(text_content)
                        if isinstance(search_results, list):
                            for r in search_results:
                                results.append(
                                    SearchResult(
                                        chunk_id=r.get("chunk_id", ""),
                                        document=r.get("document", r.get("document_name", "")),
                                        page=r.get("page", r.get("page_number", 0)),
                                        text=r.get("text", ""),
                                        score=r.get("score", r.get("relevance_score", 0.0)),
                                        section=r.get("section"),
                                    )
                                )
                    except json.JSONDecodeError:
                        # MCP server returns Markdown text, not JSON - log for debugging
                        logger.warning(
                            f"MCP response is not JSON (likely Markdown). "
                            f"Content preview: {text_content[:200]}..."
                        )

            return results

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to MCP endpoint: {e}") from e

    def health_check(self) -> bool:
        """Check if RAG server is reachable."""
        try:
            if self.mode == "mcp":
                # For MCP mode, endpoint is the MCP URL directly
                # Health check via a simple MCP ping (list tools)
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                }
                response = self._session.post(self.endpoint, json=payload, timeout=5)
                return response.status_code == 200 and "result" in response.json()
            else:
                # For HTTP mode, try readiness endpoint
                # Parse base URL from endpoint (remove /search path if present)
                base_url = self.endpoint.rsplit("/", 1)[0] if "/search" in self.endpoint else self.endpoint
                response = self._session.get(f"{base_url}/readyz", timeout=5)
                return response.status_code == 200
        except (requests.exceptions.RequestException, ValueError):
            return False

    def get_stats(self) -> dict:
        """Get RAG system statistics."""
        try:
            if self.mode == "mcp":
                # Call get_stats MCP tool
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "get_stats",
                        "arguments": {},
                    },
                }
                response = self._session.post(self.endpoint, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    raise ConnectionError(f"MCP error: {data['error']}")
                # Parse stats from MCP response
                result = data.get("result", {})
                content = result.get("content", [])
                for item in content:
                    if item.get("type") == "text":
                        text_content = item.get("text", "{}")
                        try:
                            # Stats response may have prefix text, try to extract JSON
                            # e.g., "RAG System Stats:\n{...}"
                            if "{" in text_content:
                                json_start = text_content.index("{")
                                return json.loads(text_content[json_start:])
                            return json.loads(text_content)
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Could not parse stats JSON: {e}")
                return {}
            else:
                # HTTP mode
                base_url = self.endpoint.rsplit("/", 1)[0] if "/search" in self.endpoint else self.endpoint
                response = self._session.get(f"{base_url}/stats", timeout=10)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to get stats: {e}") from e


def normalize_doc_name(name: str) -> str:
    """Normalize document names for fuzzy matching.

    Strips extensions, lowercases, and removes extra whitespace.
    """
    return name.lower().replace(".pdf", "").strip()


def make_chunk_key(document: str, page: int) -> str:
    """Create a normalized key for chunk matching.

    Uses document + page for fuzzy matching (per spec Section 2.5).
    """
    return f"{normalize_doc_name(document)}::{page}"


def matches_gold_reference(
    retrieved: SearchResult, gold_doc: str, gold_page: int, tolerance: int = 1
) -> bool:
    """Check if retrieved chunk matches a gold reference.

    Uses page ±tolerance to handle chunks spanning page boundaries.

    Args:
        retrieved: Retrieved search result
        gold_doc: Expected document name
        gold_page: Expected page number
        tolerance: Page number tolerance (default ±1)

    Returns:
        True if document matches and page is within tolerance
    """
    doc_match = normalize_doc_name(retrieved.document) == normalize_doc_name(gold_doc)
    page_match = abs(retrieved.page - gold_page) <= tolerance
    return doc_match and page_match
