"""Vector index, BM25 baseline, document chunking, and retrieval.

Provides dense retrieval via ChromaDB + sentence-transformers (or a
configured endpoint), a BM25 diagnostic baseline, KB/example indexing
with policy-preserving chunking, and a unified Retriever facade.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding manager
# ---------------------------------------------------------------------------

_CHUNKING_VERSION = "1"


class EmbeddingManager:
    """Embed texts using a local sentence-transformers model or a remote endpoint.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the sentence-transformers model.
    endpoint : str | None
        If provided and non-empty, call this HTTP endpoint for embeddings
        instead of loading the local model.
    dimension : int
        Expected embedding dimension (used for validation).
    batch_size : int
        Batch size for encoding.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        endpoint: str | None = None,
        dimension: int = 384,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.endpoint = endpoint if endpoint else None
        self.dimension = dimension
        self.batch_size = batch_size
        self._local_model: Any = None

    # -- lazy loading -------------------------------------------------------

    def _ensure_local_model(self) -> Any:
        """Load the local model on first use."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embedding. "
                    "Install with: pip install 'rhoai-model-training-lab[backend]'"
                ) from exc
            logger.info("Loading local embedding model: %s", self.model_id)
            self._local_model = SentenceTransformer(self.model_id)
        return self._local_model

    # -- public API ---------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Returns a list of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        if self.endpoint:
            return self._embed_via_endpoint(texts)
        return self._embed_local(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = self.embed_texts([query])
        return results[0]

    @property
    def revision(self) -> str:
        """Return a short identifier for the embedding model revision."""
        return self.model_id

    # -- private helpers ----------------------------------------------------

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_local_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in embeddings]

    def _embed_via_endpoint(self, texts: list[str]) -> list[list[float]]:
        """Call a remote embedding endpoint (OpenAI-compatible)."""
        import httpx

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {"input": batch, "model": self.model_id}
            resp = httpx.post(
                f"{self.endpoint}/embeddings",
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            batch_embs = sorted(data["data"], key=lambda d: d["index"])
            all_embeddings.extend([item["embedding"] for item in batch_embs])
        return all_embeddings


# ---------------------------------------------------------------------------
# Vector index (ChromaDB)
# ---------------------------------------------------------------------------


class VectorIndex:
    """Thin wrapper around a persistent ChromaDB collection.

    Parameters
    ----------
    persist_directory : str | Path
        Directory for the ChromaDB persistent storage.
    collection_prefix : str
        Prefix for collection names (e.g. ``tau_kb``).
    """

    def __init__(
        self,
        persist_directory: str | Path = "data/indexes/chromadb",
        collection_prefix: str = "tau_kb",
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.collection_prefix = collection_prefix
        self._client: Any = None
        self._collections: dict[str, Any] = {}
        self._corpus_hash: str = ""

    # -- lazy client --------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "chromadb is required for the vector index. "
                    "Install with: pip install 'rhoai-model-training-lab[backend]'"
                ) from exc
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info("ChromaDB client created at %s", self.persist_directory)
        return self._client

    # -- public API ---------------------------------------------------------

    def create_collection(self, name: str) -> Any:
        """Create (or get) a ChromaDB collection.

        The actual collection name is ``{prefix}_{name}``.
        """
        client = self._ensure_client()
        full_name = f"{self.collection_prefix}_{name}"
        collection = client.get_or_create_collection(
            name=full_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collections[name] = collection
        logger.info("Collection ready: %s (count=%d)", full_name, collection.count())
        return collection

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents with pre-computed embeddings to a collection."""
        collection = self._get_collection(collection_name)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        batch_size = 500
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        self._corpus_hash = self._compute_corpus_hash(documents)
        logger.info(
            "Added %d documents to collection '%s'",
            len(documents),
            collection_name,
        )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search a collection using a query embedding.

        Returns a list of result dicts with keys: ``id``, ``document``,
        ``metadata``, ``distance``.
        """
        collection = self._get_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, max(collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    }
                )
        return hits

    def get_fingerprint(self, embedding_revision: str = "") -> dict[str, str]:
        """Return a fingerprint dict including corpus hash, embedding revision, and chunking version."""
        return {
            "corpus_hash": self._corpus_hash,
            "embedding_revision": embedding_revision,
            "chunking_version": _CHUNKING_VERSION,
        }

    def collection_count(self, name: str) -> int:
        """Return the document count for a named collection."""
        collection = self._get_collection(name)
        return collection.count()

    # -- helpers ------------------------------------------------------------

    def _get_collection(self, name: str) -> Any:
        if name not in self._collections:
            self.create_collection(name)
        return self._collections[name]

    @staticmethod
    def _compute_corpus_hash(documents: list[str]) -> str:
        h = hashlib.sha256()
        for doc in sorted(documents):
            h.update(doc.encode("utf-8"))
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# BM25 index (diagnostic baseline)
# ---------------------------------------------------------------------------


class BM25Index:
    """BM25 sparse retrieval index for diagnostic comparison.

    Uses ``rank_bm25`` for TF-IDF–style scoring.
    """

    def __init__(self) -> None:
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._index: Any = None

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the BM25 index."""
        self._documents.extend(documents)
        if metadatas:
            self._metadatas.extend(metadatas)
        else:
            self._metadatas.extend([{} for _ in documents])
        self._rebuild_index()
        logger.info("BM25 index rebuilt with %d documents", len(self._documents))

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search the BM25 index.

        Returns a list of result dicts with keys: ``document``, ``metadata``,
        ``score``, ``rank``.
        """
        if not self._documents or self._index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append(
                    {
                        "document": self._documents[idx],
                        "metadata": self._metadatas[idx],
                        "score": float(scores[idx]),
                        "rank": rank,
                    }
                )
        return results

    def _rebuild_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for BM25 retrieval. "
                "Install with: pip install 'rhoai-model-training-lab[backend]'"
            ) from exc
        tokenized = [self._tokenize(doc) for doc in self._documents]
        self._index = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------

_POLICY_EXCEPTION_PATTERN = re.compile(
    r"(exception|unless|however|notwithstanding|provided\s+that|except\s+(when|if|for))",
    re.IGNORECASE,
)
_LINK_PATTERN = re.compile(r"(https?://\S+|see\s+(?:document|section|policy)\s+\S+)", re.IGNORECASE)


@dataclass
class Chunk:
    """A single text chunk produced by the chunker."""

    text: str
    chunk_id: str
    source_doc_id: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    preserve_policy_exceptions: bool = True,
    preserve_document_links: bool = True,
) -> list[Chunk]:
    """Split text into token-approximated chunks with overlap.

    Uses whitespace tokenisation (word count) as a proxy for token count,
    which is fast and model-independent.  Policy exception clauses and
    document links are never split across chunk boundaries when the
    preserve flags are set.

    Parameters
    ----------
    text : str
        Source document text.
    doc_id : str
        Source document identifier.
    chunk_size : int
        Target chunk size in approximate tokens (words).
    chunk_overlap : int
        Overlap between consecutive chunks in approximate tokens.
    preserve_policy_exceptions : bool
        If True, extend chunks to avoid splitting exception clauses.
    preserve_document_links : bool
        If True, extend chunks to avoid splitting document-link sentences.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        if end < len(words):
            extended = False

            if preserve_policy_exceptions and _POLICY_EXCEPTION_PATTERN.search(
                " ".join(words[max(0, end - 20) : end])
            ):
                look_ahead = min(end + 40, len(words))
                for j in range(end, look_ahead):
                    if words[j].endswith((".",";")):
                        end = j + 1
                        extended = True
                        break
                if not extended:
                    end = min(end + 20, len(words))

            if preserve_document_links and _LINK_PATTERN.search(
                " ".join(words[max(0, end - 10) : min(end + 10, len(words))])
            ):
                look_ahead = min(end + 15, len(words))
                for j in range(end, look_ahead):
                    if words[j].endswith((".", ";", ")")):
                        end = j + 1
                        break

        chunk_text_str = " ".join(words[start:end])
        char_start = text.find(words[start]) if start < len(words) else len(text)
        char_end = (
            text.find(words[end - 1], char_start) + len(words[end - 1])
            if end <= len(words)
            else len(text)
        )

        chunks.append(
            Chunk(
                text=chunk_text_str,
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                source_doc_id=doc_id,
                start_offset=char_start,
                end_offset=char_end,
            )
        )

        chunk_idx += 1
        start = max(start + 1, end - chunk_overlap)

    return chunks


# ---------------------------------------------------------------------------
# KB indexer
# ---------------------------------------------------------------------------


class KBIndexer:
    """Index knowledge-base documents for dense retrieval.

    Reads a JSONL file of KB documents (each with ``doc_id`` and ``text``
    fields, plus optional ``links`` and ``metadata``), chunks them, embeds
    the chunks, and stores them in the vector index.

    Parameters
    ----------
    embedding_manager : EmbeddingManager
        Manager for computing embeddings.
    vector_index : VectorIndex
        ChromaDB-backed vector index.
    bm25_index : BM25Index | None
        Optional BM25 index for the diagnostic baseline.
    chunk_size : int
        Chunk size in approximate tokens.
    chunk_overlap : int
        Overlap between chunks.
    preserve_policy_exceptions : bool
        Whether to preserve policy exceptions during chunking.
    preserve_document_links : bool
        Whether to preserve document links during chunking.
    """

    COLLECTION_NAME = "kb_documents"

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_index: VectorIndex,
        bm25_index: BM25Index | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        preserve_policy_exceptions: bool = True,
        preserve_document_links: bool = True,
    ) -> None:
        self.embedding = embedding_manager
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_policy_exceptions = preserve_policy_exceptions
        self.preserve_document_links = preserve_document_links

    def index_kb_documents(self, documents_jsonl_path: str | Path) -> int:
        """Chunk, embed, and index all KB documents from a JSONL file.

        Each line must be a JSON object with at least ``doc_id`` and ``text``.

        Returns the total number of chunks indexed.
        """
        path = Path(documents_jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"KB documents file not found: {path}")

        self.vector_index.create_collection(self.COLLECTION_NAME)

        all_chunks: list[Chunk] = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", line_num)
                    continue

                doc_id = doc.get("doc_id", f"doc_{line_num}")
                text = doc.get("text", "")
                if not text:
                    continue

                chunks = chunk_text(
                    text=text,
                    doc_id=doc_id,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    preserve_policy_exceptions=self.preserve_policy_exceptions,
                    preserve_document_links=self.preserve_document_links,
                )
                for c in chunks:
                    c.metadata.update(
                        {
                            "source_doc_id": doc_id,
                            "links": json.dumps(doc.get("links", [])),
                        }
                    )
                all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced from %s", path)
            return 0

        texts = [c.text for c in all_chunks]
        logger.info("Embedding %d KB chunks …", len(texts))
        embeddings = self.embedding.embed_texts(texts)

        self.vector_index.add_documents(
            collection_name=self.COLLECTION_NAME,
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in all_chunks],
            ids=[c.chunk_id for c in all_chunks],
        )

        if self.bm25_index is not None:
            self.bm25_index.add_documents(
                documents=texts,
                metadatas=[c.metadata for c in all_chunks],
            )

        logger.info("KB indexing complete: %d chunks indexed", len(all_chunks))
        return len(all_chunks)


# ---------------------------------------------------------------------------
# Example indexer
# ---------------------------------------------------------------------------


class ExampleIndexer:
    """Index training examples for optional example retrieval.

    Reads a JSONL file where each line has ``sample_id`` and ``messages``.
    The user turn is used as the searchable text.

    Parameters
    ----------
    embedding_manager : EmbeddingManager
        Manager for computing embeddings.
    vector_index : VectorIndex
        ChromaDB-backed vector index.
    """

    COLLECTION_NAME = "training_examples"

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_index: VectorIndex,
    ) -> None:
        self.embedding = embedding_manager
        self.vector_index = vector_index

    def index_training_examples(self, examples_jsonl_path: str | Path) -> int:
        """Index training examples from a JSONL file.

        Each line must be a JSON object with ``sample_id`` and ``messages``.
        The first user message content is used as the indexable text.

        Returns the total number of examples indexed.
        """
        path = Path(examples_jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Training examples file not found: {path}")

        self.vector_index.create_collection(self.COLLECTION_NAME)

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        full_examples: list[str] = []

        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", line_num)
                    continue

                sample_id = example.get("sample_id", f"example_{line_num}")
                messages = example.get("messages", [])
                user_text = ""
                for msg in messages:
                    if msg.get("role") == "user" and msg.get("content"):
                        user_text = msg["content"]
                        break
                if not user_text:
                    continue

                texts.append(user_text)
                ids.append(sample_id)
                metadatas.append({"full_example": json.dumps(example)})
                full_examples.append(json.dumps(example))

        if not texts:
            logger.warning("No indexable examples found in %s", path)
            return 0

        logger.info("Embedding %d training examples …", len(texts))
        embeddings = self.embedding.embed_texts(texts)

        self.vector_index.add_documents(
            collection_name=self.COLLECTION_NAME,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info("Example indexing complete: %d examples indexed", len(texts))
        return len(texts)


# ---------------------------------------------------------------------------
# Unified retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Unified retrieval interface supporting dense and BM25 methods.

    Parameters
    ----------
    embedding_manager : EmbeddingManager
        For dense query embedding.
    vector_index : VectorIndex
        ChromaDB vector index.
    bm25_index : BM25Index | None
        Optional BM25 index.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_index: VectorIndex,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.embedding = embedding_manager
        self.vector_index = vector_index
        self.bm25_index = bm25_index

    def retrieve(
        self,
        query: str,
        collection: str = "kb_documents",
        k: int = 5,
        method: str = "dense",
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Parameters
        ----------
        query : str
            The search query.
        collection : str
            Collection name to search (``kb_documents`` or ``training_examples``).
        k : int
            Number of results to return.
        method : str
            Retrieval method: ``dense`` (vector similarity) or ``bm25``.

        Returns
        -------
        list[dict]
            Retrieved results, each with ``document``, ``score``/``distance``,
            and ``metadata`` keys.
        """
        if method == "bm25":
            if self.bm25_index is None:
                raise ValueError("BM25 index is not initialised")
            return self.bm25_index.search(query, k=k)

        if method == "dense":
            query_embedding = self.embedding.embed_query(query)
            results = self.vector_index.search(
                collection_name=collection,
                query_embedding=query_embedding,
                k=k,
            )
            for r in results:
                r["score"] = 1.0 - r.get("distance", 0.0)
            return results

        raise ValueError(f"Unsupported retrieval method: {method!r}. Use 'dense' or 'bm25'.")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_rag_components(config: dict[str, Any]) -> dict[str, Any]:
    """Build all RAG components from a rag config dict.

    Returns a dict with keys: ``embedding``, ``vector_index``, ``bm25_index``,
    ``kb_indexer``, ``example_indexer``, ``retriever``.
    """
    retrieval_cfg = config.get("retrieval", {})
    emb_cfg = retrieval_cfg.get("embedding", {})
    vs_cfg = retrieval_cfg.get("vector_store", {})
    kb_cfg = retrieval_cfg.get("kb_index", {})
    bm25_cfg = retrieval_cfg.get("bm25", {})

    embedding = EmbeddingManager(
        model_id=emb_cfg.get("model_id", "sentence-transformers/all-MiniLM-L6-v2"),
        endpoint=emb_cfg.get("endpoint"),
        dimension=emb_cfg.get("dimension", 384),
        batch_size=emb_cfg.get("batch_size", 32),
    )

    vector_index = VectorIndex(
        persist_directory=vs_cfg.get("persist_directory", "data/indexes/chromadb"),
        collection_prefix=vs_cfg.get("collection_prefix", "tau_kb"),
    )

    bm25_index: BM25Index | None = None
    if bm25_cfg.get("enabled", True):
        bm25_index = BM25Index()

    kb_indexer = KBIndexer(
        embedding_manager=embedding,
        vector_index=vector_index,
        bm25_index=bm25_index,
        chunk_size=kb_cfg.get("chunk_size", 512),
        chunk_overlap=kb_cfg.get("chunk_overlap", 64),
        preserve_policy_exceptions=kb_cfg.get("preserve_policy_exceptions", True),
        preserve_document_links=kb_cfg.get("preserve_document_links", True),
    )

    example_indexer = ExampleIndexer(
        embedding_manager=embedding,
        vector_index=vector_index,
    )

    retriever = Retriever(
        embedding_manager=embedding,
        vector_index=vector_index,
        bm25_index=bm25_index,
    )

    return {
        "embedding": embedding,
        "vector_index": vector_index,
        "bm25_index": bm25_index,
        "kb_indexer": kb_indexer,
        "example_indexer": example_indexer,
        "retriever": retriever,
    }


__all__ = [
    "BM25Index",
    "Chunk",
    "EmbeddingManager",
    "ExampleIndexer",
    "KBIndexer",
    "Retriever",
    "VectorIndex",
    "build_rag_components",
    "chunk_text",
]
