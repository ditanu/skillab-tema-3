"""
RAG Service - Embedding + Search
"""
import logging
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from models import DocumentChunk
from repositories import DocumentChunkRepository

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy loading pentru embedding model
_embedding_model: "SentenceTransformer | None" = None


def get_embedding_model() -> "SentenceTransformer":
    """Încarcă modelul de embedding (lazy)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


class RAGService:
    """Service pentru RAG operations."""

    def __init__(self, session: Session):
        self.session = session
        self.repo = DocumentChunkRepository(session)

    def embed_text(self, text: str) -> list[float]:
        """Generează embedding pentru text."""
        model = get_embedding_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def add_chunk(
        self,
        file_name: str,
        chunk_index: int,
        content: str,
        chunk_type: str = "paragraph",
        summary: str = "",
        metadata_json: str = "{}"
    ) -> DocumentChunk:
        """Adaugă un chunk cu embedding."""
        embedding = self.embed_text(content)

        chunk = DocumentChunk(
            file_name=file_name,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            content=content,
            summary=summary,
            metadata_json=metadata_json,
            embedding=embedding,
        )

        return self.repo.add(chunk)

    def add_chunks_batch(self, file_name: str, chunks: list[dict]) -> int:
        """Adaugă mai multe chunks dintr-un fișier."""
        model = get_embedding_model()

        # Batch encode
        texts = [c["content"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)

        db_chunks = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            db_chunks.append(DocumentChunk(
                file_name=file_name,
                chunk_index=i,
                chunk_type=chunk.get("chunk_type", "paragraph"),
                content=chunk["content"],
                summary=chunk.get("summary", ""),
                metadata_json=chunk.get("metadata_json", "{}"),
                embedding=emb.tolist(),
            ))

        return self.repo.add_batch(db_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> list[tuple[DocumentChunk, float]]:
        """Caută chunks similare."""
        query_embedding = self.embed_text(query)
        return self.repo.search_similar(query_embedding, top_k, threshold)

    def stats(self) -> dict:
        """Statistici despre chunks."""
        return {
            "total_chunks": self.repo.count(),
            "chunks_per_file": self.repo.count_by_file(),
        }
