"""
RAG utilities for embeddings and retrieval.

Usage:
    from skillab.rag import get_embeddings, similarity_search

    embeddings = get_embeddings()
    vector = embeddings.embed_query("search text")
"""
import os
from typing import Optional, List, Tuple

from dotenv import load_dotenv

load_dotenv()


def get_embeddings(model_name: Optional[str] = None):
    """
    Get embeddings model.

    Tries providers in order:
    1. OpenAI (if OPENAI_API_KEY set)
    2. Sentence Transformers (local, no API key needed)

    Args:
        model_name: Override default model

    Returns:
        LangChain Embeddings instance
    """
    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small"
        )

    # Fall back to local sentence-transformers
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name or "all-MiniLM-L6-v2"
    )


def embed_texts(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """
    Embed a list of texts.

    Args:
        texts: List of strings to embed
        model_name: Override default model

    Returns:
        List of embedding vectors
    """
    embeddings = get_embeddings(model_name)
    return embeddings.embed_documents(texts)


def embed_query(text: str, model_name: Optional[str] = None) -> List[float]:
    """
    Embed a single query text.

    Args:
        text: Query string to embed
        model_name: Override default model

    Returns:
        Embedding vector
    """
    embeddings = get_embeddings(model_name)
    return embeddings.embed_query(text)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score between 0 and 1
    """
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def similarity_search(
    query: str,
    documents: List[str],
    top_k: int = 5,
    model_name: Optional[str] = None
) -> List[Tuple[int, str, float]]:
    """
    Find most similar documents to a query.

    Args:
        query: Search query
        documents: List of document texts
        top_k: Number of results to return
        model_name: Override default embeddings model

    Returns:
        List of (index, document, score) tuples, sorted by similarity
    """
    if not documents:
        return []

    # Embed query and documents
    query_vec = embed_query(query, model_name)
    doc_vecs = embed_texts(documents, model_name)

    # Compute similarities
    similarities = [
        (i, doc, cosine_similarity(query_vec, doc_vec))
        for i, (doc, doc_vec) in enumerate(zip(documents, doc_vecs))
    ]

    # Sort by similarity (descending) and return top_k
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


# === pgvector integration ===

def create_vector_extension(engine):
    """Create pgvector extension if not exists."""
    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def vector_column(dimensions: int = 1536):
    """
    SQLAlchemy column type for pgvector.

    Usage:
        from sqlalchemy import Column
        from skillab.rag import vector_column

        class Document(Base):
            embedding = Column(vector_column(1536))
    """
    from pgvector.sqlalchemy import Vector
    return Vector(dimensions)
