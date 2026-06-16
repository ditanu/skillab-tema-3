"""
Repository Pattern - Data Access Layer
"""
from typing import Any

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from models import DocumentChunk, AchizitieDirecta, AnuntInitiere, EMBEDDING_DIM


class DocumentChunkRepository:
    """Repository pentru DocumentChunk (RAG)."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, chunk: DocumentChunk) -> DocumentChunk:
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def add_batch(self, chunks: list[DocumentChunk]) -> int:
        self.session.add_all(chunks)
        self.session.flush()
        return len(chunks)

    def get_by_id(self, chunk_id: int) -> DocumentChunk | None:
        return self.session.query(DocumentChunk).filter_by(id=chunk_id).first()

    def get_by_file(self, file_name: str) -> list[DocumentChunk]:
        return (
            self.session.query(DocumentChunk)
            .filter_by(file_name=file_name)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

    def search_similar(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Caută chunks similare folosind cosine similarity.

        Returns:
            Lista de (chunk, score) ordonate descrescător după scor.
        """
        embedding_str = f"[{','.join(map(str, embedding))}]"

        # Folosim format string pentru vector cast (evită conflict cu :param)
        query = text(f"""
            SELECT id, 1 - (embedding <=> '{embedding_str}'::vector) as score
            FROM document_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT :top_k
        """)

        results = self.session.execute(
            query,
            {"top_k": top_k}
        ).fetchall()

        chunks_with_scores = []
        for row in results:
            if row.score >= threshold:
                chunk = self.get_by_id(row.id)
                if chunk:
                    chunks_with_scores.append((chunk, row.score))

        return chunks_with_scores

    def count(self) -> int:
        return self.session.query(func.count(DocumentChunk.id)).scalar() or 0

    def count_by_file(self) -> dict[str, int]:
        results = (
            self.session.query(
                DocumentChunk.file_name,
                func.count(DocumentChunk.id)
            )
            .group_by(DocumentChunk.file_name)
            .all()
        )
        return {file_name: count for file_name, count in results}

    def delete_by_file(self, file_name: str) -> int:
        count = (
            self.session.query(DocumentChunk)
            .filter_by(file_name=file_name)
            .delete()
        )
        return count

    def delete_all(self) -> int:
        count = self.session.query(DocumentChunk).delete()
        return count


class AchizitieRepository:
    """Repository pentru AchizitieDirecta."""

    def __init__(self, session: Session):
        self.session = session

    def add_batch(self, records: list[dict], progress: bool = False) -> int:
        from tqdm import tqdm
        items = tqdm(records, desc="  achizitii") if progress else records
        for record in items:
            self.session.add(AchizitieDirecta(**record))
        self.session.flush()
        return len(records)

    def count(self) -> int:
        return self.session.query(func.count(AchizitieDirecta.id)).scalar() or 0

    def delete_all(self) -> int:
        return self.session.query(AchizitieDirecta).delete()


class AnuntRepository:
    """Repository pentru AnuntInitiere."""

    def __init__(self, session: Session):
        self.session = session

    def add_batch(self, records: list[dict], progress: bool = False) -> int:
        from tqdm import tqdm
        items = tqdm(records, desc="  anunturi") if progress else records
        for record in items:
            self.session.add(AnuntInitiere(**record))
        self.session.flush()
        return len(records)

    def count(self) -> int:
        return self.session.query(func.count(AnuntInitiere.id)).scalar() or 0

    def delete_all(self) -> int:
        return self.session.query(AnuntInitiere).delete()
