"""
Seed chunks în pgvector.

1. Parsează DOCX-uri
2. Chunk cu LLM
3. Embed + insert
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))

from dotenv import load_dotenv
load_dotenv()

from skillab import get_llm
from database import transaction
from rag_service import RAGService

DOCS_DIR = Path(__file__).parent.parent / "data" / "documents"

CHUNK_PROMPT = """Analizează documentul și împarte-l în chunks pentru search.

DOCUMENT:
{text}

Returnează JSON cu chunks:
```json
{{"chunks": [
  {{"chunk_type": "header|paragraph|table", "content": "textul exact", "summary": "rezumat scurt"}}
]}}
```

REGULI:
- Păstrează content-ul COMPLET
- Max 10 chunks per document"""


def extract_text(path: Path) -> str:
    """Extrage text din DOCX."""
    from docx import Document
    doc = Document(path)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    return "\n\n".join(parts)


def chunk_document(llm, text: str) -> list[dict]:
    """Chunk document cu LLM."""
    prompt = CHUNK_PROMPT.format(text=text[:4000])
    response = llm.generate_sync([{"role": "user", "content": prompt}])

    try:
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        json_str = match.group(1) if match else response
        data = json.loads(json_str)
        return data.get("chunks", [])
    except:
        return [{"chunk_type": "document", "content": text[:1500], "summary": "Document complet"}]


def main():
    print("Seed chunks în pgvector")
    print("=" * 40)

    files = list(DOCS_DIR.glob("*.docx"))
    if not files:
        print(f"Nu sunt documente în {DOCS_DIR}")
        print("Rulează: python scripts/generate_docs.py")
        return

    print(f"Găsite {len(files)} documente")

    llm = get_llm()
    total = 0

    with transaction() as db:
        rag = RAGService(db)

        # Verifică dacă există deja
        stats = rag.stats()
        if stats["total_chunks"] > 0:
            print(f"Există {stats['total_chunks']} chunks. Șterg...")
            from repositories import DocumentChunkRepository
            DocumentChunkRepository(db).delete_all()

        for path in files:
            print(f"\n{path.name}...")

            text = extract_text(path)
            if len(text) < 50:
                print("  (prea scurt, skip)")
                continue

            chunks = chunk_document(llm, text)
            print(f"  {len(chunks)} chunks")

            if chunks:
                rag.add_chunks_batch(path.name, chunks)
                total += len(chunks)

    print(f"\n✓ {total} chunks inserate")


if __name__ == "__main__":
    main()
