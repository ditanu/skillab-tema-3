"""
Șterge toate datele din tabele.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database import transaction
from repositories import AchizitieRepository, AnuntRepository, DocumentChunkRepository

print("Clear tables")
print("=" * 40)

with transaction() as db:
    n1 = AchizitieRepository(db).delete_all()
    print(f"achizitii_directe: {n1} deleted")

    n2 = AnuntRepository(db).delete_all()
    print(f"anunturi_initiere: {n2} deleted")

    # n3 = DocumentChunkRepository(db).delete_all()
    # print(f"document_chunks: {n3} deleted")

print("\nDone")
