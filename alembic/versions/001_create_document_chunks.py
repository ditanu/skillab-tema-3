"""create document_chunks table

Revision ID: 001
Revises:
Create Date: 2024-01-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def upgrade() -> None:
    # Activează pgvector
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    op.create_table(
        'document_chunks',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('file_name', sa.String(255), nullable=False, index=True),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('chunk_type', sa.String(50)),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text()),
        sa.Column('metadata_json', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # Adaugă coloana vector separat (pgvector type)
    op.execute(f'ALTER TABLE document_chunks ADD COLUMN embedding vector({EMBEDDING_DIM})')


def downgrade() -> None:
    op.drop_table('document_chunks')
