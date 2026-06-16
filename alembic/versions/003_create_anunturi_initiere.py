"""create anunturi_initiere table

Revision ID: 003
Revises: 002
Create Date: 2024-01-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'anunturi_initiere',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('tip_anunt', sa.String(200)),
        sa.Column('numar_anunt_invitatie', sa.String(100)),
        sa.Column('data_publicare', sa.DateTime()),
        sa.Column('denumire_ac', sa.String(500)),
        sa.Column('cui', sa.String(50)),
        sa.Column('judet', sa.String(100)),
        sa.Column('tip_contract', sa.String(200)),
        sa.Column('utilitati', sa.String(100)),
        sa.Column('tip_procedura', sa.String(200)),
        sa.Column('criteriu_atribuire', sa.String(200)),
        sa.Column('valoare_estimata', sa.Numeric(15, 2)),
        sa.Column('moneda', sa.String(10)),
        sa.Column('modalitate_desfasurare', sa.String(200)),
        sa.Column('trimis_ojeu', sa.String(50)),
        sa.Column('fonduri_comunitare', sa.String(50)),
        sa.Column('main_cpv_code', sa.String(50)),
        sa.Column('main_cpv_name', sa.String(500)),
    )

    op.create_index('ix_anunturi_denumire', 'anunturi_initiere', ['denumire_ac'])
    op.create_index('ix_anunturi_data', 'anunturi_initiere', ['data_publicare'])
    op.create_index('ix_anunturi_judet', 'anunturi_initiere', ['judet'])


def downgrade() -> None:
    op.drop_table('anunturi_initiere')
