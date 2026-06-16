"""create achizitii_directe table

Revision ID: 002
Revises: 001
Create Date: 2024-01-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'achizitii_directe',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('castigator', sa.String(500)),
        sa.Column('castigator_cui', sa.String(50)),
        sa.Column('castigator_tara', sa.String(100)),
        sa.Column('castigator_localitate', sa.String(200)),
        sa.Column('castigator_adresa', sa.Text()),
        sa.Column('tip_procedura', sa.String(200)),
        sa.Column('autoritate_contractanta', sa.String(500)),
        sa.Column('autoritate_contractanta_cui', sa.String(50)),
        sa.Column('numar_anunt', sa.String(100)),
        sa.Column('data_anunt', sa.DateTime()),
        sa.Column('descriere', sa.Text()),
        sa.Column('tip_incheiere_contract', sa.String(200)),
        sa.Column('numar_contract', sa.String(100)),
        sa.Column('data_contract', sa.DateTime()),
        sa.Column('titlu_contract', sa.Text()),
        sa.Column('valoare', sa.Numeric(15, 2)),
        sa.Column('moneda', sa.String(10)),
        sa.Column('valoare_ron', sa.Numeric(15, 2)),
        sa.Column('valoare_eur', sa.Numeric(15, 2)),
        sa.Column('cpv_code_id', sa.String(50)),
        sa.Column('cpv_code', sa.String(200)),
    )

    # Indexuri utile
    op.create_index('ix_achizitii_castigator', 'achizitii_directe', ['castigator'])
    op.create_index('ix_achizitii_autoritate', 'achizitii_directe', ['autoritate_contractanta'])
    op.create_index('ix_achizitii_data', 'achizitii_directe', ['data_anunt'])


def downgrade() -> None:
    op.drop_table('achizitii_directe')
