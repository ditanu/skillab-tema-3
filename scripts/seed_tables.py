"""
Seed tabele SQL din Excel.
Folosește raw SQL cu text() - funcționează cu Timestamp-uri pandas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datetime import datetime
from sqlalchemy import text
from tqdm import tqdm
from database import engine

DATA_DIR = Path(__file__).parent.parent / "data" / "documents-sql"


def parse_dt(val):
    if pd.isna(val):
        return None
    try:
        return datetime.strptime(str(val), "%d-%m-%Y %H:%M:%S")
    except:
        return None


print("Seed tabele SQL")
print("=" * 40)

# === ACHIZITII ===
print("\nachizitii_directe:")
df = pd.read_excel(DATA_DIR / "achizitiidirecte2018t2.xlsx")
df.columns = df.columns.str.lower()
df['data_anunt'] = df['data_anunt'].apply(parse_dt)
df['data_contract'] = df['data_contract'].apply(parse_dt)
df['castigator_cui'] = df['castigator_cui'].astype(str)
df['autoritate_contractanta_cui'] = df['autoritate_contractanta_cui'].astype(str)
df['cpv_code_id'] = df['cpv_code_id'].astype(str)
df = df.where(pd.notnull(df), None)

insert_achizitii = text("""
    INSERT INTO achizitii_directe (
        castigator, castigator_cui, castigator_tara, castigator_localitate,
        castigator_adresa, tip_procedura, autoritate_contractanta,
        autoritate_contractanta_cui, numar_anunt, data_anunt, descriere,
        tip_incheiere_contract, numar_contract, data_contract, titlu_contract,
        valoare, moneda, valoare_ron, valoare_eur, cpv_code_id, cpv_code
    ) VALUES (
        :castigator, :castigator_cui, :castigator_tara, :castigator_localitate,
        :castigator_adresa, :tip_procedura, :autoritate_contractanta,
        :autoritate_contractanta_cui, :numar_anunt, :data_anunt, :descriere,
        :tip_incheiere_contract, :numar_contract, :data_contract, :titlu_contract,
        :valoare, :moneda, :valoare_ron, :valoare_eur, :cpv_code_id, :cpv_code
    )
""")

records = df.to_dict('records')
with engine.connect() as conn:
    batch_size = 1000
    for i in tqdm(range(0, len(records), batch_size), desc="  achizitii"):
        batch = records[i:i+batch_size]
        conn.execute(insert_achizitii, batch)
        conn.commit()
print(f"✓ {len(df)} rânduri")


# === ANUNTURI ===
print("\nanunturi_initiere:")
df = pd.read_excel(DATA_DIR / "anunturiinitiere2018t2.xlsx")
df.columns = df.columns.str.lower()
df['data_publicare'] = df['data_publicare'].apply(parse_dt)
df['cui'] = df['cui'].astype(str)
df = df.where(pd.notnull(df), None)

insert_anunturi = text("""
    INSERT INTO anunturi_initiere (
        tip_anunt, numar_anunt_invitatie, data_publicare, denumire_ac,
        cui, judet, tip_contract, utilitati, tip_procedura, criteriu_atribuire,
        valoare_estimata, moneda, modalitate_desfasurare, trimis_ojeu,
        fonduri_comunitare, main_cpv_code, main_cpv_name
    ) VALUES (
        :tip_anunt, :numar_anunt_invitatie, :data_publicare, :denumire_ac,
        :cui, :judet, :tip_contract, :utilitati, :tip_procedura, :criteriu_atribuire,
        :valoare_estimata, :moneda, :modalitate_desfasurare, :trimis_ojeu,
        :fonduri_comunitare, :main_cpv_code, :main_cpv_name
    )
""")

records = df.to_dict('records')
with engine.connect() as conn:
    batch_size = 1000
    for i in tqdm(range(0, len(records), batch_size), desc="  anunturi"):
        batch = records[i:i+batch_size]
        conn.execute(insert_anunturi, batch)
        conn.commit()
print(f"✓ {len(df)} rânduri")

print("\nDone")
