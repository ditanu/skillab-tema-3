# skillab-py

## Multi-Agent NL2SQL

```
Question → ANALYST → NL2SQL agents → PostgreSQL
              │
              ├── make_plan      (LLM face plan)
              ├── execute_step   (query/filter/join)
              └── synthesize     (LLM răspuns final)
```

---

## Analyst Agent

| Nod | Ce face |
|-----|---------|
| `make_plan` | LLM → plan JSON (query_table, filter, join) |
| `execute_step` | Execută pas: NL2SQL sau tool |
| `synthesize` | LLM → răspuns final din rezultate |

**Flow:** `make_plan → [execute_step]* → synthesize`

---

## NL2SQL Agent

| Nod | Ce face |
|-----|---------|
| `get_context` | Load schema + business rules |
| `generate_sql` | LLM → SQL |
| `validate_sql` | Check syntax, table, no DROP |
| `execute_sql` | Run în PostgreSQL |
| `handle_error` | LLM retry cu feedback (max 2x) |

**Flow:** `get_context → generate_sql → validate_sql → execute_sql`
                                                    ↺ `handle_error`

---

## Traces

```
traces/2024-01-15_14-30/
  ├── analyst_001_make_plan_prompt.txt
  ├── nl2sql_achizitii_001_generate_sql_prompt.txt
  └── ...
```

---

## Quick Start

```python
analyst = AnalystAgent(
    tables_config={"achizitii": {...}, "anunturi": {...}},
    db_url="postgresql://...",
    trace_dir="traces",
)
result = analyst.chat("Top 5 furnizori?")
```

---

## Install

```bash
pip install -e ".[all-llm]"
```

## Env

```
OPENAI_API_KEY / ANTHROPIC_API_KEY
DATABASE_URL
```
