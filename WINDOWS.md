# Running Synera 2.0 on Windows (No Docker)

## Prerequisites

- **Python 3.11+** — [Download](https://www.python.org/downloads/) — check "Add Python to PATH"
- **Supabase account** — [supabase.com](https://supabase.com)
- **Groq API key** (optional, for cloud LLM) — [console.groq.com](https://console.groq.com)

## First-time setup

```cmd
cd C:\Users\Asus\ArogyaLink-techgium
run-windows.bat env
```

Edit `.env` and set:

- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `DATABASE_URL` (from Supabase project settings)
- `GROQ_API_KEY` (if using `LLM_PROVIDER=groq`)
- `LLM_PROVIDER=groq` or `LLM_PROVIDER=ollama`

Then:

1. In Supabase Dashboard → SQL Editor, run the contents of **`supabase/schema.sql`**.
2. `run-windows.bat install`
3. `run-windows.bat seed`
4. `run-windows.bat ingest`
5. `run-windows.bat start` (keep terminal open)
6. In a second terminal: `run-windows.bat simulate`
7. `run-windows.bat health`

## Run and test (two commands)

```cmd
pip install -r requirements.txt
python run.py
```

Then in another terminal: `python scripts/data_gen/mock_simulator.py`

## Tests (no network)

```cmd
run-windows.bat test
```

See **SUPABASE_SETUP.md** for Supabase project and schema steps.
