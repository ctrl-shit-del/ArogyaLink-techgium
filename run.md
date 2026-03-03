# Running Synera 2.0 — Complete Guide

> Quick reference for the most common path: **Windows + PowerShell + Supabase + Groq**.
> For Linux/Mac, swap PowerShell commands for their Bash equivalents (`export` instead of `$env:`, `/` paths).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [First-Time Setup (do once)](#2-first-time-setup-do-once)
3. [Step 1 — Start the Backend](#3-step-1--start-the-backend)
4. [Step 2 — Verify Health](#4-step-2--verify-health)
5. [Step 3 — Run the Mock Simulator](#5-step-3--run-the-mock-simulator)
6. [Step 4 — Run the Simulation Report](#6-step-4--run-the-simulation-report)
7. [Optional: Test RAG Manually](#7-optional-test-rag-manually)
8. [Optional: Run Unit Tests](#8-optional-run-unit-tests)
9. [Using run-windows.bat (shortcut)](#9-using-run-windowsbat-shortcut)
10. [Common Issues](#10-common-issues)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Add to PATH during install |
| Supabase account | free tier | [supabase.com](https://supabase.com) — see README §10 |
| Groq API key | free tier | [console.groq.com](https://console.groq.com) |
| Cohere API key | free tier | [dashboard.cohere.com](https://dashboard.cohere.com) |
| Git | any | For cloning only |

> **Network:** Supabase is blocked on many campus/institutional Wi-Fi networks. Use a **personal mobile hotspot** if you see SSL errors or connection timeouts.

---

## 2. First-Time Setup (do once)

Run these steps exactly once before the first `python run.py`.

### 2a — Clone and enter the repo

```powershell
git clone https://github.com/ctrl-shit-del/ArogyaLink-techgium.git
cd ArogyaLink-techgium
```

### 2b — Create the `.env` file

```powershell
# PowerShell
Copy-Item .env.example .env
notepad .env        # or any editor

# cmd
run-windows.bat env
notepad .env
```

Fill in **all six required keys**:

```env
SUPABASE_URL=https://<your-ref>.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...
DATABASE_URL=postgresql://postgres.<ref>:<password>@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres

GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
```

Leave everything else at defaults for a first run.

> **Where to find credentials:** Supabase dashboard → **Project Settings → API** (URL + anon key + service_role key). The DATABASE_URL is under **Settings → Database → Transaction pooler**.

### 2c — Install Python dependencies

```powershell
pip install -r requirements.txt
```

Takes 3–5 minutes (PyTorch + Stable-Baselines3 are large). Run once; no need to repeat unless requirements.txt changes.

### 2d — Run the Supabase schema

1. Open [supabase.com](https://supabase.com) → your project → **SQL Editor → New Query**
2. Copy the entire contents of `supabase/schema.sql`
3. Paste and click **Run**

This creates the `patients`, `vitals_history`, `alert_events`, `medical_knowledge`, and `clinical_cases` tables plus HNSW vector indexes.

### 2e — Validate the connection

```powershell
$env:PYTHONPATH = $PWD
python scripts/setup/init_supabase.py
```

Expected output:
```
✅ Supabase connected
✅ Table patients — exists
✅ Table vitals_history — exists
✅ Table alert_events — exists
✅ pgvector extension active
```

### 2f — Seed 5 mock patients

```powershell
$env:PYTHONPATH = $PWD
python scripts/setup/seed_data.py
```

Creates PT-0001 through PT-0005 in Supabase with full MedID profiles, baselines, and medication records.

### 2g — Index the knowledge base into pgvector

```powershell
$env:PYTHONPATH = $PWD
python -m rag.knowledge_base.processed.indexer --collection all
```

Embeds and indexes **22 medical protocol chunks** (MOHFW, WHO IMCI, Indian Pharmacopoeia 2022) and **60 synthetic clinical cases** into Supabase pgvector. Takes 60–120 seconds on first run (Cohere API calls).

> This only needs to be run once. Re-running is safe (deduplication by `chunk_id`).

---

## 3. Step 1 — Start the Backend

Open a **dedicated terminal** for the server. It must stay running.

```powershell
# PowerShell (recommended)
$env:PYTHONPATH  = $PWD
$env:PYTHONUTF8  = "1"      # prevents cp1252 encoding errors on Windows
python run.py
```

```cmd
:: cmd alternative
set PYTHONPATH=%CD%
set PYTHONUTF8=1
python run.py
```

```bash
# Linux/Mac
export PYTHONPATH=$PWD
python run.py
```

### Expected startup output

```
✅ Embedding provider: Cohere API (no local model needed)
✅ DRL triage agent loaded
✅ Synera engine started
✅ LLM provider: groq
✅ Event bus running
📡 Ready. Start simulator: python scripts/data_gen/mock_simulator.py
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

All **5 green checkmarks** must appear. If any fail, check your `.env` values and network connectivity.

> **Startup time:** ~15–30 seconds for the first start (Cohere warmup + DB pool init). Subsequent restarts are faster.

---

## 4. Step 2 — Verify Health

In a **second terminal** (server keeps running in the first):

```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/v1/health | ConvertTo-Json -Depth 3
```

```bash
# curl equivalent
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

Expected response:

```json
{
  "status": "ok",
  "database": "connected (Supabase)",
  "vector_store": "connected (Supabase pgvector)",
  "llm_provider": "groq",
  "llm_status": "api_key_present",
  "collections": {
    "medical_knowledge": 22,
    "clinical_cases": 60
  }
}
```

If `collections` shows `0` for either table, re-run step 2g.

---

## 5. Step 3 — Run the Mock Simulator

In the **second terminal** (or a third):

```powershell
$env:PYTHONPATH  = $PWD
$env:PYTHONUTF8  = "1"
python scripts/data_gen/mock_simulator.py
```

```bash
# Linux/Mac
export PYTHONPATH=$PWD
python scripts/data_gen/mock_simulator.py
```

### What happens

The simulator publishes vitals for **5 patients every 5 seconds**:

| Patient | Scenario | Expected event |
|---------|----------|----------------|
| PT-0001 Rajesh Kumar | Stable, normal HR | No alert |
| PT-0002 Priya Sharma | Post-op sepsis — exponential HR acceleration | **SYNERA_STATE fires at HR ≈ 92–103** |
| PT-0003 Arjun Mehta | Bathroom trip — HR spike + high motion | `EXERTION_LOGGED` |
| PT-0004 Fatima Begum | Sensor glitch — HR spike to 228 | `ARTIFACT` (silent discard) |
| PT-0005 Suresh Patel | Stable | No alert |

### Watching for the alert

In the **server terminal**, look for:

```
[ENGINE] Rule C FIRED for PT-0002  deviation_sigma=2.1  hr=96.3
[DRL] PT-0002 → priority=URGENT  confidence=0.81
[RAG ] PT-0002 — generating clinical brief...
[RAG ] PT-0002 — brief ready (3.2s, groq)
[ALERT] PT-0002 — SYNERA_STATE dispatched  alert_id=...
```

SYNERA_STATE fires for PT-0002 **approximately 2–3 minutes** into the simulator run (after the 15 stable readings + enough exponential readings to fill the 10-reading sliding window).

> **Lead time:** Synera alerts at HR ≈ 92–103 BPM, approximately **+135 seconds before** a static HR > 120 threshold would fire.

---

## 6. Step 4 — Run the Simulation Report

After at least one SYNERA_STATE alert has fired for PT-0002, run the quality report:

```powershell
$env:PYTHONUTF8 = "1"
python scripts/testing/simulation_report.py
```

Expected output:

```
================================================================
  Synera 2.0 — Simulation Quality Report
================================================================

[1/5] Health check …
      status        : ok
      ...

[4/5] PT-0002 (Priya Sharma) trajectory analysis …
      Synera alert index : XX  (HR≈92.5, ts=...)
      Lead-time advantage: +135s (trigger HR 92.5 → static 120, ~1 bpm/5s)
      PT-0002 alerts     : N

[5/5] False-positive audit (PT-0001/0003/0004/0005) …
      PT-0001: ✅ CLEAN
      PT-0003: ✅ CLEAN
      PT-0004: ✅ CLEAN
      PT-0005: ✅ CLEAN

────────────────────────────────────────────────────────────────
  SUMMARY
────────────────────────────────────────────────────────────────
  PT-0002 SYNERA_STATE fired : YES ✅
  Alert trigger HR           : 92.5 bpm
  Lead time vs static (>120): +135s
  False positives (4 stable): 0
  Backend health             : ok ✅
────────────────────────────────────────────────────────────────

  Report saved → docs\simulation_report.json
```

The JSON report is saved to `docs/simulation_report.json`.

---

## 7. Optional: Test RAG Manually

Trigger the RAG pipeline without running the simulator:

```powershell
$body = @{
    patient_id     = "PT-0002"
    trigger_vital  = "heart_rate"
    trigger_value  = 103
    motion_score   = 1
} | ConvertTo-Json

Invoke-RestMethod -Method POST `
    -Uri http://localhost:8000/api/v1/rag/trigger `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 10
```

```bash
# curl
curl -s -X POST http://localhost:8000/api/v1/rag/trigger \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"PT-0002","trigger_vital":"heart_rate","trigger_value":103,"motion_score":1}' \
  | python -m json.tool
```

Returns a full `ClinicalBrief` JSON with differential diagnoses, recommended actions, and Supabase chunk sources. Takes 1–5 seconds on Groq free tier.

---

## 8. Optional: Run Unit Tests

The 48 unit tests (Rule A × 16, Rule B × 16, Rule C × 16) require **no network** — no Supabase, Groq, or Cohere keys needed.

```powershell
$env:PYTHONPATH = $PWD
python -m pytest backend/tests/unit/ -v --tb=short
```

All 48 should pass. These cover edge cases for artifact rejection, exertion detection, and trajectory acceleration logic.

---

## 9. Using run-windows.bat (shortcut)

`run-windows.bat` wraps the common commands:

```cmd
run-windows.bat env        :: copy .env.example → .env
run-windows.bat install    :: pip install -r requirements.txt
run-windows.bat seed       :: python scripts/setup/seed_data.py
run-windows.bat ingest     :: index knowledge base into pgvector
run-windows.bat start      :: python run.py
run-windows.bat simulate   :: open simulator in new cmd window
run-windows.bat test       :: pytest unit tests
run-windows.bat health     :: curl /api/v1/health
```

PowerShell equivalent: `.\run-windows.ps1 <command>` with the same subcommands.

---

## 10. Common Issues

### `UnicodeEncodeError: 'cp1252' codec can't encode character`

**Cause:** Windows cp1252 console can't print emoji (✅) from `run.py`.

**Fix:**
```powershell
$env:PYTHONUTF8 = "1"
python run.py
```
Or add `PYTHONUTF8=1` to your `.env` file so it's always set.

---

### `ModuleNotFoundError: No module named 'backend'`

**Fix:** PYTHONPATH is not set. Run:
```powershell
$env:PYTHONPATH = $PWD
```
Run this in **every new terminal** before any `python ...` command.

---

### Supabase connection timeout / SSL error

**Fix:** You're on a network that blocks Supabase (Singapore endpoint). Switch to a **personal mobile hotspot** and retry.

---

### PT-0002 never fires SYNERA_STATE

**Symptoms:** Simulator runs for several minutes, HR climbs past 120, no `[ENGINE] Rule C FIRED` in server logs.

**Check:** Make sure you have the latest code — specifically `backend/core/trajectory/derivatives.py` and `backend/core/trajectory/calculator.py`. The acceleration detection requires the fallback logic in `calculator.py` for the exponential plateau case. If you pulled an older version, do a `git pull`.

---

### Simulator shows `[PT-0002] Error:` with no message

**Cause:** The RAG pipeline (Groq API call) takes longer than the simulator's 10-second HTTP timeout. The alert **did fire and was stored**; the simulator just didn't wait for the response.

**Fix:** Not a real error — confirm the alert was stored:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/alerts/?limit=5" | ConvertTo-Json -Depth 3
```

---

### DRL model not loaded (`⚠️ DRL agent not loaded`)

**Cause:** `models/drl/synera_triage_ppo.zip` is missing or corrupt.

**Fix:** Pre-train the agent:
```powershell
$env:PYTHONPATH = $PWD
python scripts/drl/run_pretrain.py
```
This takes 5–10 minutes and saves the model. Alerts still fire without the DRL agent — the engine falls back to a rule-based priority tier.

---

### Groq `AuthenticationError` or `RateLimitError`

**Fix:** Check `GROQ_API_KEY` in `.env`. Free tier limits: 14,400 requests/day, 30 requests/minute — more than enough for demo use.

---

## Quick Reference — Full Demo Sequence

```powershell
# Terminal 1 — Server (keep open)
$env:PYTHONPATH = $PWD ; $env:PYTHONUTF8 = "1" ; python run.py

# Terminal 2 — Wait for 5 green ticks, then verify
Invoke-RestMethod -Uri http://localhost:8000/api/v1/health | Select-Object status,database,llm_status

# Terminal 2 — Simulator (runs ~3 min until SYNERA_STATE fires for PT-0002)
$env:PYTHONPATH = $PWD ; $env:PYTHONUTF8 = "1" ; python scripts/data_gen/mock_simulator.py

# Terminal 2 — After alert fires, run quality report
$env:PYTHONUTF8 = "1" ; python scripts/testing/simulation_report.py
```

Swagger UI for interactive API exploration: **http://localhost:8000/docs**
