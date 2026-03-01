# ArogyaLink — Synera 2.0 RAG Clinical Co-Pilot

Physiological trajectory intelligence for Indian primary health centres. Backend + RAG only (no frontend, no firmware).

## Quick start

**Linux/Mac (Make):**
```bash
cp .env.example .env
make up
make migrate
make seed
make ollama-pull   # one-time: ~4.7GB
make ingest       # seed ChromaDB collections
make simulate     # run MQTT simulator (5 patients)
make health       # check API
```

**Windows (no Make):** See [WINDOWS.md](WINDOWS.md). Summary:
- Copy env: `run-windows.bat env` or `.\run-windows.ps1 env`
- Install: `run-windows.bat install`
- **Run unit tests (no Docker):** `run-windows.bat test` or `.\run-windows.ps1 test`
- **Docker:** Start Docker Desktop first, then from repo root: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d`

## Stack

- **Backend:** FastAPI, async SQLAlchemy 2, PostgreSQL 15 + TimescaleDB, Redis, MQTT (aiomqtt), WebSocket
- **RAG:** LangChain 0.2, ChromaDB, BGE-M3 embeddings, Ollama (llama3.1:8b) — all local, no external LLM APIs (DPDP compliant)
- **Synera engine:** 3-rule pipeline (artifact → exertion → trajectory), sliding window, derivatives, RAG brief in &lt;3s

## Makefile

| Target | Description |
|--------|-------------|
| `up` | Start stack (docker-compose + dev overrides) |
| `down` | Stop stack |
| `migrate` | Alembic upgrade head |
| `seed` | Insert mock patients + seed RAG if needed |
| `ingest` | Seed ChromaDB (protocols + cases) |
| `simulate` | MQTT simulator (5 patient profiles) |
| `demo` | Run judge demo scenarios |
| `test` | Backend pytest |
| `test-rag` | RAG pytest |
| `health` | GET /api/v1/health |
| `chroma-check` | GET RAG knowledge chunks |

## License

Proprietary — ArogyaLink · Techgium Season 9.
