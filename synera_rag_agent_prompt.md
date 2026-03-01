# SYNERA 2.0 — RAG CLINICAL CO-PILOT BACKEND
## Agent Build Prompt · ArogyaLink · Techgium Season 9
### Backend Only · No Frontend · No Firmware

---

## MISSION

You are building the **RAG Clinical Co-Pilot** for Synera 2.0 — a physiological trajectory intelligence system for Indian primary health centres. This is **not a chatbot**. It is a clinical reasoning engine that fires when a patient's vitals hit trajectory acceleration at rest, queries a medical knowledge base + patient records simultaneously, and returns a structured JSON clinical brief in **under 3 seconds**.

You are building two modules from the ArogyaLink monorepo:
- **`backend/`** — FastAPI orchestration engine, Synera 3-rule pipeline, MQTT subscriber, WebSocket emitter
- **`rag/`** — LangChain + ChromaDB + Ollama pipeline, knowledge ingestion, prompt templates, parsers

**Privacy constraint (non-negotiable):** No patient data leaves the server. No external LLM API calls. Ollama runs locally. This is DPDP Act 2023 compliance by architecture.

---

## EXACT DIRECTORY STRUCTURE TO CREATE

Build precisely this layout. Do not add or rename folders.

```
ArogyaLink/
├── README.md
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env.example
├── Makefile
├── .gitignore
│
├── backend/
│   ├── main.py                          ← FastAPI app entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── patients.py              ← CRUD for MedID records
│   │   │   ├── alerts.py                ← Alert history + acknowledge
│   │   │   ├── vitals.py                ← Vitals history (paginated)
│   │   │   ├── medid.py                 ← MedID search + update
│   │   │   └── auth.py                  ← Basic API key auth (stub for now)
│   │   ├── middleware/
│   │   │   ├── auth.py                  ← API key validation
│   │   │   ├── rate_limit.py            ← Per-route rate limiting
│   │   │   └── logging.py               ← Request/response logging
│   │   └── websocket/
│   │       ├── manager.py               ← ConnectionManager (broadcast to all clients)
│   │       ├── events.py                ← WebSocket event schema definitions
│   │       └── handlers.py              ← WS route handlers
│   │
│   ├── core/                            ← THE SYNERA ENGINE
│   │   ├── synera_engine/
│   │   │   ├── engine.py                ← Main orchestrator (MQTT → rules → RAG → WS)
│   │   │   ├── pipeline.py              ← 3-rule sequential pipeline runner
│   │   │   └── state_manager.py         ← STABLE / WATCH / SYNERA_STATE per patient
│   │   ├── trajectory/
│   │   │   ├── calculator.py            ← Calls derivatives.py, returns trajectory verdict
│   │   │   ├── derivatives.py           ← Central difference 1st + 2nd derivative (NumPy)
│   │   │   └── window_buffer.py         ← 10-reading sliding window per patient (deque)
│   │   ├── rules/
│   │   │   ├── rule_a_artifact.py       ← ±40 BPM delta reject | ±5% SpO₂ | ±0.5°C temp
│   │   │   ├── rule_b_exertion.py       ← HR elev >15 BPM AND motion_score >4 → exertion
│   │   │   ├── rule_c_trajectory.py     ← deviation>1.5σ AND accel>0 AND motion<2 → FIRE
│   │   │   └── rule_runner.py           ← Runs A→B→C in sequence, returns RuleResult
│   │   └── buffer/
│   │       ├── patient_buffer.py        ← In-memory dict[patient_id → deque], Redis sync
│   │       └── ring_buffer.py           ← Generic ring buffer implementation
│   │
│   ├── services/
│   │   ├── mqtt/
│   │   │   ├── client.py                ← aiomqtt connection setup
│   │   │   ├── subscriber.py            ← Async subscribe loop (synera/patient/+/vitals)
│   │   │   └── parser.py                ← Parse MQTT JSON → VitalPayload schema
│   │   ├── database/
│   │   │   ├── patient_repo.py          ← Patient CRUD (async SQLAlchemy)
│   │   │   ├── alert_repo.py            ← Alert events CRUD
│   │   │   ├── vitals_repo.py           ← Vitals history writes + reads
│   │   │   └── medid_repo.py            ← MedID baseline update logic
│   │   └── notifications/
│   │       ├── alert_dispatcher.py      ← Sends WebSocket event when Synera State fires
│   │       └── sms_gateway.py           ← SMS stub (for rural fallback, no internet)
│   │
│   ├── models/
│   │   ├── schemas/                     ← Pydantic request/response schemas
│   │   │   ├── patient.py
│   │   │   ├── alert.py
│   │   │   ├── vitals.py
│   │   │   └── clinical_brief.py        ← ClinicalBrief output schema (see spec below)
│   │   └── orm/                         ← SQLAlchemy ORM models
│   │       ├── patient.py
│   │       ├── alert_event.py
│   │       ├── vitals_history.py
│   │       └── base.py
│   │
│   ├── config/
│   │   ├── settings.py                  ← pydantic-settings, loads .env
│   │   ├── database.py                  ← Async SQLAlchemy engine + session factory
│   │   └── mqtt.py                      ← MQTT broker connection config
│   │
│   └── tests/
│       ├── unit/
│       │   ├── test_rule_a.py
│       │   ├── test_rule_b.py
│       │   ├── test_rule_c.py
│       │   └── test_derivatives.py
│       └── integration/
│           ├── test_pipeline_end_to_end.py
│           └── test_mqtt_handler.py
│
├── rag/
│   ├── pipeline/
│   │   ├── rag_pipeline.py              ← Master pipeline: retrieval → LLM → parse
│   │   ├── clinical_brief_generator.py  ← AlertContext → ClinicalBrief JSON
│   │   └── alert_context_builder.py     ← Assembles context from patient + vitals
│   │
│   ├── knowledge_base/
│   │   ├── raw_docs/                    ← Drop PDFs here (MOHFW, WHO IMCI, IP2022)
│   │   ├── processed/
│   │   │   ├── chunker.py               ← PDF → section-aware chunks (300-400 tokens)
│   │   │   ├── embedder.py              ← BGE-M3 batch embedding
│   │   │   └── indexer.py               ← Writes chunks + embeddings to ChromaDB
│   │   └── embeddings/                  ← ChromaDB persistent vector store (gitignored)
│   │
│   ├── llm/
│   │   ├── prompts/
│   │   │   ├── clinical_brief.txt       ← Main alert brief prompt template
│   │   │   ├── triage_reasoning.txt     ← DRL triage context prompt
│   │   │   └── drug_interaction.txt     ← Polypharmacy check prompt
│   │   ├── chains/
│   │   │   ├── clinical_chain.py        ← LangChain chain: retrieve → prompt → LLM
│   │   │   └── drug_chain.py            ← Drug interaction retrieval + check chain
│   │   └── parsers/
│   │       ├── brief_parser.py          ← JSON extraction + Pydantic validation
│   │       └── citation_parser.py       ← Extracts + formats source citations
│   │
│   ├── retrieval/
│   │   ├── retriever.py                 ← ChromaDB similarity search wrapper
│   │   ├── chroma_store.py              ← ChromaDB client init + collection management
│   │   └── reranker.py                  ← Cross-encoder reranking of top-10 → top-5
│   │
│   ├── evaluation/
│   │   ├── ragas_eval.py                ← RAGAS faithfulness + answer relevancy eval
│   │   └── precision_recall.py          ← Retrieval precision@k measurement
│   │
│   └── tests/
│       ├── test_pipeline.py             ← End-to-end RAG pipeline with mock alert
│       └── test_retrieval.py            ← Vector search accuracy tests
│
├── scripts/
│   ├── setup/
│   │   ├── install_deps.sh              ← pip install + Ollama model pull
│   │   ├── init_db.sh                   ← Run Alembic migrations + seed patients
│   │   └── seed_data.py                 ← Insert 5 mock patients into PostgreSQL
│   ├── data_gen/
│   │   ├── mock_simulator.py            ← MQTT publisher: 5 patient profiles
│   │   ├── patient_profiles.py          ← Patient scenario definitions
│   │   └── scenario_generator.py        ← Generates realistic vital trajectories
│   ├── deploy/
│   │   ├── deploy_pi.sh                 ← Raspberry Pi 4 deployment script
│   │   └── deploy_cloud.sh              ← Cloud/VPS deployment script
│   └── testing/
│       ├── demo_scenarios.py            ← Runs all 3 judge demo scenarios
│       └── load_test.py                 ← Locust load test: 100 concurrent patients
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.backend
│   │   ├── Dockerfile.rag               ← Separate container: heavier dependencies
│   │   └── mosquitto.conf               ← Eclipse Mosquitto MQTT broker config
│   ├── nginx/
│   │   └── nginx.conf                   ← Reverse proxy: /api → backend, /ws → websocket
│   ├── monitoring/
│   │   ├── prometheus.yml               ← Scrape config for FastAPI metrics
│   │   └── grafana_dashboard.json       ← Alert latency + patient count dashboard
│   └── ci_cd/
│       └── github_actions_ci.yml        ← Lint + test on push
│
└── docs/
    ├── architecture/
    │   ├── system_overview.md
    │   ├── data_flow.md
    │   └── api_contracts.md
    ├── api_reference/
    │   ├── rest_api.md
    │   └── websocket_events.md
    ├── hardware/
    │   ├── schematic.md
    │   └── bom.md
    └── research/
        ├── novelty_brief.md
        └── algorithm_spec.md
```

---

## TECH STACK — USE EXACTLY THESE VERSIONS

| Component | Technology | Notes |
|---|---|---|
| Backend framework | FastAPI (latest) | Async, WebSocket native |
| Database | PostgreSQL 15 + TimescaleDB | Hypertable for vitals_history |
| ORM | SQLAlchemy 2.x async + asyncpg | |
| Vector store | **ChromaDB** (persistent) | Matches `rag/knowledge_base/embeddings/` |
| Embeddings | sentence-transformers `BAAI/bge-m3` | 1024-dim, multilingual |
| LLM | **Ollama local** with `llama3.1:8b` | No external API. Ever. |
| RAG framework | **LangChain 0.2** | Use chains/, not raw LangChain |
| MQTT | aiomqtt (async) | QoS 1 |
| Queue | Redis Streams | MQTT → ML decoupling |
| Signal math | NumPy | Derivatives, sliding windows |
| Migrations | Alembic | |
| Config | pydantic-settings | |
| Containers | Docker + Docker Compose | |

---

## DATABASE SCHEMA

### PostgreSQL Tables

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- patients table (MedID store)
CREATE TABLE patients (
    patient_id          VARCHAR(20) PRIMARY KEY,           -- "PT-8472"
    name                VARCHAR(100) NOT NULL,
    dob                 DATE,
    gender              VARCHAR(10),
    blood_group         VARCHAR(5),
    abha_id             VARCHAR(50) UNIQUE,
    primary_language    VARCHAR(30) DEFAULT 'Hindi',
    ward                VARCHAR(50),
    bed_number          VARCHAR(10),
    admission_date      DATE,
    attending_clinician VARCHAR(100),
    
    -- Medical history (ICD-11 coded)
    diagnosed_conditions    JSONB DEFAULT '[]',
    past_surgeries          JSONB DEFAULT '[]',
    hospitalisation_history JSONB DEFAULT '[]',
    family_history          JSONB DEFAULT '[]',
    
    -- Pharmacological profile
    current_medications  JSONB DEFAULT '[]',   -- [{name, dose, frequency, since}]
    known_allergies      JSONB DEFAULT '[]',   -- [{substance, reaction_type, severity}]
    adverse_reactions    JSONB DEFAULT '[]',
    
    -- Physiological baseline (Synera-learned, rolling 7-day)
    baseline_hr_mean        FLOAT,
    baseline_hr_std         FLOAT,
    baseline_spo2_mean      FLOAT,
    baseline_spo2_std       FLOAT,
    baseline_temp_mean      FLOAT,
    baseline_bp_sys_mean    FLOAT,
    baseline_bp_dia_mean    FLOAT,
    baseline_last_updated   TIMESTAMPTZ,
    calibration_complete    BOOLEAN DEFAULT FALSE,
    
    -- Genomic risk tiers (no raw DNA stored)
    genomic_risk_cardiac         VARCHAR(10) DEFAULT 'Unknown',
    genomic_risk_respiratory     VARCHAR(10) DEFAULT 'Unknown',
    genomic_risk_sepsis          VARCHAR(10) DEFAULT 'Unknown',
    genomic_risk_diabetic        VARCHAR(10) DEFAULT 'Unknown',
    
    last_clinical_notes  TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

-- vitals_history table (TimescaleDB hypertable)
CREATE TABLE vitals_history (
    id              BIGSERIAL,
    patient_id      VARCHAR(20) REFERENCES patients(patient_id),
    recorded_at     TIMESTAMPTZ NOT NULL,
    heart_rate      FLOAT,
    spo2            FLOAT,
    temperature     FLOAT,
    sys_bp_est      FLOAT,
    dia_bp_est      FLOAT,
    motion_score    INTEGER,
    is_active       BOOLEAN,
    battery_pct     INTEGER,
    reconstruction_error  FLOAT,
    pre_alert       BOOLEAN DEFAULT FALSE,
    source          VARCHAR(20) DEFAULT 'wearable',
    PRIMARY KEY (id, recorded_at)
);
SELECT create_hypertable('vitals_history', 'recorded_at');

-- alert_events table
CREATE TABLE alert_events (
    alert_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id        VARCHAR(20) REFERENCES patients(patient_id),
    trigger_timestamp TIMESTAMPTZ DEFAULT NOW(),
    trigger_vital     VARCHAR(20),
    trigger_value     FLOAT,
    baseline_value    FLOAT,
    deviation_sigma   FLOAT,
    second_derivative FLOAT,
    motion_score      INTEGER,
    vitals_snapshot   JSONB,
    
    -- RAG output (full brief stored for audit + DRL training)
    rag_clinical_brief    JSONB,
    retrieved_chunk_ids   TEXT[],       -- ChromaDB chunk IDs used
    llm_generation_time_ms INTEGER,
    
    -- DRL training signals (filled by clinician response)
    priority_tier_assigned    VARCHAR(20),
    clinician_acknowledged    BOOLEAN DEFAULT FALSE,
    acknowledgement_timestamp TIMESTAMPTZ,
    response_time_minutes     FLOAT,
    alert_dismissed           BOOLEAN DEFAULT FALSE,
    patient_deteriorated_after_dismissal BOOLEAN DEFAULT FALSE,
    
    patient_state_before  VARCHAR(20),  -- STABLE / WATCH
    patient_state_after   VARCHAR(20),  -- SYNERA_STATE
    created_at            TIMESTAMPTZ DEFAULT NOW()
);
```

---

## CHROMADB COLLECTIONS — OVERVIEW

ChromaDB holds **two collections**. Never mix them. The full schema and embedding rules are in the **Embedding Architecture & Retrieval Strategy** section below.

```python
# Collection 1: "medical_knowledge"  — protocol chunks from MOHFW/WHO/IP2022
# Collection 2: "clinical_cases"     — 60 synthetic anonymised case summaries

CHROMA_COLLECTIONS = {
    "medical_knowledge": "Clinical protocol chunks (300-400 tokens each)",
    "clinical_cases":    "Anonymised outcome case summaries (~200 tokens each)",
}

KNOWLEDGE_SOURCES = {
    "MOHFW_Sepsis_Protocol_2023":  "MOHFW Sepsis Management Protocol 2023",
    "MOHFW_PostOp_2023":           "MOHFW Post-Operative Care Protocol 2023",
    "MOHFW_Cardiac_2022":          "MOHFW Cardiac Emergency Protocol 2022",
    "WHO_IMCI":                    "WHO Integrated Management of Childhood Illness",
    "IP2022":                      "Indian Pharmacopoeia 2022 - Drug Interactions",
}
# See "Embedding Architecture & Retrieval Strategy" section for full schemas,
# what NOT to embed, query construction rules, and two-stage retrieval flow.
```

---

## EMBEDDING ARCHITECTURE & RETRIEVAL STRATEGY

This section defines exactly what gets embedded, where it lives, what stays out of ChromaDB, and how the retrieval query is constructed from patient data. The agent must implement this precisely — wrong data in the wrong store is the most common failure mode in RAG systems.

---

### WHAT GETS EMBEDDED INTO CHROMADB (and what does NOT)

#### ✅ EMBED THIS — Collection 1: `medical_knowledge`

Protocol text chunks from static medical documents. These are the authoritative clinical rules the LLM reasons against.

Each chunk is 300–400 tokens of continuous text from one of these sources:

| Source ID | Content Type | Why It's Embedded |
|---|---|---|
| `MOHFW_Sepsis_Protocol_2023` | Sepsis recognition, blood culture timing, antibiotic initiation, ICU escalation criteria | Most common Synera State cause in post-op patients |
| `MOHFW_PostOp_2023` | Post-operative monitoring, surgical site infection signs, wound assessment, pain management | PT-0002 demo scenario uses this directly |
| `MOHFW_Cardiac_2022` | Tachycardia differentials, cardiac emergency response, ECG indications | Triggered when patient has cardiac history or genomic_risk_cardiac = High |
| `WHO_IMCI` | Paediatric danger signs, breathing rate thresholds, fever + tachycardia in children | Only retrieved when patient age < 18 |
| `IP2022` | Drug-drug interactions, contraindications, polypharmacy warnings (Indian Pharmacopoeia) | Retrieved when patient has ≥1 current medication |

**Chunk metadata that MUST be stored with every document:**
```python
{
    "source_document": "MOHFW_Sepsis_Protocol_2023",   # exact source ID from KNOWLEDGE_SOURCES
    "source_label":    "MOHFW Sepsis Management Protocol 2023",
    "section":         "Section 4.2 - Sepsis Management",
    "page_number":     18,
    "clinical_domain": "sepsis",   # sepsis | cardiac | respiratory | paediatric | pharmacology
    "keywords":        ["sepsis", "tachycardia", "blood culture", "antibiotics", "empirical"]
}
```

---

#### ✅ EMBED THIS — Collection 2: `clinical_cases`

Anonymised synthetic clinical case summaries. These give the LLM concrete outcome examples — not just protocol rules — so it can reason about "patients with this pattern had this outcome" rather than only "the protocol says do X."

**Generate exactly 60 synthetic cases programmatically** in `scripts/setup/seed_data.py`. Do not use real patient data. Generate them as realistic but fictional clinical narratives.

Each case document format (embed as plain text, ~200 tokens):
```
"[CASE] 58F, post-operative day 3, Type 2 Diabetes, Hypertension. 
HR accelerated from 84 to 138 over 22 minutes at rest. SpO2 declined 
to 90%. Temperature 38.4°C. Motion score 1 throughout.
DIAGNOSIS: Surgical site sepsis (confirmed blood culture: E. coli).
INTERVENTION: Blood cultures x2, IV Meropenem 1g started within 1 hour, 
O2 supplementation 4L/min, ICU transfer.
OUTCOME: Recovered. ICU stay 4 days. [MOHFW Sepsis Protocol §4.2]"
```

**Generate 60 cases across these clinical archetypes (12 per archetype):**

| Archetype | Key Features | Primary Protocol Referenced |
|---|---|---|
| Post-op sepsis | Tachycardia + SpO2 drop, diabetic, day 2-5, fever | MOHFW Sepsis + PostOp |
| Cardiac event | HR acceleration, chest pain history, cardiac risk High | MOHFW Cardiac |
| PE (pulmonary embolism) | Post-op, HR + SpO2 drop, no fever | MOHFW PostOp + Sepsis |
| Hypoglycaemia | Diabetic, tachycardia, diaphoresis, low temp | IP2022 + Sepsis |
| COPD exacerbation | SpO2 decline, respiratory history, older patient | MOHFW Respiratory |

**Case document metadata:**
```python
{
    "source_document": "clinical_cases",
    "archetype":       "post_op_sepsis",    # matches archetypes above
    "patient_age_range": "50-65",
    "conditions":      ["diabetes", "post_operative"],
    "trigger_vital":   "heart_rate",
    "outcome":         "sepsis_confirmed",
    "severity":        "high"
}
```

---

#### ❌ DO NOT EMBED THESE INTO CHROMADB

This is explicit. The agent must not embed any of the following:

| Data | Why NOT in ChromaDB | Where it lives instead |
|---|---|---|
| Raw vital sign readings (HR: 103, SpO2: 95) | Meaningless semantically. No protocol will match numbers. | TimescaleDB `vitals_history` table |
| Full patient MedID records | DPDP Act compliance. Patient data stays in one place. Also wrong retrieval paradigm — you fetch by patient_id, not by similarity. | PostgreSQL `patients` table |
| Alert history / past alerts | Queried by patient_id + time range, not by semantic similarity. | PostgreSQL `alert_events` table |
| Baseline statistics (mean HR, std) | Structured numerical data. Query by patient_id. | PostgreSQL `patients.baseline_hr_mean` etc. |
| The current alert's own vitals snapshot | This is injected directly into the prompt as a formatted table, not retrieved. | Built into AlertContext, passed to prompt builder |

---

### HOW PATIENT DATA SHAPES THE RETRIEVAL QUERY

This is the most important mechanism for personalisation. The retrieval query sent to ChromaDB is NOT a generic clinical term. It is constructed dynamically from the patient's MedID + the alert trigger. This is what makes each brief specific to this patient, not just any tachycardia patient.

**Build this function in `rag/pipeline/alert_context_builder.py`:**

```python
def build_retrieval_query(alert_context: AlertContext) -> str:
    """
    Constructs a semantically rich query string from the alert + patient MedID.
    This query is embedded with BGE-M3 and used to search both ChromaDB collections.
    
    CONSTRUCTION RULES (apply ALL that match, concatenate with spaces):
    """
    query_parts = []

    # 1. TRIGGER VITAL → clinical term mapping (always included)
    vital_term_map = {
        "heart_rate": "tachycardia heart rate acceleration",
        "spo2":       "hypoxia oxygen desaturation SpO2 decline",
        "blood_pressure": "hypotension blood pressure drop haemodynamic instability"
    }
    query_parts.append(vital_term_map.get(alert_context.trigger_vital, alert_context.trigger_vital))

    # 2. Always append "at rest acceleration" — motion_score < 2 is always true here
    query_parts.append("at rest unexplained acceleration trajectory")

    # 3. Patient's diagnosed conditions → condition keywords
    condition_keyword_map = {
        "Type 2 Diabetes":       "diabetic hyperglycaemia hypoglycaemia insulin",
        "Hypertension":          "hypertensive blood pressure antihypertensive",
        "COPD":                  "COPD chronic obstructive pulmonary respiratory exacerbation",
        "Atrial Fibrillation":   "atrial fibrillation arrhythmia cardiac anticoagulation",
        "CKD":                   "chronic kidney disease renal impairment electrolytes",
        "Post-appendectomy":     "post-operative surgical site infection sepsis wound",
        "Post-operative":        "post-operative surgical complication sepsis wound infection",
    }
    for condition in alert_context.patient.diagnosed_conditions:
        name = condition.get("name", "")
        for key, keywords in condition_keyword_map.items():
            if key.lower() in name.lower():
                query_parts.append(keywords)
                break
        else:
            query_parts.append(name)  # fallback: use condition name as-is

    # 4. Age → paediatric flag (triggers WHO IMCI retrieval)
    if alert_context.patient_age < 18:
        query_parts.append("paediatric child IMCI danger signs")

    # 5. Medications → drug names for IP2022 retrieval
    if alert_context.patient.current_medications:
        med_names = [m.get("name", "") for m in alert_context.patient.current_medications]
        query_parts.append("drug interaction " + " ".join(med_names))

    # 6. Genomic risk tiers → risk-specific terms
    if alert_context.patient.genomic_risk_cardiac in ["High", "Critical"]:
        query_parts.append("sudden cardiac event high cardiac risk arrhythmia")
    if alert_context.patient.genomic_risk_sepsis in ["High", "Critical"]:
        query_parts.append("sepsis susceptibility high risk early sepsis recognition")
    if alert_context.patient.genomic_risk_respiratory in ["High", "Critical"]:
        query_parts.append("respiratory failure high risk oxygen supplementation")

    # 7. SpO2 component (if SpO2 also declining, even if not primary trigger)
    vitals = alert_context.vitals_window
    if vitals and vitals[-1].spo2 < 95:
        query_parts.append("oxygen supplementation pulse oximetry SpO2 management")

    return " ".join(query_parts)

# EXAMPLE OUTPUT for PT-0002 (Priya Sharma, post-op diabetic):
# "tachycardia heart rate acceleration at rest unexplained acceleration trajectory
#  diabetic hyperglycaemia hypoglycaemia insulin post-operative surgical site infection
#  sepsis wound drug interaction Cefazolin Metformin Paracetamol
#  oxygen supplementation pulse oximetry SpO2 management"
```

---

### TWO-STAGE RETRIEVAL FLOW

The pipeline runs TWO parallel retrieval stages, then merges and reranks. This is implemented in `rag/pipeline/rag_pipeline.py`.

```python
async def retrieve_and_rerank(query: str, alert_context: AlertContext) -> list[RetrievedChunk]:
    """
    Stage 1 + Stage 2 run in parallel via asyncio.gather.
    Results merged, duplicates removed, reranked, top 6 returned.
    """
    
    query_embedding = embedding_service.embed_text(
        f"Represent this sentence for searching relevant passages: {query}"
    )

    # STAGE 1: Protocol knowledge — general + domain-filtered
    # Run 2 searches in parallel within stage 1:
    #   A) General search across all medical_knowledge (top 8)
    #   B) Domain-filtered search if patient has medications (top 4, pharmacology domain only)
    
    stage1_general = retriever.search(
        collection="medical_knowledge",
        embedding=query_embedding,
        top_k=8
    )
    
    stage1_drug = None
    if alert_context.patient.current_medications:
        stage1_drug = retriever.search(
            collection="medical_knowledge",
            embedding=query_embedding,
            top_k=4,
            where={"clinical_domain": "pharmacology"}   # ChromaDB metadata filter
        )
    
    # STAGE 2: Clinical cases — outcome examples
    # Filter by matching archetype if determinable from patient context
    stage2_cases = retriever.search(
        collection="clinical_cases",
        embedding=query_embedding,
        top_k=4,
        where={"trigger_vital": alert_context.trigger_vital}  # match alert type
    )

    # Run all in parallel
    results = await asyncio.gather(stage1_general, stage1_drug or asyncio.sleep(0), stage2_cases)
    
    # Merge all results, deduplicate by chunk id
    merged = deduplicate(results[0] + (results[1] or []) + results[2])
    
    # RERANK: cross-encoder scores merged results against the original query
    # Use: cross-encoder/ms-marco-MiniLM-L-6-v2
    # Input: (query, chunk_text) pairs
    # Output: relevance scores → sort descending → take top 6
    reranked = reranker.rerank(query=query, chunks=merged, top_k=6)
    
    return reranked
```

**Format retrieved chunks for the prompt:**
```python
def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Formats retrieved chunks for injection into the LLM prompt.
    Protocol chunks and case examples are formatted differently.
    """
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        if chunk.metadata["source_document"] == "clinical_cases":
            # Case examples: label clearly as outcome reference
            formatted.append(
                f"[CASE EXAMPLE {i}]\n{chunk.text}\n"
            )
        else:
            # Protocol chunks: include source citation
            formatted.append(
                f"[GUIDELINE {i} — {chunk.metadata['source_label']}, {chunk.metadata['section']}]\n"
                f"{chunk.text}\n"
            )
    return "\n".join(formatted)
```

---

### WHAT IS INJECTED DIRECTLY INTO THE PROMPT (NOT RETRIEVED)

These are assembled in `alert_context_builder.py` and passed directly to the prompt template. They are never searched semantically — they are always included in full:

```python
# 1. PATIENT SUMMARY (from PostgreSQL patients table)
patient_context = f"""
Name: {patient.name} | Age: {age} | Gender: {patient.gender} | Blood: {patient.blood_group}
Ward: {patient.ward} | Bed: {patient.bed_number}
Conditions: {', '.join([c['name'] for c in patient.diagnosed_conditions])}
Medications: {', '.join([f"{m['name']} {m['dose']} {m['frequency']}" for m in patient.current_medications])}
Allergies: {', '.join([a['substance'] for a in patient.known_allergies]) or 'None known'}
Genomic risk — Cardiac: {patient.genomic_risk_cardiac} | Respiratory: {patient.genomic_risk_respiratory} | Sepsis: {patient.genomic_risk_sepsis}
Last clinical notes: {patient.last_clinical_notes or 'None recorded'}
"""

# 2. VITALS WINDOW TABLE (last 10 readings, computed from sliding window buffer)
vitals_table = format_vitals_as_table(alert_context.vitals_window)
# Output looks like:
# T-50s: HR=82  SpO2=97% Temp=37.1 BP=118/76 Motion=1
# T-45s: HR=87  SpO2=97% Temp=37.1 BP=119/77 Motion=1
# T-40s: HR=91  SpO2=96% Temp=37.2 BP=120/78 Motion=1
# ...
# T-0s:  HR=125 SpO2=93% Temp=37.8 BP=121/79 Motion=1  ← ALERT FIRED HERE

# 3. TRAJECTORY DESCRIPTION (plain language, computed by alert_assembler)
trend_description = f"Heart rate rose from {baseline_hr} to {current_hr} BPM over {minutes_elapsed} minutes while patient was at rest. Rate of rise is accelerating (2nd derivative = {second_derivative:.4f})."

# 4. ALERT METADATA
alert_metadata = {
    "trigger_vital": alert_context.trigger_vital,
    "trigger_value": alert_context.trigger_value,
    "baseline_value": alert_context.baseline_value,
    "deviation_sigma": alert_context.deviation_sigma,
    "second_derivative": alert_context.second_derivative,
    "motion_score": alert_context.motion_score,
}
```

---

### SEED DATA REQUIREMENTS FOR DEV (No Real PDFs Needed)

Since real MOHFW/WHO PDFs may not be available during development, `scripts/setup/seed_data.py` must generate realistic synthetic content for both ChromaDB collections. This allows the full RAG pipeline to function end-to-end from day one.

**Generate and ingest the following synthetic chunks:**

```python
# Minimum viable seed data for development
# Each entry becomes one ChromaDB document

SYNTHETIC_PROTOCOL_CHUNKS = [
    # MOHFW Sepsis — 8 chunks
    {
        "source": "MOHFW_Sepsis_Protocol_2023", "section": "Section 2.1 - Sepsis Recognition",
        "text": "Sepsis should be suspected in any patient presenting with unexplained tachycardia (HR > 90 BPM) combined with altered mental status, hypotension, or fever above 38°C or below 36°C. In post-surgical patients, new onset tachycardia on day 2-5 should be treated as sepsis until proven otherwise. Early recognition is critical — each hour of delay in antibiotic administration increases mortality by 7%.",
        "domain": "sepsis"
    },
    {
        "source": "MOHFW_Sepsis_Protocol_2023", "section": "Section 4.2 - Sepsis Management",
        "text": "Blood cultures (2 sets from different sites) must be obtained before antibiotic administration. However, culture collection must not delay antibiotic initiation beyond 45 minutes. Empirical broad-spectrum antibiotics should be started within 1 hour of sepsis recognition. In community-acquired sepsis: Piperacillin-tazobactam 4.5g IV Q8H. In hospital-acquired (>48hrs): Meropenem 1g IV Q8H.",
        "domain": "sepsis"
    },
    {
        "source": "MOHFW_Sepsis_Protocol_2023", "section": "Section 3.1 - Sepsis in Diabetic Patients",
        "text": "Diabetic patients with sepsis present atypically. Fever may be absent in up to 30% of diabetic sepsis cases due to impaired immune response. Tachycardia with SpO2 decline in a diabetic post-operative patient should trigger immediate blood glucose check — hypoglycaemia can mimic sepsis presentation. If glucose < 70 mg/dL, treat hypoglycaemia first before escalating sepsis protocol.",
        "domain": "sepsis"
    },
    # ... 5 more sepsis chunks covering: fluid resuscitation, ICU criteria, monitoring, lactate, vasopressors

    # MOHFW PostOp — 6 chunks  
    {
        "source": "MOHFW_PostOp_2023", "section": "Section 6.3 - Post-Operative Monitoring",
        "text": "Post-operative patients should be monitored for tachycardia, fever, and wound signs for 72 hours post-surgery. Heart rate acceleration above the patient's personal baseline by more than 20% warrants immediate clinical assessment. SpO2 below 94% in a previously well-oxygenated post-operative patient is an emergency sign requiring immediate intervention.",
        "domain": "sepsis"
    },
    # ... 5 more post-op chunks

    # MOHFW Cardiac — 5 chunks
    {
        "source": "MOHFW_Cardiac_2022", "section": "Section 2.3 - Tachycardia Differentials",
        "text": "Unexplained tachycardia at rest should prompt evaluation for: (1) Sepsis/infection — most common in hospitalised patients, (2) Pulmonary embolism — especially in post-operative, immobile patients, (3) Cardiac arrhythmia — AF with rapid ventricular response, SVT, (4) Hypovolaemia — blood loss, dehydration, (5) Pain/anxiety — rule out by clinical assessment. ECG is mandatory for all new-onset tachycardia exceeding 120 BPM at rest.",
        "domain": "cardiac"
    },
    # ... 4 more cardiac chunks

    # WHO IMCI — 4 chunks (paediatric)
    # IP2022 — 8 chunks (drug interactions: Warfarin+cephalosporins, Metformin+contrast, Digoxin+diuretics, etc.)
]

# After generating synthetic protocol chunks, also generate the 60 clinical case summaries
# using the 5 archetypes × 12 cases formula described above.
# Both sets get embedded with BGE-M3 and ingested into their respective ChromaDB collections.
```

---

## CORE IMPLEMENTATION SPECIFICATIONS

### `backend/core/trajectory/derivatives.py`

```python
import numpy as np
from collections import deque

def compute_first_derivative(readings: list[float], dt: float = 5.0) -> list[float]:
    """Central difference formula: (r[n] - r[n-2]) / (2 * dt)"""
    # Return array of rates. Edges use forward/backward difference.
    ...

def compute_second_derivative(rates: list[float], dt: float = 5.0) -> list[float]:
    """Central difference on rates: acceleration of change."""
    ...

def is_sustained_acceleration(second_derivatives: list[float], window: int = 3) -> bool:
    """True if last `window` second derivatives are ALL positive AND increasing."""
    # Single-point spikes must NOT trigger this. Must be sustained.
    ...

def compute_sigma_deviation(current: float, mean: float, std: float) -> float:
    """How many standard deviations current reading is from personal baseline."""
    if std == 0: return 0.0
    return abs(current - mean) / std
```

### `backend/core/rules/rule_runner.py`

```python
from dataclasses import dataclass
from enum import Enum

class RuleResult(Enum):
    ARTIFACT    = "ARTIFACT"     # Rule A rejected — discard packet
    EXERTION    = "EXERTION"     # Rule B — log as routine exertion
    SYNERA_STATE = "SYNERA_STATE" # Rule C — fire alert + trigger RAG
    STABLE      = "STABLE"       # All rules passed, no alert condition
    WATCH       = "WATCH"        # Deviation >1σ but no acceleration yet

# Sequential pipeline: A → B → C
# If A fails: return ARTIFACT (stop, discard packet)
# If B catches: return EXERTION (stop, log silently)
# If C fires: return SYNERA_STATE (trigger full RAG pipeline)
# Otherwise: STABLE or WATCH based on deviation level
```

### `backend/core/synera_engine/engine.py`

```python
# This is the main orchestrator. It does:
# 1. Receives parsed VitalPayload from MQTT subscriber
# 2. Looks up patient baseline from DB (cached in Redis, TTL 5min)
# 3. Updates patient_buffer (window_buffer.py)
# 4. Runs rule_runner.run(payload, buffer, baseline)
# 5. If SYNERA_STATE:
#      a. Call rag_service.generate_brief(alert_context)  ← async, fire with timeout
#      b. Write alert_event to DB
#      c. Update patient state to SYNERA_STATE
#      d. Emit WebSocket event via alert_dispatcher
# 6. If WATCH: update state, emit STATE_CHANGE websocket event
# 7. If EXERTION: log to vitals_history, no alert
# 8. If ARTIFACT: discard, log warning only

# RAG call MUST complete within 3000ms. Use asyncio.wait_for(timeout=3.0)
# If RAG times out: emit alert with rule-based fallback brief, log timeout
```

### `rag/pipeline/rag_pipeline.py`

```python
# This is the RAG pipeline master. Instantiated once at startup.
# Key method: async def generate_brief(alert_context: AlertContext) -> ClinicalBrief

# Internal steps (ALL async, use asyncio.gather for parallel retrieval):
# 1. Build semantic query from alert context (see query building spec below)
# 2. Embed query with BGE-M3
# 3. PARALLEL:
#    A. retriever.search(embedding, top_k=5) — general knowledge
#    B. retriever.search(embedding, top_k=3, filter={"clinical_domain": "pharmacology"})
#       — drug interactions, only if patient has ≥1 medication
# 4. Reranker: cross-encoder rerank merged results → top 5
# 5. Build patient context string from AlertContext.patient (MedID)
# 6. Load prompt template from rag/llm/prompts/clinical_brief.txt
# 7. Render prompt with: alert data + retrieved chunks + patient context
# 8. Call clinical_chain.invoke(prompt)
# 9. Pass raw LLM output to brief_parser.parse() → ClinicalBrief
# 10. If parse fails: retry once with drug_chain approach
# 11. If retry fails: return rule_based_fallback_brief(alert_context)
# 12. Log generation time, return ClinicalBrief

def build_semantic_query(alert_context: AlertContext) -> str:
    """
    Build a rich query string for vector retrieval.
    Combine: trigger vital name + patient conditions + alert severity
    
    Example output:
    "tachycardia acceleration at rest post-operative day 2 diabetic patient 
     sepsis management blood culture fever SpO2 declining"
    
    Rules:
    - Always include trigger_vital (heart_rate → "tachycardia", spo2 → "hypoxia", etc.)
    - Include all diagnosed_conditions names
    - Include "at rest" since motion_score < 2 is always true here
    - Include "acceleration" to retrieve trajectory-relevant protocols
    - If patient age < 18: include "paediatric" to pull WHO IMCI chunks
    - If patient has medications: include medication names for IP2022 retrieval
    """
```

---

## OLLAMA PROMPT TEMPLATES

### `rag/llm/prompts/clinical_brief.txt`

```
SYSTEM:
You are Synera Clinical Co-Pilot, a medical decision support engine at Indian primary health centres.
You generate structured clinical briefs when a physiological trajectory alert fires.

RULES:
1. Output ONLY valid JSON. No prose. No markdown. No code blocks.
2. Never diagnose definitively. Frame as differential for clinical verification.
3. Base all reasoning on the retrieved guideline excerpts provided below.
4. Flag every drug interaction risk for the patient's current medications.
5. Reference MOHFW protocols by name and section where available.
6. If uncertain, state so in confidence_note. Never fabricate clinical facts.
7. Keep language clear for a rural nurse AND precise enough for a physician.
8. Recommended actions must be ordered by urgency (1 = do immediately).

USER:
## ALERT TRIGGER
Patient: {patient_id} | Ward: {ward} | Bed: {bed_number}
Alert time: {trigger_timestamp}

{trend_description}

Vital triggered: {trigger_vital} = {trigger_value}
Personal baseline: {baseline_value} (deviation: {deviation_sigma:.1f}σ)
Trajectory acceleration (2nd derivative): {second_derivative:.4f} — POSITIVE, GROWING
Motion score: {motion_score}/10 — PATIENT AT REST

## VITAL SIGNS — LAST 10 READINGS (5-second intervals)
{vitals_table}

## PATIENT MEDICAL RECORD (MedID)
Name: {patient_name} | Age: {patient_age} | Gender: {patient_gender} | Blood: {blood_group}

Conditions: {conditions_list}
Medications: {medications_list}
Allergies: {allergies_list}
Genomic risk — Cardiac: {genomic_cardiac} | Respiratory: {genomic_respiratory} | Sepsis: {genomic_sepsis}
Last notes: {last_clinical_notes}

## RETRIEVED MEDICAL GUIDELINES
{retrieved_chunks_formatted}

## OUTPUT JSON SCHEMA
{json_schema}

Output the JSON now:
```

### `rag/llm/prompts/drug_interaction.txt`

```
SYSTEM:
You are a clinical pharmacology checker for Synera.
Given a patient's current medications and an acute physiological alert,
identify drug interactions or medication effects that may be contributing to or
complicating the alert. Output ONLY JSON.

USER:
Patient medications: {medications_list}
Current alert: {alert_summary}
Indian Pharmacopoeia excerpts: {ip2022_chunks}

Return JSON array of drug_interaction_flags:
[{"medication": str, "flag": str, "severity": "Critical|Warning|Info"}]
```

---

## CLINICAL BRIEF — OUTPUT JSON SCHEMA

```python
# rag/pipeline/clinical_brief_generator.py must produce this exact schema
# Defined in backend/models/schemas/clinical_brief.py

class DifferentialItem(BaseModel):
    condition: str
    likelihood: Literal["Most Likely", "Possible", "Rule Out"]
    reasoning: str

class RecommendedAction(BaseModel):
    priority: int                        # 1=immediate, 2=urgent, 3=when possible
    action: str
    rationale: str

class DrugInteractionFlag(BaseModel):
    medication: str
    flag: str
    severity: Literal["Critical", "Warning", "Info"]

class Source(BaseModel):
    document: str                        # e.g. "MOHFW Sepsis Management Protocol 2023"
    section: str                         # e.g. "Section 4.2"
    relevance: str                       # one sentence explaining why this was retrieved

class ClinicalBrief(BaseModel):
    alert_id: str
    patient_id: str
    generated_at: str                    # ISO-8601
    trigger_summary: str                 # Plain language: what happened and why it's significant
    differential_diagnosis: list[DifferentialItem]
    recommended_actions: list[RecommendedAction]
    drug_interaction_flags: list[DrugInteractionFlag]
    relevant_history: list[str]          # Patient-specific facts that informed this brief
    sources: list[Source]
    confidence_note: str                 # Always end with: "This is decision support. Verify clinically."
    generation_time_ms: int
```

---

## WEBSOCKET EVENT CONTRACT

```python
# SYNERA_STATE alert event (emitted from backend/services/notifications/alert_dispatcher.py)
{
    "event_type": "SYNERA_STATE",
    "alert_id": "uuid-v4",
    "patient_id": "PT-0002",
    "priority_tier": "TIER_1",
    "trigger_timestamp": "2024-03-15T14:23:07Z",
    "trigger_summary": "Heart rate rose from 82 to 125 BPM over 18 minutes at rest. Rate of rise is accelerating.",
    "vitals_snapshot": {
        "heart_rate": 125, "spo2": 93, "temperature": 37.8,
        "sys_bp_est": 118, "dia_bp_est": 79, "motion_score": 1
    },
    "clinical_brief": { ...ClinicalBrief JSON... },
    "requires_acknowledgement": true
}

# State change event (STABLE → WATCH, no alert)
{
    "event_type": "STATE_CHANGE",
    "patient_id": "PT-0005",
    "new_state": "WATCH",
    "previous_state": "STABLE",
    "reason": "HR deviation 1.3σ above baseline. Trajectory flat. Monitoring.",
    "timestamp": "2024-03-15T14:21:00Z"
}

# Routine exertion event (Rule B caught, logged only)
{
    "event_type": "EXERTION_LOGGED",
    "patient_id": "PT-0003",
    "timestamp": "2024-03-15T14:20:30Z",
    "motion_score": 8,
    "hr_elevation": 38
}
```

---

## MOCK PATIENT SIMULATOR — 5 PROFILES

**File:** `scripts/data_gen/mock_simulator.py` + `patient_profiles.py`

This simulator publishes MQTT payloads and is the backbone of all testing. Build it first.

```python
PATIENT_PROFILES = {
    "PT-0001": {
        "name": "Rajesh Kumar", "age": 62, "gender": "Male",
        "scenario": "STABLE",
        "ward": "General Ward A", "bed": "12B",
        "conditions": [{"code": "I10", "name": "Hypertension"}, {"code": "E11", "name": "Type 2 Diabetes"}],
        "medications": [{"name": "Metformin", "dose": "500mg", "frequency": "BD"},
                       {"name": "Amlodipine", "dose": "5mg", "frequency": "OD"}],
        "baseline_hr": 88, "baseline_spo2": 96,
        "trajectory": "flat_with_noise",   # HR 85-91, SpO₂ 95-97, no drift
        "expected_outcome": "STABLE — no alert ever"
    },

    "PT-0002": {
        "name": "Priya Sharma", "age": 34, "gender": "Female",
        "scenario": "TRAJECTORY_ACCELERATION",  # ← THE DEMO PATIENT
        "ward": "Surgical Ward B", "bed": "4A",
        "conditions": [{"code": "Z48.8", "name": "Post-appendectomy Day 2"}, 
                      {"code": "E11", "name": "Type 2 Diabetes"}],
        "medications": [{"name": "Cefazolin", "dose": "1g IV", "frequency": "Q8H"},
                       {"name": "Metformin", "dose": "500mg", "frequency": "BD"},
                       {"name": "Paracetamol", "dose": "500mg", "frequency": "Q6H PRN"}],
        "baseline_hr": 82, "baseline_spo2": 97,
        "trajectory": "exponential_acceleration",
        # HR: 82 → 91 → 103 → 119 → 140 over 18 min (readings every 5 sec)
        # SpO₂: 97 → 96 → 95 → 93 → 91
        # motion_score: 1 throughout (patient at rest, post-op)
        "expected_outcome": "SYNERA_STATE fired at ~103 BPM (8-15 min before threshold)"
    },

    "PT-0003": {
        "name": "Arjun Mehta", "age": 28, "gender": "Male",
        "scenario": "EXERTION_FALSE_POSITIVE",
        "ward": "General Ward A", "bed": "7C",
        "conditions": [],
        "medications": [],
        "baseline_hr": 68, "baseline_spo2": 99,
        "trajectory": "exertion_spike",
        # HR: 68 → 106 → 116 over 90 seconds, motion_score=8 throughout
        # Then returns to 72 BPM as patient sits back down
        "expected_outcome": "EXERTION_LOGGED — NO alert. Rule B catches this."
    },

    "PT-0004": {
        "name": "Fatima Begum", "age": 71, "gender": "Female",
        "scenario": "GLITCH_SPIKE",
        "ward": "ICU Step-Down", "bed": "2",
        "conditions": [{"code": "J44", "name": "COPD"}, {"code": "I48", "name": "Atrial Fibrillation"}],
        "medications": [{"name": "Warfarin", "dose": "5mg", "frequency": "OD"},
                       {"name": "Salbutamol", "dose": "100mcg inhaler", "frequency": "PRN"},
                       {"name": "Digoxin", "dose": "0.25mg", "frequency": "OD"}],
        "baseline_hr": 79, "baseline_spo2": 94,
        "trajectory": "glitch_spike",
        # Normal readings: 79, 81, 80 → SPIKE: 228 → 81, 80 (back to normal)
        # The 228 is a sensor artifact (PPG displaced by movement)
        "expected_outcome": "ARTIFACT — packet silently discarded. No alert. No DB write."
    },

    "PT-0005": {
        "name": "Suresh Patel", "age": 55, "gender": "Male",
        "scenario": "SLOW_DRIFT_WATCH",
        "ward": "General Ward C", "bed": "9A",
        "conditions": [{"code": "I10", "name": "Hypertension"}, {"code": "N18.3", "name": "CKD Stage 3"}],
        "medications": [{"name": "Losartan", "dose": "50mg", "frequency": "OD"},
                       {"name": "Furosemide", "dose": "40mg", "frequency": "OD"}],
        "baseline_hr": 76, "baseline_spo2": 97,
        "trajectory": "slow_linear_drift",
        # HR rises linearly from 76 to 92 over 45 minutes
        # Second derivative near zero (linear = constant rate, no acceleration)
        # Should cross 1.5σ and enter WATCH, NOT SYNERA_STATE
        "expected_outcome": "WATCH tier — deviation triggered, no acceleration, no RAG call"
    }
}
```

**Trajectory generators in `scenario_generator.py`:**
- `flat_with_noise(baseline, noise=3)` — Gaussian noise around baseline
- `exponential_acceleration(start, end, duration_readings)` — accelerating curve
- `exertion_spike(baseline, peak, rise_readings, motion_score=8)` — fast rise + high motion
- `glitch_spike(baseline, glitch_value, glitch_index)` — single bad reading embedded in normal data
- `slow_linear_drift(start, end, duration_readings)` — constant rate, zero acceleration

---

## REST API ENDPOINTS

```
# MedID
POST   /api/v1/patients/                    Create patient MedID
GET    /api/v1/patients/{patient_id}        Get full patient record
PUT    /api/v1/patients/{patient_id}        Update MedID fields
GET    /api/v1/patients/                    List all patients (ward filter optional)

# Vitals
GET    /api/v1/patients/{patient_id}/vitals  Vitals history (pagination: page, limit, from_ts, to_ts)
GET    /api/v1/patients/{patient_id}/state   Current patient state (STABLE/WATCH/SYNERA_STATE)

# Alerts
GET    /api/v1/alerts/                      List all alerts (patient_id filter, tier filter)
GET    /api/v1/alerts/{alert_id}            Get alert + full clinical brief
POST   /api/v1/alerts/{alert_id}/acknowledge  Clinician acknowledges alert (logs response time)

# RAG (manual trigger for testing)
POST   /api/v1/rag/trigger                  Manually trigger RAG with mock AlertContext
GET    /api/v1/rag/search?q=...&top_k=5    Test ChromaDB retrieval

# Knowledge base (admin)
POST   /api/v1/knowledge/ingest             Trigger PDF ingestion (runs chunker+embedder+indexer)
GET    /api/v1/knowledge/chunks             List all chunks (source filter)
DELETE /api/v1/knowledge/chunks/{chunk_id}  Remove a chunk

# Health
GET    /api/v1/health                       Returns: DB status, Ollama status, Redis status, ChromaDB status
GET    /ws                                  WebSocket endpoint
```

---

## DOCKER COMPOSE

```yaml
# docker-compose.yml — production stack
# docker-compose.dev.yml — development (with volume mounts for hot reload)

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: synera
      POSTGRES_USER: synera
      POSTGRES_PASSWORD: synera_dev
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  mosquitto:
    image: eclipse-mosquitto:2
    volumes:
      - ./infra/docker/mosquitto.conf:/mosquitto/config/mosquitto.conf
    ports:
      - "1883:1883"

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
    # Entrypoint script: pull llama3.1:8b on first run
    # This downloads ~4.7GB — add a health check that waits for model ready

  backend:
    build:
      context: .
      dockerfile: infra/docker/Dockerfile.backend
    depends_on: [postgres, redis, mosquitto, ollama, rag]
    environment:
      DATABASE_URL: postgresql+asyncpg://synera:synera_dev@postgres:5432/synera
      REDIS_URL: redis://redis:6379
      MQTT_BROKER_HOST: mosquitto
      MQTT_BROKER_PORT: 1883
      RAG_SERVICE_URL: http://rag:8001
      OLLAMA_BASE_URL: http://ollama:11434
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend   # dev only, hot reload

  rag:
    build:
      context: .
      dockerfile: infra/docker/Dockerfile.rag
    depends_on: [ollama]
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
      OLLAMA_MODEL: llama3.1:8b
      EMBEDDING_MODEL: BAAI/bge-m3
      CHROMA_PERSIST_DIR: /app/rag/knowledge_base/embeddings
    ports:
      - "8001:8001"
    volumes:
      - ./rag:/app/rag
      - chroma_data:/app/rag/knowledge_base/embeddings

volumes:
  postgres_data:
  ollama_models:
  chroma_data:
```

---

## MAKEFILE SHORTCUTS

```makefile
# Makefile at repo root — for dev convenience

up:             docker compose -f docker-compose.dev.yml up -d
down:           docker compose down
logs:           docker compose logs -f backend rag
migrate:        docker compose exec backend alembic upgrade head
seed:           docker compose exec backend python scripts/setup/seed_data.py
seed-protocols: docker compose exec rag python -m rag.knowledge_base.processed.indexer --collection medical_knowledge
seed-cases:     docker compose exec rag python -m rag.knowledge_base.processed.indexer --collection clinical_cases
ingest:         make seed-protocols && make seed-cases
simulate:       python scripts/data_gen/mock_simulator.py
demo:           python scripts/testing/demo_scenarios.py
test:           docker compose exec backend pytest backend/tests/ -v
test-rag:       docker compose exec rag pytest rag/tests/ -v
ollama-pull:    docker compose exec ollama ollama pull llama3.1:8b
health:         curl http://localhost:8000/api/v1/health | python -m json.tool
chroma-check:   curl "http://localhost:8001/api/v1/knowledge/chunks" | python -m json.tool
```

---

## `.env.example`

```
# Database
DATABASE_URL=postgresql+asyncpg://synera:synera_dev@postgres:5432/synera
POSTGRES_PASSWORD=synera_dev

# Redis
REDIS_URL=redis://redis:6379

# MQTT
MQTT_BROKER_HOST=mosquitto
MQTT_BROKER_PORT=1883
MQTT_QOS=1

# RAG service (internal)
RAG_SERVICE_URL=http://rag:8001

# LLM
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TEMPERATURE=0.1
OLLAMA_TOP_P=0.9
RAG_TIMEOUT_SECONDS=3.0

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3
CHROMA_PERSIST_DIR=./rag/knowledge_base/embeddings
CHROMA_COLLECTION_PROTOCOLS=medical_knowledge
CHROMA_COLLECTION_CASES=clinical_cases
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_TOP_K_PROTOCOLS=8
RETRIEVAL_TOP_K_CASES=4
RETRIEVAL_FINAL_TOP_K=6

# Alert thresholds
ARTIFACT_REJECT_HR_DELTA=40
ARTIFACT_REJECT_SPO2_DELTA=5
EXERTION_MOTION_THRESHOLD=4
EXERTION_HR_ELEVATION=15
SYNERA_STATE_SIGMA_THRESHOLD=1.5
SYNERA_STATE_MOTION_MAX=2
SYNERA_ACCELERATION_WINDOW=3

# App
ENVIRONMENT=development
LOG_LEVEL=INFO
API_KEY=synera-dev-key-change-in-prod
```

---

## BUILD ORDER — FOLLOW EXACTLY

Build in this sequence. Test each step before moving to the next.

**Phase 1 — Infrastructure (do first)**
1. `docker-compose.yml` + `docker-compose.dev.yml` + all Dockerfiles
2. `.env.example` + `Makefile`
3. `backend/config/settings.py` (pydantic-settings, loads .env)
4. `backend/config/database.py` (async SQLAlchemy engine)
5. `backend/models/orm/` (all 4 ORM models)
6. Alembic migration: create all tables, run hypertable command
7. Verify: `make up && make migrate` works cleanly

**Phase 2 — Synera Engine Core**
8. `backend/core/trajectory/derivatives.py` + unit tests in `backend/tests/unit/test_derivatives.py`
9. `backend/core/trajectory/window_buffer.py` + `calculator.py`
10. `backend/core/rules/rule_a_artifact.py` + test
11. `backend/core/rules/rule_b_exertion.py` + test
12. `backend/core/rules/rule_c_trajectory.py` + test
13. `backend/core/rules/rule_runner.py`
14. `backend/core/buffer/patient_buffer.py` (in-memory + Redis sync)
15. `backend/core/synera_engine/state_manager.py`
16. Verify: unit tests for all 3 rules pass with mock data

**Phase 3 — MQTT + Simulator**
17. `backend/services/mqtt/parser.py` (MQTT JSON → VitalPayload)
18. `backend/services/mqtt/client.py` + `subscriber.py`
19. `scripts/data_gen/scenario_generator.py` (5 trajectory generators)
20. `scripts/data_gen/patient_profiles.py`
21. `scripts/data_gen/mock_simulator.py` (publishes MQTT every 5 sec)
22. Verify: run simulator, confirm MQTT broker receives payloads

**Phase 4 — RAG Pipeline**
23. `rag/knowledge_base/processed/chunker.py` (PDF → section-aware chunks)
24. `rag/knowledge_base/processed/embedder.py` (BGE-M3 batch embedding — uses instruction prefix for queries, raw text for documents)
25. `rag/knowledge_base/processed/indexer.py` (→ ChromaDB, handles BOTH `medical_knowledge` and `clinical_cases` collections)
26. `scripts/setup/seed_data.py`:
    - Generate and ingest the **synthetic protocol chunks** (min 30 chunks across all 5 KNOWLEDGE_SOURCES) into `medical_knowledge` collection
    - Generate and ingest **60 synthetic clinical case summaries** (12 per archetype × 5 archetypes) into `clinical_cases` collection
    - Insert 5 mock patients into PostgreSQL
    - Verify both ChromaDB collections have documents: `GET /api/v1/knowledge/chunks`
27. `rag/retrieval/chroma_store.py` — init client, create/get both collections
28. `rag/retrieval/retriever.py` — search wrapper with `collection` and `where` filter params
29. `rag/retrieval/reranker.py` (`cross-encoder/ms-marco-MiniLM-L-6-v2` — merge + rerank to top 6)
30. `rag/llm/parsers/brief_parser.py` + `citation_parser.py`
31. `rag/llm/chains/clinical_chain.py` (LangChain: two-stage retrieve → rerank → prompt → Ollama)
32. `rag/llm/chains/drug_chain.py`
33. `rag/pipeline/alert_context_builder.py` — includes `build_retrieval_query()` with full patient-data-shaped query logic
34. `rag/pipeline/clinical_brief_generator.py`
35. `rag/pipeline/rag_pipeline.py` (master pipeline with `retrieve_and_rerank()`)
36. Verify: `make test-rag` passes. Manual trigger for PT-0002 profile returns ClinicalBrief with sepsis differential + cited MOHFW source + case example.

**Phase 5 — Backend Orchestration**
36. `backend/api/websocket/manager.py` (ConnectionManager: broadcast to all)
37. `backend/services/notifications/alert_dispatcher.py`
38. `backend/core/synera_engine/engine.py` (the main orchestrator)
39. `backend/core/synera_engine/pipeline.py`
40. `backend/services/database/` repos (patient, alert, vitals, medid)
41. All REST API routes
42. `backend/main.py` — wire everything together, start MQTT subscriber on startup

**Phase 6 — Integration + Demo**
43. `scripts/setup/seed_data.py` — insert 5 mock patients into PostgreSQL
44. `scripts/testing/demo_scenarios.py` — runs all 3 judge scenarios, measures latency
45. Run: `make simulate` → watch PT-0002 trigger SYNERA_STATE → verify ClinicalBrief in API
46. Run: `make demo` → confirm all 3 scenarios produce expected outcomes (see below)

---

## SUCCESS CRITERIA — YOUR BUILD IS CORRECT WHEN:

| Test | Expected Result |
|---|---|
| PT-0002 simulator runs | SYNERA_STATE fires within 18 minutes of trajectory starting |
| RAG brief for PT-0002 | Differential includes sepsis, cites MOHFW Sepsis §4.2, includes a CASE EXAMPLE from `clinical_cases` collection |
| Drug flag for PT-0002 | Cefazolin + Metformin interaction flagged (IP2022 chunk retrieved via drug-filtered Stage 1B search) |
| PT-0003 bathroom walk | EXERTION_LOGGED only. Zero alerts. Clinician not disturbed. |
| PT-0004 HR=228 spike | Packet silently discarded (Rule A). No DB write. No WebSocket event. |
| PT-0005 slow drift | Patient enters WATCH tier. No SYNERA_STATE. No RAG called. |
| Query shaping test | `alert_context_builder.build_retrieval_query()` for PT-0002 includes: "tachycardia", "post-operative", "diabetic", "Cefazolin", "Metformin" |
| Two-collection retrieval | `GET /api/v1/rag/search?q=post-op+tachycardia+sepsis` returns chunks from BOTH `medical_knowledge` AND `clinical_cases` |
| What NOT embedded | `GET /api/v1/rag/search?q=HR=103+SpO2=95` returns zero results (raw vitals were never embedded) |
| ChromaDB isolation | Patient records NOT in ChromaDB — confirmed by checking `chroma_store.list_collections()` returns only `medical_knowledge` and `clinical_cases` |
| End-to-end latency | MQTT receipt → ClinicalBrief in WebSocket event: **< 3000ms** (log and assert this) |
| Patient data isolation | `/api/v1/health` confirms Ollama is local. Zero outbound API calls. |

---

## WHAT NOT TO BUILD

- **No frontend** — React dashboard comes in a separate sprint
- **No firmware** — ESP32 code is a separate module
- **No federated learning** — present as architecture only
- **No genomic analysis** — risk tiers come in as pre-computed patient data
- WebSocket: build endpoint, test with `wscat`, no browser client needed yet
- Physical Mosquitto setup: Docker handles it, no manual config needed

---

*ArogyaLink · Synera 2.0 · RAG Backend Sprint · Techgium Season 9*
*Build the brain. The eyes and hands come later.*
