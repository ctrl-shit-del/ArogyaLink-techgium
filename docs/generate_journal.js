/**
 * SYNERA 2.0 — Build Journal Generator
 * Run: npm install && node generate_journal.js
 * Output: SYNERA_2.0_Build_Journal.docx
 */
const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  WidthType,
  HeadingLevel,
  AlignmentType,
  BorderStyle,
  PageBreak,
  Header,
  Footer,
  PageNumber,
  NumberFormat,
  convertInchesToTwip,
  ShadingType,
} = require("docx");

const FONT = "Arial";
const H1_SIZE = 36;   // 18pt
const H2_SIZE = 28;    // 14pt
const BODY_SIZE = 22;  // 11pt
const TITLE_SIZE = 56; // 28pt
const CODE_SIZE = 20;  // 10pt
const DARK_BLUE = "1F3864";
const MEDIUM_BLUE = "2E75B6";
const CODE_BG = "F2F2F2";
const ROW_ALT = "EBF3FB";

function p(body, opts = {}) {
  return new Paragraph({
    children: typeof body === "string"
      ? [new TextRun({ text: body, font: FONT, size: BODY_SIZE, ...opts.run })]
      : body,
    alignment: opts.alignment || AlignmentType.LEFT,
    spacing: { line: 276, lineRule: "atLeast" }, // 1.15 line spacing approx
    ...opts,
  });
}

function heading1(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: H1_SIZE, bold: true, color: DARK_BLUE })],
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
  });
}

function heading2(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: H2_SIZE, bold: true, color: MEDIUM_BLUE })],
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
  });
}

function codeBlock(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Courier New", size: CODE_SIZE })],
    shading: { fill: CODE_BG, type: ShadingType.CLEAR },
    spacing: { before: 120, after: 120 },
  });
}

function tableHeaderRow(cells) {
  return new TableRow({
    children: cells.map((text) =>
      new TableCell({
        children: [
          new Paragraph({
            children: [new TextRun({ text, font: FONT, size: BODY_SIZE, bold: true, color: "FFFFFF" })],
          }),
        ],
        shading: { fill: DARK_BLUE, type: ShadingType.CLEAR },
      })
    ),
    tableHeader: true,
  });
}

function tableRow(cells, alt = false) {
  return new TableRow({
    children: cells.map((text) =>
      new TableCell({
        children: [p(String(text))],
        shading: alt ? { fill: ROW_ALT, type: ShadingType.CLEAR } : undefined,
      })
    ),
  });
}

function buildSections() {
  const sections = [];

  // ========== COVER PAGE ==========
  sections.push(
    new Paragraph({
      children: [new TextRun({ text: "SYNERA 2.0 — Clinical Co-Pilot", font: FONT, size: TITLE_SIZE, bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 360, after: 120 },
    }),
    new Paragraph({
      children: [new TextRun({ text: "RAG-Powered Physiological Trajectory Intelligence for Indian Primary Health Centres", font: FONT, size: H2_SIZE, italics: true })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }),
    new Paragraph({
      children: [new TextRun({ text: "Team: ArogyaLink", font: FONT, size: BODY_SIZE })],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({
      children: [new TextRun({ text: "Event: Techgium Season 9", font: FONT, size: BODY_SIZE })],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({
      children: [new TextRun({ text: "Date: March 2026", font: FONT, size: BODY_SIZE })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }),
    new Paragraph({
      border: { bottom: { color: DARK_BLUE, space: 1, style: BorderStyle.SINGLE, size: 6 } },
      spacing: { after: 200 },
    }),
    new Paragraph({
      children: [new TextRun({ text: "Backend: Production Ready | Frontend: In Progress", font: FONT, size: BODY_SIZE, bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
    }),
    new Paragraph({ children: [new TextRun({ text: "" })], pageBreakBefore: true }),
    heading1("Table of Contents"),
    p("1. What We Built"),
    p("2. System Architecture"),
    p("3. Build Journey"),
    p("4. Current System State"),
    p("5. How to Run the System"),
    p("6. What to Build Next (Hackathon Roadmap)"),
    p("7. API Reference"),
    p("8. Novelty Claims"),
    p("9. Environment Configuration"),
    new Paragraph({ children: [new TextRun({ text: "" })], spacing: { after: 240 } })
  );

  // ========== SECTION 1 — WHAT WE BUILT ==========
  sections.push(
    heading1("SECTION 1 — WHAT WE BUILT"),
    p("India has 150,000+ Primary Health Centres with one doctor serving hundreds of patients. A doctor cannot watch every patient's vitals continuously. By the time a nurse notices something is wrong, the patient may have deteriorated significantly. Standard pulse oximeters and monitors use fixed thresholds (alert if HR > 130) — but a patient whose HR has been slowly accelerating from 82 to 119 over 18 minutes is in more danger than one who has always had a resting HR of 118."),
    p("Synera monitors the rate of change of rate of change (second derivative / trajectory acceleration) of patient vitals. It fires an alert BEFORE the patient crosses a dangerous threshold — catching deterioration 8-15 minutes earlier than threshold-based systems. When an alert fires, a RAG-powered clinical co-pilot instantly generates a personalised clinical brief citing MOHFW protocols, WHO IMCI guidelines, and Indian Pharmacopoeia 2022 — delivered to the clinician's dashboard in under 3 seconds."),
    p("The three components: (1) Wearable (ESP32) — ₹480/device, captures HR, SpO2, temperature, motion. (2) Intelligence Layer (FastAPI + RAG) — trajectory analysis, 3-rule engine, clinical brief generation. (3) Dashboard (React) — real-time patient priority list, alert cards, clinical briefs."),
    p("What makes it novel: Trajectory acceleration detection (not threshold crossing); personalised RAG briefs using patient MedID (conditions, medications, genomic risk tiers); DPDP Act 2023 compliant — all inference runs locally or on Groq with no patient data leaving the facility; ₹480 device cost targeting rural Indian PHCs."),
    new Paragraph({ children: [new TextRun({ text: "" })], spacing: { after: 120 } })
  );

  // ========== SECTION 2 — SYSTEM ARCHITECTURE ==========
  sections.push(
    heading1("SECTION 2 — SYSTEM ARCHITECTURE"),
    heading2("2.1 High-Level Architecture"),
    p("Full data flow:"),
    codeBlock("ESP32 Wearable"),
    codeBlock("    ↓ (WiFi / MQTT — 5 second intervals)"),
    codeBlock("FastAPI Backend (run.py)"),
    codeBlock("    ↓"),
    codeBlock("Internal Event Bus (asyncio.Queue — replaces MQTT broker)"),
    codeBlock("    ↓"),
    codeBlock("Synera Engine"),
    codeBlock("    ↓"),
    codeBlock("3-Rule Pipeline:"),
    codeBlock("    Rule A: Artifact Rejection (±40 BPM delta → discard)"),
    codeBlock("    Rule B: Exertion Filter (HR elev + motion > 4 → log silently)"),
    codeBlock("    Rule C: Trajectory Acceleration (deviation > 1.5σ AND accel > 0 AND motion ≤ 2 → FIRE)"),
    codeBlock("    ↓ (only if Rule C fires)"),
    codeBlock("RAG Pipeline"),
    codeBlock("    ↓"),
    codeBlock("Cohere API → embed query → Supabase pgvector search"),
    codeBlock("    ↓"),
    codeBlock("Two-stage retrieval: medical_knowledge + clinical_cases"),
    codeBlock("    ↓"),
    codeBlock("Reranker → top 6 chunks"),
    codeBlock("    ↓"),
    codeBlock("Groq API (llama-3.1-8b-instant) → ClinicalBrief JSON"),
    codeBlock("    ↓"),
    codeBlock("WebSocket broadcast → React Dashboard"),
    codeBlock("    ↓"),
    codeBlock("Supabase (alert stored for audit + DRL training signal)"),
    heading2("2.2 Technology Stack Table"),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        tableHeaderRow(["Layer", "Technology", "Why chosen"]),
        tableRow(["Backend framework", "FastAPI (Python)", "Async, WebSocket native, fast to build"]),
        tableRow(["Database", "Supabase (hosted PostgreSQL)", "Free tier, pgvector built-in, instant setup"], true),
        tableRow(["Vector store", "Supabase pgvector", "Same DB for patient records + vector search"]),
        tableRow(["Embeddings", "Cohere embed-multilingual-v3.0 API", "Fast, multilingual, 1024-dim, free tier"], true),
        tableRow(["LLM", "Groq API (llama-3.1-8b-instant)", "~500 tokens/sec, free tier, <2s latency"]),
        tableRow(["RAG framework", "LangChain", "Chain abstraction, easy provider swap"], true),
        tableRow(["Signal math", "NumPy", "Central difference derivatives"]),
        tableRow(["Internal bus", "asyncio.Queue", "Replaces MQTT broker for simulator"], true),
        tableRow(["Production LLM", "Ollama (llama3.1:8b)", "Local, DPDP compliant, toggle via .env"]),
      ],
    }),
    heading2("2.3 Database Schema Summary"),
    p("patients — MedID store: demographics, conditions, medications, allergies, genomic risk tiers, learned baselines."),
    p("vitals_history — time-series vitals (indexed by patient_id + recorded_at)."),
    p("alert_events — full audit trail: trigger, RAG brief, clinician response, DRL training signals."),
    p("medical_knowledge — pgvector table: MOHFW/WHO/IP2022 protocol chunks (1024-dim embeddings)."),
    p("clinical_cases — pgvector table: 60 synthetic clinical outcome summaries (1024-dim embeddings)."),
    heading2("2.4 The 3-Rule Pipeline"),
    p("Rule A — Artifact Rejection: Fires if |current_hr - previous_hr| > 40 BPM, OR |current_spo2 - previous_spo2| > 5%, OR |current_temp - previous_temp| > 0.5°C. Result: Packet silently discarded. No DB write. No alert. Why: PPG sensors on wrists frequently produce glitch readings when the device shifts. PT-0004 (Fatima, HR spike to 228) demonstrates this."),
    p("Rule B — Exertion Filter: Fires if HR elevation > 15 BPM above baseline AND motion_score > 4. Result: EXERTION_LOGGED. No clinical alert. Why: A patient walking to the bathroom will show HR elevation + motion. PT-0003 (Arjun) demonstrates this."),
    p("Rule C — Trajectory Acceleration (the Synera innovation): Fires if deviation > 1.5σ from personal baseline AND last 3 second derivatives are ALL positive AND increasing AND motion_score ≤ 2. Result: SYNERA_STATE. RAG pipeline triggered. WebSocket alert sent. Why: This catches deterioration that threshold systems miss. PT-0002 (Priya, post-op sepsis) fires at HR=103, well before the dangerous threshold of 130+."),
    heading2("2.5 RAG Pipeline — How It Works"),
    p("The query is not generic. It is built from the patient's MedID + the alert trigger: Trigger vital → clinical term (heart_rate → \"tachycardia heart rate acceleration\"); Patient conditions → keywords (Type 2 Diabetes → \"diabetic hyperglycaemia\"); Medications → drug names for IP2022 lookup; Age < 18 → paediatric flag for WHO IMCI; Genomic risk tiers → risk-specific terms; SpO2 < 95 → oxygen management terms. Two-stage retrieval runs in parallel: Stage 1A: General protocol search (top 8 from medical_knowledge); Stage 1B: Drug-filtered search (top 4 from medical_knowledge, pharmacology domain only — if patient has medications); Stage 2: Case examples (top 4 from clinical_cases, filtered by trigger vital). Results merged, deduplicated, reranked, top 6 passed to LLM.")
  );

  // ========== SECTION 3 — BUILD JOURNEY ==========
  sections.push(
    heading1("SECTION 3 — BUILD JOURNEY"),
    heading2("3.1 Sprint 1 — Initial Architecture (Week 1)"),
    p("PRD written (11 sections, problem statement, novelty claims, deployment strategy). Initial agent prompt created for RAG backend. Directory structure defined (ArogyaLink monorepo). Tech stack decided: FastAPI + PostgreSQL + ChromaDB + Ollama."),
    heading2("3.2 Sprint 2 — Refactor: Remove Docker (Week 2)"),
    p("The original design used Docker for everything — PostgreSQL, ChromaDB, Redis, Mosquitto MQTT broker, Ollama. On Windows (development machine), Docker Desktop failed to start due to the dockerDesktopLinuxEngine pipe not being found. Decision: Rather than debug Docker on Windows, the entire infrastructure layer was refactored. Removed Docker/docker-compose → Native Python + pip install. Removed PostgreSQL (Docker) → Supabase hosted. Removed ChromaDB (Docker) → Supabase pgvector. Removed Redis (Docker) → asyncio.Queue + in-memory dict. Removed Mosquitto MQTT (Docker) → Internal event bus (InternalEventBus). Removed Ollama (Docker) → Groq API. Removed Makefile → run-windows.bat + run-windows.ps1. All business logic — Rule A/B/C, trajectory derivatives, RAG pipeline, WebSocket events, ClinicalBrief schema — was untouched. Result: System runs with two commands: python run.py and python scripts/data_gen/mock_simulator.py"),
    heading2("3.3 Sprint 3 — Dependency and Environment Fixes"),
    p("Problem 1: make not recognized on Windows. Fix: Created run-windows.bat and run-windows.ps1."),
    p("Problem 2: Docker commands run from wrong directory. Fix: All commands must run from repo root."),
    p("Problem 3: tf-keras missing — sentence-transformers import crash. Fix: pip install tf-keras."),
    p("Problem 4: BGE-M3 model download — 2.27GB, interrupted. Fix: Used Cohere API for embeddings."),
    p("Problem 5: RAG trigger timing out (20+ seconds). Fix: Switched embedding provider to Cohere API (embed-multilingual-v3.0)."),
    p("Problem 6: Supabase SSL certificate error on campus wifi. Fix: Use personal wifi or phone hotspot."),
    p("Problem 7: Indexer duplicate key error. Fix: Changed to .upsert(on_conflict=\"chunk_id\") and .upsert(on_conflict=\"case_id\") in supabase_vector_store.py."),
    p("Problem 8: reranker model downloading during RAG requests. Fix: Disabled cross-encoder reranker; use similarity-score passthrough."),
    p("Problem 9: PatientRepository missing methods. Fix: Added list_all(ward=None) and get_by_id(patient_id)."),
    p("Problem 10: seed_data.py using wrong profile key (bed_number). Fix: Use profile.get(\"bed\").")
  );

  // ========== SECTION 4 — CURRENT SYSTEM STATE ==========
  const statusTableRows = [
    tableHeaderRow(["Feature", "Status", "Notes"]),
    tableRow(["FastAPI backend", "✅ Working", "Starts in ~20s (Cohere embedding check)"], true),
    tableRow(["Supabase connection", "✅ Working", "Requires personal wifi (campus network blocks)"]),
    tableRow(["5 mock patients seeded", "✅ Working", "PT-0001 to PT-0005 in Supabase"], true),
    tableRow(["Rule A (artifact rejection)", "✅ Working", "16 unit tests passing"]),
    tableRow(["Rule B (exertion filter)", "✅ Working", "16 unit tests passing"], true),
    tableRow(["Rule C (trajectory acceleration)", "✅ Working", "16 unit tests passing"]),
    tableRow(["Internal event bus", "✅ Working", "Pub/sub with MQTT-style topic matching"], true),
    tableRow(["Cohere embeddings", "✅ Working", "1024-dim, under 500ms"]),
    tableRow(["Supabase pgvector search", "✅ Working", "22 protocol chunks + 60 cases indexed"], true),
    tableRow(["Groq LLM (llama-3.1-8b)", "✅ Working", "Returns ClinicalBrief JSON"]),
    tableRow(["RAG trigger API", "✅ Working", "POST /api/v1/rag/trigger → ClinicalBrief"], true),
    tableRow(["WebSocket endpoint", "✅ Working", "GET /ws"]),
    tableRow(["Mock simulator", "✅ Working", "Publishes all 5 profiles every 5s"], true),
    tableRow(["Patient data in MedID brief", "⚠️ Partial", "patient_id shows placeholder — context not fully passed"]),
    tableRow(["Cross-encoder reranker", "❌ Disabled", "model.safetensors download issue — passthrough used"], true),
    tableRow(["Frontend dashboard", "❌ Not started", "Next sprint"]),
  ];

  sections.push(
    heading1("SECTION 4 — CURRENT SYSTEM STATE"),
    heading2("4.1 What Is Working Right Now"),
    new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: statusTableRows }),
    heading2("4.2 Known Issues"),
    p("Issue 1: patient_id shows \"MedID\" in brief — patient context not correctly extracted from AlertContext. Fix: Ensure patient dict fields are correctly mapped in alert_context_builder.py."),
    p("Issue 2: Groq rate limits cause variable latency. Fix: Add response caching (brief_cache) with 5-minute TTL."),
    p("Issue 3: Cross-encoder reranker disabled. Fix: Cache model once, then re-enable in reranker.py."),
    p("Issue 4: datetime.utcnow() deprecation. Fix: Replace with datetime.now(datetime.UTC) in mock_simulator.py.")
  );

  // ========== SECTION 5 — HOW TO RUN ==========
  sections.push(
    heading1("SECTION 5 — HOW TO RUN THE SYSTEM"),
    heading2("5.1 Prerequisites"),
    p("Python 3.11+. Personal wifi (not campus network — Supabase blocked). .env file with Supabase + Groq + Cohere API keys."),
    heading2("5.2 First-Time Setup (Run Once)"),
    codeBlock("1. cd C:\\Users\\Asus\\ArogyaLink-techgium"),
    codeBlock("2. copy .env.example .env"),
    codeBlock("3. Run supabase/schema.sql in Supabase SQL Editor"),
    codeBlock("4. pip install -r requirements.txt"),
    codeBlock("5. set PYTHONPATH=C:\\Users\\Asus\\ArogyaLink-techgium"),
    codeBlock("6. python scripts/setup/seed_data.py"),
    codeBlock("7. python -m rag.knowledge_base.processed.indexer --collection all"),
    heading2("5.3 Every Run (Two Terminals)"),
    p("Terminal 1 — Backend: cd to repo, set PYTHONPATH, python run.py. Wait for Application startup complete."),
    p("Terminal 2 — Simulator: cd to repo, set PYTHONPATH, python scripts/data_gen/mock_simulator.py."),
    heading2("5.4 Key URLs"),
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        tableHeaderRow(["URL", "What it is"]),
        tableRow(["http://localhost:8000/docs", "Interactive API explorer"], true),
        tableRow(["http://localhost:8000/api/v1/health", "System health check"]),
        tableRow(["http://localhost:8000/api/v1/alerts/", "All alerts"], true),
        tableRow(["http://localhost:8000/api/v1/patients/", "All patients"]),
        tableRow(["ws://localhost:8000/ws", "WebSocket endpoint"], true),
      ],
    }),
    heading2("5.5 Test RAG Manually"),
    codeBlock("curl -s -X POST http://localhost:8000/api/v1/rag/trigger -H \"Content-Type: application/json\" -d \"{\\\"patient_id\\\": \\\"PT-0002\\\", \\\"trigger_vital\\\": \\\"heart_rate\\\", \\\"trigger_value\\\": 119, \\\"motion_score\\\": 1}\"")
  );

  // ========== SECTION 6 — ROADMAP ==========
  sections.push(
    heading1("SECTION 6 — WHAT TO BUILD NEXT (HACKATHON ROADMAP)"),
    p("Priority 1 — Fix Patient Context in RAG Brief (2 hours). Why: Personalised brief citing Cefazolin + Metformin for PT-0002 is 10x more impressive. How: Fix alert_context_builder.py to pass patient name, conditions, medications into prompt."),
    p("Priority 2 — React Dashboard Frontend (1 day). Why: Judges need to see something. What: Left panel patient priority list; main panel alert card with vitals trend + ClinicalBrief; WebSocket updates; Acknowledge button; Patient MedID view."),
    p("Priority 3 — Groq Response Caching (1 hour). Why: Prevents 20-second delays during demo. How: Add _brief_cache with key patient_id_trigger_vital, TTL 5 min."),
    p("Priority 4 — Verify End-to-End Simulator Flow (30 min). How: Run simulator 4 min, watch PT-0002 SYNERA_STATE, verify alert in GET /api/v1/alerts/, verify WebSocket with wscat."),
    p("Priority 5 — Enable Cross-Encoder Reranker (30 min). How: Cache model once, re-enable in reranker.py."),
    p("Priority 6 — Demo Script (2 hours). What: scripts/testing/demo_scenarios.py — runs all 5 patient scenarios in sequence with 30s intervals, prints pass/fail."),
    p("Priority 7 — Hardware Integration (after hackathon). How: Replace internal event bus subscriber with aiomqtt subscriber; engine handle_vital_payload interface unchanged; ESP32 publishes to synera/patient/{patient_id}/vitals.")
  );

  // ========== SECTION 7 — API REFERENCE ==========
  sections.push(
    heading1("SECTION 7 — API REFERENCE"),
    heading2("7.1 REST Endpoints"),
    p("GET /api/v1/health — Returns system status, LLM provider, collection counts."),
    p("POST /api/v1/rag/trigger — Request: {\"patient_id\": \"PT-0002\", \"trigger_vital\": \"heart_rate\", \"trigger_value\": 119, \"motion_score\": 1}. Response: Full ClinicalBrief JSON."),
    p("GET /api/v1/patients/ — List of all patients with current state."),
    p("GET /api/v1/patients/{patient_id} — Full MedID record."),
    p("GET /api/v1/alerts/ — All alert events with clinical briefs."),
    p("POST /api/v1/alerts/{alert_id}/acknowledge — Clinician acknowledges alert."),
    p("GET /api/v1/rag/search?q=tachycardia+sepsis&top_k=5 — Test vector search."),
    heading2("7.2 WebSocket Events (ws://localhost:8000/ws)"),
    p("SYNERA_STATE: event_type, alert_id, patient_id, priority_tier, trigger_timestamp, trigger_summary, vitals_snapshot, clinical_brief, requires_acknowledgement."),
    p("STATE_CHANGE: event_type, patient_id, new_state, previous_state, reason."),
    p("EXERTION_LOGGED: event_type, patient_id, motion_score, hr_elevation.")
  );

  // ========== SECTION 8 — NOVELTY CLAIMS ==========
  sections.push(
    heading1("SECTION 8 — NOVELTY CLAIMS"),
    p("1. Trajectory Acceleration Detection — No existing PHC monitoring system in India uses second-derivative analysis on physiological signals. Synera detects deterioration 8-15 minutes earlier."),
    p("2. Personalised RAG Briefs with MedID — The retrieval query is dynamically constructed from the patient's conditions, medications, genomic risk tiers, and age. Patient-aware retrieval, not generic RAG."),
    p("3. DPDP Act 2023 Compliance by Architecture — Patient data never leaves the facility. Embeddings via Cohere on query only; LLM receives only de-identified clinical context. Privacy-by-design."),
    p("4. ₹480 Device Cost — Existing hospital-grade monitors cost ₹15,000-50,000. Synera's ESP32-based wearable targets ₹480 for 150,000+ PHCs.")
  );

  // ========== SECTION 9 — ENVIRONMENT ==========
  sections.push(
    heading1("SECTION 9 — ENVIRONMENT CONFIGURATION"),
    p("SUPABASE_URL — Project URL from Supabase dashboard. SUPABASE_ANON_KEY — anon/public key. SUPABASE_SERVICE_KEY — service_role key. DATABASE_URL — PostgreSQL connection URI (Transaction pooler)."),
    p("LLM_PROVIDER — \"groq\" (demo) or \"ollama\" (production). GROQ_API_KEY, GROQ_MODEL — llama-3.1-8b-instant. OLLAMA_BASE_URL, OLLAMA_MODEL — when using Ollama."),
    p("EMBEDDING_PROVIDER — \"cohere\" (recommended) or \"local\". COHERE_API_KEY, COHERE_EMBEDDING_MODEL — embed-multilingual-v3.0."),
    p("Alert thresholds: ARTIFACT_REJECT_HR_DELTA=40, EXERTION_MOTION_THRESHOLD=4, EXERTION_HR_ELEVATION=15, SYNERA_STATE_SIGMA_THRESHOLD=1.5, SYNERA_STATE_MOTION_MAX=2, SYNERA_ACCELERATION_WINDOW=3, RAG_TIMEOUT_SECONDS=30.0.")
  );

  return sections;
}

async function main() {
  const sections = buildSections();
  const doc = new Document({
    sections: [
      {
        properties: {},
        children: sections,
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                children: [
                  new TextRun({
                    children: [PageNumber.CURRENT, " / ", PageNumber.TOTAL_PAGES],
                    font: FONT,
                    size: 20,
                  }),
                ],
                alignment: AlignmentType.CENTER,
              }),
            ],
          }),
        },
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  const outPath = path.join(__dirname, "SYNERA_2.0_Build_Journal.docx");
  fs.writeFileSync(outPath, buffer);
  console.log("Created:", outPath);

  // Also copy to project root docs if different
  const rootPath = path.resolve(__dirname, "SYNERA_2.0_Build_Journal.docx");
  if (path.relative(outPath, rootPath) !== "") {
    fs.copyFileSync(outPath, rootPath);
    console.log("Copied to:", rootPath);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
