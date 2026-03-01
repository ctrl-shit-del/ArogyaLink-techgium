-- ============================================================
-- SYNERA 2.0 — Supabase Schema
-- Run this in Supabase SQL Editor (Project → SQL Editor → New query)
-- ============================================================

-- Enable pgvector (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- PATIENTS TABLE (MedID store)
-- ============================================================
CREATE TABLE IF NOT EXISTS patients (
    patient_id          VARCHAR(20) PRIMARY KEY,
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

    diagnosed_conditions    JSONB DEFAULT '[]',
    past_surgeries          JSONB DEFAULT '[]',
    hospitalisation_history JSONB DEFAULT '[]',
    family_history          JSONB DEFAULT '[]',

    current_medications  JSONB DEFAULT '[]',
    known_allergies      JSONB DEFAULT '[]',
    adverse_reactions    JSONB DEFAULT '[]',

    baseline_hr_mean        FLOAT,
    baseline_hr_std         FLOAT,
    baseline_spo2_mean      FLOAT,
    baseline_spo2_std       FLOAT,
    baseline_temp_mean      FLOAT,
    baseline_bp_sys_mean     FLOAT,
    baseline_bp_dia_mean     FLOAT,
    baseline_last_updated   TIMESTAMPTZ,
    calibration_complete    BOOLEAN DEFAULT FALSE,

    genomic_risk_cardiac         VARCHAR(10) DEFAULT 'Unknown',
    genomic_risk_respiratory     VARCHAR(10) DEFAULT 'Unknown',
    genomic_risk_sepsis          VARCHAR(10) DEFAULT 'Unknown',
    genomic_risk_diabetic        VARCHAR(10) DEFAULT 'Unknown',

    last_clinical_notes  TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- VITALS HISTORY TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS vitals_history (
    id              BIGSERIAL PRIMARY KEY,
    patient_id      VARCHAR(20) REFERENCES patients(patient_id),
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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
    source          VARCHAR(20) DEFAULT 'wearable'
);

CREATE INDEX IF NOT EXISTS idx_vitals_patient_time
ON vitals_history(patient_id, recorded_at DESC);

-- ============================================================
-- ALERT EVENTS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS alert_events (
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
    rag_clinical_brief    JSONB,
    retrieved_chunk_ids   TEXT[],
    llm_generation_time_ms INTEGER,
    llm_provider          VARCHAR(20),
    priority_tier_assigned    VARCHAR(20),
    clinician_acknowledged    BOOLEAN DEFAULT FALSE,
    acknowledgement_timestamp TIMESTAMPTZ,
    response_time_minutes     FLOAT,
    alert_dismissed           BOOLEAN DEFAULT FALSE,
    patient_deteriorated_after_dismissal BOOLEAN DEFAULT FALSE,
    patient_state_before  VARCHAR(20),
    patient_state_after   VARCHAR(20),
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- VECTOR TABLE 1: medical_knowledge
-- ============================================================
CREATE TABLE IF NOT EXISTS medical_knowledge (
    id              BIGSERIAL PRIMARY KEY,
    chunk_id        VARCHAR(100) UNIQUE NOT NULL,
    source_document VARCHAR(100) NOT NULL,
    source_label    VARCHAR(200),
    section_title   VARCHAR(200),
    page_number     INTEGER,
    clinical_domain VARCHAR(50),
    keywords        TEXT[],
    chunk_text      TEXT NOT NULL,
    embedding       vector(1024),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_medical_knowledge_embedding
ON medical_knowledge USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================
-- VECTOR TABLE 2: clinical_cases
-- ============================================================
CREATE TABLE IF NOT EXISTS clinical_cases (
    id              BIGSERIAL PRIMARY KEY,
    case_id         VARCHAR(100) UNIQUE NOT NULL,
    archetype       VARCHAR(50),
    patient_age_range VARCHAR(20),
    conditions      TEXT[],
    trigger_vital   VARCHAR(20),
    outcome         VARCHAR(100),
    severity        VARCHAR(20),
    case_text       TEXT NOT NULL,
    embedding       vector(1024),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clinical_cases_embedding
ON clinical_cases USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================
-- VECTOR SEARCH FUNCTIONS
-- ============================================================

CREATE OR REPLACE FUNCTION search_medical_knowledge(
    query_embedding vector(1024),
    match_count     INT DEFAULT 8,
    domain_filter   TEXT DEFAULT NULL
)
RETURNS TABLE (
    chunk_id        VARCHAR,
    source_document VARCHAR,
    source_label    VARCHAR,
    section_title   VARCHAR,
    clinical_domain  VARCHAR,
    keywords        TEXT[],
    chunk_text      TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        mk.chunk_id,
        mk.source_document,
        mk.source_label,
        mk.section_title,
        mk.clinical_domain,
        mk.keywords,
        mk.chunk_text,
        1 - (mk.embedding <=> query_embedding) AS similarity
    FROM medical_knowledge mk
    WHERE (domain_filter IS NULL OR mk.clinical_domain = domain_filter)
    ORDER BY mk.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION search_clinical_cases(
    query_embedding  vector(1024),
    match_count      INT DEFAULT 4,
    vital_filter     TEXT DEFAULT NULL
)
RETURNS TABLE (
    case_id       VARCHAR,
    archetype     VARCHAR,
    conditions    TEXT[],
    trigger_vital VARCHAR,
    outcome       VARCHAR,
    case_text     TEXT,
    similarity    FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        cc.case_id,
        cc.archetype,
        cc.conditions,
        cc.trigger_vital,
        cc.outcome,
        cc.case_text,
        1 - (cc.embedding <=> query_embedding) AS similarity
    FROM clinical_cases cc
    WHERE (vital_filter IS NULL OR cc.trigger_vital = vital_filter)
    ORDER BY cc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
