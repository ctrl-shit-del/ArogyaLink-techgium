-- DRL Triage Agent (Section 4.4): columns for priority_tier and online training
-- Run in Supabase SQL Editor if not already present.

ALTER TABLE alert_events
  ADD COLUMN IF NOT EXISTS drl_confidence FLOAT;

-- Index for daily training time-range queries
CREATE INDEX IF NOT EXISTS idx_alert_events_trigger_timestamp
  ON alert_events (trigger_timestamp DESC);
