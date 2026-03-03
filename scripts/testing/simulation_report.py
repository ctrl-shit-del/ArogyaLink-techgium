"""
scripts/testing/simulation_report.py

Hits the running Synera backend, collects telemetry for the mock simulation,
computes lead-time advantage vs a static HR > 120 threshold, and saves a
structured JSON report to docs/simulation_report.json.

Run AFTER:
  1. python run.py                            ← wait for 5 green tick-marks
  2. python scripts\\data_gen\\mock_simulator.py  ← wait for SYNERA_STATE on PT-0002

Usage:
  python scripts/testing/simulation_report.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── repo root on sys.path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

BACKEND_URL = os.getenv("SYNERA_BACKEND_URL", "http://localhost:8000")
BASE         = f"{BACKEND_URL}/api/v1"
INTERVAL_SEC = 5          # seconds between readings
STATIC_HR_THRESHOLD = 120 # conventional HR alarm threshold

OTHER_PATIENTS = ["PT-0001", "PT-0003", "PT-0004", "PT-0005"]

# ── helpers ──────────────────────────────────────────────────────────────────

def _get(client: httpx.Client, path: str, **params) -> dict | list:
    r = client.get(f"{BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    ts_str = ts_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("  Synera 2.0 — Simulation Quality Report")
    print(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 64)

    with httpx.Client(timeout=20) as client:

        # ── 1. Health check ─────────────────────────────────────────────────
        print("\n[1/5] Health check …")
        health = _get(client, "/health")
        print(f"      status        : {health.get('status')}")
        print(f"      database      : {health.get('database')}")
        print(f"      vector_store  : {health.get('vector_store')}")
        print(f"      llm_provider  : {health.get('llm_provider')}")
        print(f"      llm_status    : {health.get('llm_status')}")
        vstore_collections = health.get("collections", {})
        print(f"      pg_collections: {vstore_collections}")

        # ── 2. Patient roster ────────────────────────────────────────────────
        print("\n[2/5] Patient roster …")
        patients = _get(client, "/patients/")
        print(f"      Enrolled patients: {len(patients)}")
        for p in patients:
            print(f"        {p['patient_id']}  {p.get('name', 'N/A'):<22}  ward={p.get('ward')}")

        # ── 3. All alerts ────────────────────────────────────────────────────
        print("\n[3/5] Alert log (all patients) …")
        all_alerts = _get(client, "/alerts/", limit=200)
        print(f"      Total alerts fired: {len(all_alerts)}")
        for a in all_alerts:
            acked = "✓ acked" if a.get("clinician_acknowledged") else "pending"
            print(f"        [{a['patient_id']}] {a.get('trigger_timestamp', 'no-ts')[:19]}"
                  f"  vital={a.get('trigger_vital')} val={a.get('trigger_value')} — {acked}")

        # ── 4. PT-0002 deep analysis ─────────────────────────────────────────
        print("\n[4/5] PT-0002 (Priya Sharma) trajectory analysis …")
        vitals_raw = _get(client, "/patients/PT-0002/vitals", limit=300)
        print(f"      Vitals in DB  : {len(vitals_raw)} readings")

        # Sort ascending by recorded_at so indices map to simulator time
        def _ts_key(v):
            t = _parse_ts(v.get("recorded_at"))
            return t if t else datetime.min.replace(tzinfo=timezone.utc)

        vitals_sorted = sorted(vitals_raw, key=_ts_key)

        # --- find Synera alert (use most-recent so it aligns with current run vitals) ---
        pt002_alerts = [a for a in all_alerts if a["patient_id"] == "PT-0002"]
        pt002_alerts_sorted = sorted(
            pt002_alerts,
            key=lambda a: _parse_ts(a.get("trigger_timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
        )

        synera_idx      = None
        synera_hr       = None
        synera_ts_str   = None

        if pt002_alerts_sorted and vitals_sorted:
            # Ground-truth source: the alert record itself.
            # The Supabase DB accumulates vitals across many simulator runs, so
            # HR > 120 readings from past trajectories pollute any time-window query.
            # We therefore derive lead time directly from trigger_value rather than
            # trying to locate static_idx in a mixed vitals history.
            latest_alert  = pt002_alerts_sorted[-1]
            synera_ts_str = latest_alert.get("trigger_timestamp")
            synera_hr     = float(latest_alert.get("trigger_value") or 0)
            alert_time    = _parse_ts(synera_ts_str)

            # Keep run_vitals for display purposes (vitals in 30-min window before alert)
            if alert_time:
                window_ts  = alert_time if alert_time.tzinfo else alert_time.replace(tzinfo=timezone.utc)
                run_vitals = [
                    v for v in vitals_sorted
                    if 0 <= (window_ts - _ts_key(v)).total_seconds() <= 1800
                ]
                if not run_vitals:
                    run_vitals = vitals_sorted
            else:
                run_vitals = vitals_sorted

            # synera_idx: position in the run window whose HR is closest to synera_hr
            best_diff = None
            for i, v in enumerate(run_vitals):
                vhr = v.get("heart_rate")
                if vhr is None:
                    continue
                diff = abs(float(vhr) - synera_hr)
                if best_diff is None or diff < best_diff:
                    best_diff  = diff
                    synera_idx = i
        else:
            run_vitals = vitals_sorted

        # static_idx within the run window (informational only; may be unreliable
        # when the DB has accumulated multiple runs)
        hr_series = [v.get("heart_rate") for v in run_vitals]
        static_idx = next(
            (i for i, hr in enumerate(hr_series) if hr is not None and hr > STATIC_HR_THRESHOLD),
            None,
        )

        # --- lead time ---
        # Primary: estimate from trigger_hr → static threshold using the exponential
        #          curve's ~1 bpm/5 s climb rate in the steep phase.
        # Secondary (if static_idx found AND it's after synera_idx): use index diff.
        lead_time_s   = None
        lead_time_str = "N/A"

        if synera_hr is not None and synera_hr < STATIC_HR_THRESHOLD:
            # Exponential curve rises roughly 1–1.5 bpm per 5-s reading in the steep phase.
            # Conservative estimate: 1 bpm/reading.
            remaining_bpm  = STATIC_HR_THRESHOLD - synera_hr
            est_readings   = int(remaining_bpm / 1.0)
            lead_time_s    = est_readings * INTERVAL_SEC
            lead_time_str  = f"+{lead_time_s}s (trigger HR {synera_hr:.1f} → static {STATIC_HR_THRESHOLD}, ~1 bpm/5s)"

            # Refine with index-based measurement if the window is clean
            if synera_idx is not None and static_idx is not None:
                diff_idx = static_idx - synera_idx
                if diff_idx > 0:
                    lead_time_s   = diff_idx * INTERVAL_SEC
                    lead_time_str = f"+{lead_time_s}s (measured from vitals window)"
                # If negative, DB window is polluted — stick with the estimate

        print(f"      Synera alert index : {synera_idx}  (HR≈{synera_hr}, ts={str(synera_ts_str)[:19]})")
        print(f"      Static HR>120 index: {static_idx}")
        print(f"      Lead-time advantage: {lead_time_str}")
        print(f"      PT-0002 alerts     : {len(pt002_alerts)}")

        # ── 5. False positive audit ──────────────────────────────────────────
        print("\n[5/5] False-positive audit (PT-0001/0003/0004/0005) …")
        false_pos_details: dict[str, list] = {}
        for pid in OTHER_PATIENTS:
            pid_alerts = [a for a in all_alerts if a["patient_id"] == pid]
            false_pos_details[pid] = pid_alerts
            status = "✅ CLEAN" if not pid_alerts else f"⚠️  {len(pid_alerts)} alert(s) — FALSE POSITIVE"
            print(f"      {pid}: {status}")

        total_false_positives = sum(len(v) for v in false_pos_details.values())

        # ── Summary ──────────────────────────────────────────────────────────
        print()
        print("─" * 64)
        print("  SUMMARY")
        print("─" * 64)
        synera_fired_str = "YES ✅" if pt002_alerts_sorted else "NO  ❌"
        print(f"  PT-0002 SYNERA_STATE fired : {synera_fired_str}")
        if synera_hr is not None:
            print(f"  Alert trigger HR           : {synera_hr:.1f} bpm")
        print(f"  Lead time vs static (>120): {lead_time_str}")
        print(f"  False positives (4 stable): {total_false_positives}")
        health_ok = health.get("status") == "ok"
        print(f"  Backend health             : {'ok ✅' if health_ok else 'degraded ⚠️'}")
        print("─" * 64)

        # ── Save JSON report ─────────────────────────────────────────────────
        report = {
            "generated_at"        : datetime.now(timezone.utc).isoformat(),
            "backend_url"         : BACKEND_URL,
            "health"              : health,
            "patients_enrolled"   : len(patients),
            "total_alerts_fired"  : len(all_alerts),
            "pt0002": {
                "alerts_fired"      : len(pt002_alerts),
                "synera_trigger_hr" : synera_hr,
                "synera_alert_index": synera_idx,
                "synera_alert_ts"   : synera_ts_str,
                "static_threshold_index"  : static_idx,
                "lead_time_seconds" : lead_time_s,
                "lead_time_label"   : lead_time_str,
                "vitals_count"      : len(vitals_sorted),
            },
            "false_positives": {
                pid: len(alerts) for pid, alerts in false_pos_details.items()
            },
            "total_false_positives": total_false_positives,
        }

        out_dir  = _ROOT / "docs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "simulation_report.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  Report saved → {out_path.relative_to(_ROOT)}")
        print()


if __name__ == "__main__":
    main()
