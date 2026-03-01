"""Initial schema: patients, vitals_history (hypertable), alert_events.

Revision ID: 001
Revises:
Create Date: 2024-03-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    op.create_table(
        "patients",
        sa.Column("patient_id", sa.String(20), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("dob", sa.Date(), nullable=True),
        sa.Column("gender", sa.String(10), nullable=True),
        sa.Column("blood_group", sa.String(5), nullable=True),
        sa.Column("abha_id", sa.String(50), nullable=True),
        sa.Column("primary_language", sa.String(30), server_default="Hindi"),
        sa.Column("ward", sa.String(50), nullable=True),
        sa.Column("bed_number", sa.String(10), nullable=True),
        sa.Column("admission_date", sa.Date(), nullable=True),
        sa.Column("attending_clinician", sa.String(100), nullable=True),
        sa.Column("diagnosed_conditions", postgresql.JSONB(astext_type=sa.Text()), server_default="[]"),
        sa.Column("past_surgeries", postgresql.JSONB(astext_type=sa.Text()), server_default="[]"),
        sa.Column("hospitalisation_history", postgresql.JSONB(astext_type=sa.Text()), server_default="[]"),
        sa.Column("family_history", postgresql.JSONB(astext_type=sa.Text()), server_default="[]"),
        sa.Column("current_medications", postgresql.JSONB(astext_type=sa.Text()), server_default="[]"),
        sa.Column("known_allergies", postgresql.JSONB(astext_type=sa.Text()), server_default="[]"),
        sa.Column("adverse_reactions", postgresql.JSONB(astext_type=sa.Text()), server_default="[]"),
        sa.Column("baseline_hr_mean", sa.Float(), nullable=True),
        sa.Column("baseline_hr_std", sa.Float(), nullable=True),
        sa.Column("baseline_spo2_mean", sa.Float(), nullable=True),
        sa.Column("baseline_spo2_std", sa.Float(), nullable=True),
        sa.Column("baseline_temp_mean", sa.Float(), nullable=True),
        sa.Column("baseline_bp_sys_mean", sa.Float(), nullable=True),
        sa.Column("baseline_bp_dia_mean", sa.Float(), nullable=True),
        sa.Column("baseline_last_updated", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("calibration_complete", sa.Boolean(), server_default="false"),
        sa.Column("genomic_risk_cardiac", sa.String(10), server_default="Unknown"),
        sa.Column("genomic_risk_respiratory", sa.String(10), server_default="Unknown"),
        sa.Column("genomic_risk_sepsis", sa.String(10), server_default="Unknown"),
        sa.Column("genomic_risk_diabetic", sa.String(10), server_default="Unknown"),
        sa.Column("last_clinical_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_patients_abha_id", "patients", ["abha_id"], unique=True)

    op.create_table(
        "vitals_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("patient_id", sa.String(20), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("recorded_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("heart_rate", sa.Float(), nullable=True),
        sa.Column("spo2", sa.Float(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("sys_bp_est", sa.Float(), nullable=True),
        sa.Column("dia_bp_est", sa.Float(), nullable=True),
        sa.Column("motion_score", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("battery_pct", sa.Integer(), nullable=True),
        sa.Column("reconstruction_error", sa.Float(), nullable=True),
        sa.Column("pre_alert", sa.Boolean(), server_default="false"),
        sa.Column("source", sa.String(20), server_default="wearable"),
        sa.PrimaryKeyConstraint("id", "recorded_at"),
    )
    op.execute("SELECT create_hypertable('vitals_history', 'recorded_at');")

    op.create_table(
        "alert_events",
        sa.Column("alert_id", postgresql.UUID(as_uuid=False), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("patient_id", sa.String(20), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("trigger_timestamp", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("trigger_vital", sa.String(20), nullable=True),
        sa.Column("trigger_value", sa.Float(), nullable=True),
        sa.Column("baseline_value", sa.Float(), nullable=True),
        sa.Column("deviation_sigma", sa.Float(), nullable=True),
        sa.Column("second_derivative", sa.Float(), nullable=True),
        sa.Column("motion_score", sa.Integer(), nullable=True),
        sa.Column("vitals_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("rag_clinical_brief", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("retrieved_chunk_ids", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("llm_generation_time_ms", sa.Integer(), nullable=True),
        sa.Column("priority_tier_assigned", sa.String(20), nullable=True),
        sa.Column("clinician_acknowledged", sa.Boolean(), server_default="false"),
        sa.Column("acknowledgement_timestamp", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("response_time_minutes", sa.Float(), nullable=True),
        sa.Column("alert_dismissed", sa.Boolean(), server_default="false"),
        sa.Column("patient_deteriorated_after_dismissal", sa.Boolean(), nullable=True),
        sa.Column("patient_state_before", sa.String(20), nullable=True),
        sa.Column("patient_state_after", sa.String(20), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("alert_events")
    op.drop_table("vitals_history")
    op.drop_table("patients")
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE;")
