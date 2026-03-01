"""Alert event ORM — stores RAG brief and DRL signals."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Boolean, Float, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from backend.config.database import Base


class AlertEventORM(Base):
    __tablename__ = "alert_events"

    alert_id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    patient_id: Mapped[str] = mapped_column(String(20), ForeignKey("patients.patient_id"), nullable=False)
    trigger_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    trigger_vital: Mapped[str | None] = mapped_column(String(20), nullable=True)
    trigger_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    deviation_sigma: Mapped[float | None] = mapped_column(Float, nullable=True)
    second_derivative: Mapped[float | None] = mapped_column(Float, nullable=True)
    motion_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vitals_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    rag_clinical_brief: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    retrieved_chunk_ids: Mapped[list | None] = mapped_column(ARRAY(Text), nullable=True)
    llm_generation_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    priority_tier_assigned: Mapped[str | None] = mapped_column(String(20), nullable=True)
    clinician_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledgement_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    response_time_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    alert_dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    patient_deteriorated_after_dismissal: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    patient_state_before: Mapped[str | None] = mapped_column(String(20), nullable=True)
    patient_state_after: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
