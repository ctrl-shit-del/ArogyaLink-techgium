"""Vitals history ORM — TimescaleDB hypertable."""
from datetime import datetime
from sqlalchemy import BigInteger, Boolean, Float, Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from backend.config.database import Base


class VitalsHistoryORM(Base):
    __tablename__ = "vitals_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(20), ForeignKey("patients.patient_id"), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(primary_key=True, nullable=False)
    heart_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    spo2: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    sys_bp_est: Mapped[float | None] = mapped_column(Float, nullable=True)
    dia_bp_est: Mapped[float | None] = mapped_column(Float, nullable=True)
    motion_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    battery_pct: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reconstruction_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    pre_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(20), default="wearable")
