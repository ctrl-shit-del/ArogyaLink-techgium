"""Patient (MedID) ORM model."""
from datetime import date, datetime
from sqlalchemy import Boolean, Date, Float, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.config.database import Base


class PatientORM(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(10), nullable=True)
    blood_group: Mapped[str | None] = mapped_column(String(5), nullable=True)
    abha_id: Mapped[str | None] = mapped_column(String(50), unique=True, nullable=True)
    primary_language: Mapped[str] = mapped_column(String(30), default="Hindi")
    ward: Mapped[str | None] = mapped_column(String(50), nullable=True)
    bed_number: Mapped[str | None] = mapped_column(String(10), nullable=True)
    admission_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    attending_clinician: Mapped[str | None] = mapped_column(String(100), nullable=True)

    diagnosed_conditions: Mapped[dict] = mapped_column(JSON, default=list)
    past_surgeries: Mapped[dict] = mapped_column(JSON, default=list)
    hospitalisation_history: Mapped[dict] = mapped_column(JSON, default=list)
    family_history: Mapped[dict] = mapped_column(JSON, default=list)

    current_medications: Mapped[dict] = mapped_column(JSON, default=list)
    known_allergies: Mapped[dict] = mapped_column(JSON, default=list)
    adverse_reactions: Mapped[dict] = mapped_column(JSON, default=list)

    baseline_hr_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_hr_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_spo2_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_spo2_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_temp_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_bp_sys_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_bp_dia_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_last_updated: Mapped[datetime | None] = mapped_column(nullable=True)
    calibration_complete: Mapped[bool] = mapped_column(Boolean, default=False)

    genomic_risk_cardiac: Mapped[str] = mapped_column(String(10), default="Unknown")
    genomic_risk_respiratory: Mapped[str] = mapped_column(String(10), default="Unknown")
    genomic_risk_sepsis: Mapped[str] = mapped_column(String(10), default="Unknown")
    genomic_risk_diabetic: Mapped[str] = mapped_column(String(10), default="Unknown")

    last_clinical_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)
