export type Tier = "SYNERA_STATE" | "WATCH" | "EXERTION" | "STABLE" | "ARTIFACT";

export interface Patient {
  patient_id: string;
  name: string;
  age: number;
  gender: string;
  ward: string;
  bed: string;
  tier: Tier;
  baseline_hr: number;
  baseline_mse_mean: number;
  baseline_mse_std: number;
  latest: {
    heart_rate: number;
    spo2: number;
    temperature: number;
    motion_score: number;
    reconstruction_error: number;
    deviation_sigma: number;
    recorded_at: string;
  };
  diagnosed_conditions: { name: string }[];
  current_medications: { name: string; dose: string }[];
  genomic_risk_cardiac: string;
  genomic_risk_sepsis: string;
  genomic_risk_respiratory: string;
}
