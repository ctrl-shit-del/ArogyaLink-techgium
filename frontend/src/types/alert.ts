export interface ClinicalBrief {
    differential_diagnosis: {
        condition: string;
        likelihood: "Likely" | "Possible" | "Rule Out" | string;
        reasoning: string;
    }[];
    recommended_actions: {
        priority: string;
        action: string;
        rationale: string;
    }[];
    drug_interaction_flags: {
        medication: string;
        flag: string;
        severity: "Info" | "Warning" | "Critical" | string;
    }[];
    sources: {
        document: string;
        section: string;
    }[];
}

export interface Alert {
    alert_id: string;
    patient_id: string;
    patient_name: string;
    trigger_vital: string;
    trigger_value: number;
    deviation_sigma: number;
    reconstruction_error: number;
    drl_priority: "ELEVATED" | "URGENT" | "IMMEDIATE";
    clinical_brief: ClinicalBrief;
    timestamp: string;
    acknowledged: boolean;
}
