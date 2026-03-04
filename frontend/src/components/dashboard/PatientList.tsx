import React, { useMemo } from 'react';
import { usePatientStore } from '../../store/patientStore';
import { useUiStore } from '../../store/uiStore';
import { PatientCard } from './PatientCard';
import { Patient, Tier } from '../../types/patient';

const TIER_ORDER: Record<Tier, number> = {
    SYNERA_STATE: 1,
    WATCH: 2,
    EXERTION: 3,
    STABLE: 4,
    ARTIFACT: 5,
};

export const PatientList: React.FC = () => {
    const patients = usePatientStore((state) => Object.values(state.patients));
    const selectedPatientId = useUiStore((state) => state.selectedPatientId);

    // Sort patients
    const sortedPatients = useMemo(() => {
        return [...patients].sort((a, b) => {
            // 1. Sort by tier
            const tierDiff = TIER_ORDER[a.tier] - TIER_ORDER[b.tier];
            if (tierDiff !== 0) return tierDiff;

            // 2. Sort by deviation sigma if available
            const sigmaA = a.latest?.deviation_sigma || 0;
            const sigmaB = b.latest?.deviation_sigma || 0;
            if (sigmaA !== sigmaB) return sigmaB - sigmaA; // Descending

            // 3. Fallback to name
            return a.name.localeCompare(b.name);
        });
    }, [patients]);

    if (patients.length === 0) {
        return (
            <div className="h-full flex items-center justify-center p-6 text-[#94A3B8] text-sm text-center border-r border-[#334155]">
                No patients loaded or connecting to stream...
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col border-r border-[#334155] bg-[#0F172A]">
            <div className="p-4 border-b border-[#334155] shrink-0 bg-[#1E293B]">
                <h2 className="text-white font-semibold flex items-center justify-between">
                    <span>Active Patient List</span>
                    <span className="text-xs bg-[#334155] text-[#E2E8F0] px-2 py-1 rounded-full">{patients.length} Total</span>
                </h2>
            </div>

            <div className="flex-1 overflow-y-auto p-3 scrollbar-thin scrollbar-thumb-[#334155] scrollbar-track-transparent">
                {sortedPatients.map((patient) => (
                    <PatientCard
                        key={patient.patient_id}
                        patient={patient}
                        isSelected={selectedPatientId === patient.patient_id}
                    />
                ))}
            </div>
        </div>
    );
};
