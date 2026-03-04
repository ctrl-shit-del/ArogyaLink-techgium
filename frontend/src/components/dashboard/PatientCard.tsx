import React from 'react';
import { Patient } from '../../types/patient';
import { StatusDot } from '../shared/StatusDot';
import { VitalSparkline } from '../charts/VitalSparkline';
import { useUiStore } from '../../store/uiStore';
import { AlertTriangle } from 'lucide-react';

export interface PatientCardProps {
    patient: Patient;
    isSelected: boolean;
}

export const PatientCard: React.FC<PatientCardProps> = ({ patient, isSelected }) => {
    const setSelectedPatient = useUiStore((state) => state.setSelectedPatient);
    const isSynera = patient.tier === 'SYNERA_STATE';
    const isWatch = patient.tier === 'WATCH';
    const mseHigh = patient.latest?.reconstruction_error > 0.00168;

    // Helper for border color
    let borderLeftColor = 'border-l-[#6B7280]';
    if (isSynera) borderLeftColor = 'border-l-[#DC2626]';
    if (isWatch) borderLeftColor = 'border-l-[#D97706]';
    if (patient.tier === 'EXERTION') borderLeftColor = 'border-l-[#2563EB]';
    if (patient.tier === 'STABLE') borderLeftColor = 'border-l-[#16A34A]';

    return (
        <div
            onClick={() => setSelectedPatient(patient.patient_id)}
            className={`
        w-full bg-[#1E293B] rounded-lg border border-[#334155] p-3 mb-2 cursor-pointer 
        transition-all duration-200 hover:bg-[#334155]/50 flex items-center justify-between
        border-l-4 ${borderLeftColor}
        ${isSelected ? 'bg-[#334155]/80 ring-1 ring-[#94A3B8]' : ''}
      `}
        >
            {/* Patient Info */}
            <div className="flex items-center gap-3 w-1/3">
                <StatusDot tier={patient.tier} size="md" />
                <div className="flex flex-col">
                    <span className="text-white font-medium text-sm truncate">{patient.name}</span>
                    <span className="text-[#94A3B8] text-xs">Ward {patient.ward} • Bed {patient.bed}</span>
                </div>
            </div>

            {/* Quick Vitals */}
            {patient.latest && (
                <div className="flex gap-4 items-center w-1/3 justify-center">
                    <div className="flex flex-col items-center">
                        <span className="text-[#94A3B8] text-[10px] uppercase">HR</span>
                        <span className="text-white font-mono text-sm">{patient.latest.heart_rate}</span>
                    </div>
                    <div className="flex flex-col items-center">
                        <span className="text-[#94A3B8] text-[10px] uppercase">SpO2</span>
                        <span className="text-white font-mono text-sm">{patient.latest.spo2}%</span>
                    </div>
                    <div className="flex flex-col items-center">
                        <span className="text-[#94A3B8] text-[10px] uppercase">MSE</span>
                        <span className={`font-mono text-sm flex items-center gap-1 ${mseHigh ? 'text-[#DC2626] font-bold' : 'text-white'}`}>
                            {patient.latest.reconstruction_error.toFixed(4)}
                            {mseHigh && <AlertTriangle size={12} />}
                        </span>
                    </div>
                </div>
            )}

            {/* Sparkline */}
            <div className="w-1/3 flex justify-end">
                <VitalSparkline patientId={patient.patient_id} isSynera={isSynera} />
            </div>
        </div>
    );
};
