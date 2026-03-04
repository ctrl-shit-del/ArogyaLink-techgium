import React from 'react';
import { Patient } from '../../types/patient';
import { VitalsStrip } from './VitalsStrip';

export interface MedIDPanelProps {
    patient: Patient;
}

export const MedIDPanel: React.FC<MedIDPanelProps> = ({ patient }) => {

    const getRiskBadge = (risk: string) => {
        switch (risk.toUpperCase()) {
            case 'HIGH': return 'bg-[#DC2626] text-white';
            case 'MEDIUM': return 'bg-[#D97706] text-white';
            case 'LOW': return 'bg-[#16A34A] text-white';
            default: return 'bg-[#475569] text-gray-300';
        }
    };

    return (
        <div className="w-full h-full bg-[#1E293B] border-r border-[#334155] flex flex-col p-6 overflow-y-auto">

            {/* Patient Header */}
            <div className="flex items-center gap-4 mb-8">
                <div className="w-16 h-16 rounded-full bg-[#334155] text-white flex items-center justify-center text-2xl font-bold shrink-0">
                    {patient.name.charAt(0)}
                </div>
                <div>
                    <h2 className="text-2xl font-bold text-white leading-tight">{patient.name}</h2>
                    <div className="text-[#94A3B8] text-sm mt-1">
                        {patient.gender} • {patient.age} y/o • Ward {patient.ward}, Bed {patient.bed}
                    </div>
                    <div className="text-[#94A3B8] text-sm font-mono mt-1">
                        ABHA: {patient.patient_id.substring(0, 14)}...
                    </div>
                </div>
            </div>

            {/* Vitals Strip */}
            <VitalsStrip patient={patient} />

            {/* Conditions */}
            <div className="mb-8">
                <h3 className="text-[#94A3B8] uppercase text-xs font-semibold tracking-wider mb-3">Diagnosed Conditions</h3>
                {patient.diagnosed_conditions?.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                        {patient.diagnosed_conditions.map((c, i) => (
                            <span key={i} className="bg-[#0F172A] border border-[#334155] text-[#E2E8F0] px-2 py-1 rounded text-sm">
                                {c.name}
                            </span>
                        ))}
                    </div>
                ) : (
                    <div className="text-[#6B7280] italic text-sm">No recorded conditions</div>
                )}
            </div>

            {/* Medications */}
            <div className="mb-8">
                <h3 className="text-[#94A3B8] uppercase text-xs font-semibold tracking-wider mb-3">Current Medications</h3>
                {patient.current_medications?.length > 0 ? (
                    <div className="bg-[#0F172A] rounded border border-[#334155] overflow-hidden">
                        <table className="w-full text-left text-sm text-[#E2E8F0]">
                            <thead className="bg-[#1E293B] text-[#94A3B8]">
                                <tr>
                                    <th className="p-2 font-medium">Medication</th>
                                    <th className="p-2 font-medium">Dose</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-[#334155]">
                                {patient.current_medications.map((m, i) => (
                                    <tr key={i}>
                                        <td className="p-2">{m.name}</td>
                                        <td className="p-2 font-mono text-xs">{m.dose}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="text-[#6B7280] italic text-sm">No active prescriptions</div>
                )}
            </div>

            {/* Genomic Risks */}
            {[patient.genomic_risk_cardiac, patient.genomic_risk_respiratory, patient.genomic_risk_sepsis].some(r => r && r.toLowerCase() !== 'unknown') && (
                <div>
                    <h3 className="text-[#94A3B8] uppercase text-xs font-semibold tracking-wider mb-3">Genomic Risk Profiles</h3>
                    <div className="space-y-2">
                        <div className="flex justify-between items-center p-2 rounded bg-[#0F172A] border border-[#334155]">
                            <span className="text-sm text-[#E2E8F0]">Cardiac</span>
                            <span className={`text-[10px] px-2 py-0.5 rounded uppercase font-bold ${getRiskBadge(patient.genomic_risk_cardiac || 'Unknown')}`}>
                                {patient.genomic_risk_cardiac || 'Unknown'}
                            </span>
                        </div>
                        <div className="flex justify-between items-center p-2 rounded bg-[#0F172A] border border-[#334155]">
                            <span className="text-sm text-[#E2E8F0]">Respiratory</span>
                            <span className={`text-[10px] px-2 py-0.5 rounded uppercase font-bold ${getRiskBadge(patient.genomic_risk_respiratory || 'Unknown')}`}>
                                {patient.genomic_risk_respiratory || 'Unknown'}
                            </span>
                        </div>
                        <div className="flex justify-between items-center p-2 rounded bg-[#0F172A] border border-[#334155]">
                            <span className="text-sm text-[#E2E8F0]">Sepsis</span>
                            <span className={`text-[10px] px-2 py-0.5 rounded uppercase font-bold ${getRiskBadge(patient.genomic_risk_sepsis || 'Unknown')}`}>
                                {patient.genomic_risk_sepsis || 'Unknown'}
                            </span>
                        </div>
                    </div>
                </div>
            )}

        </div>
    );
};
