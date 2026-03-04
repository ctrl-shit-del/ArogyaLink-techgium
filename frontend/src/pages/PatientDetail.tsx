import React from 'react';
import { usePatientStore } from '../store/patientStore';
import { MedIDPanel } from '../components/patient/MedIDPanel';
import { TrendChart } from '../components/charts/TrendChart';
import { ClinicalQueryPanel } from '../components/patient/ClinicalQueryPanel';

export interface PatientDetailProps {
    patientId: string;
}

export const PatientDetail: React.FC<PatientDetailProps> = ({ patientId }) => {
    const patient = usePatientStore((state) => state.patients[patientId]);

    if (!patient) {
        return <div className="p-8 text-[#94A3B8]">Loading patient data...</div>;
    }

    return (
        <div className="flex w-full h-full overflow-hidden bg-[#0F172A]">
            {/* Sidebar: MedID */}
            <div className="w-1/3 max-w-sm h-full shrink-0">
                <MedIDPanel patient={patient} />
            </div>

            {/* Main Area: Charts & Vitals */}
            <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">

                {/* Vitals Summary Strip */}
                {patient.latest && (
                    <div className="flex gap-4">
                        <div className="flex-1 bg-[#1E293B] border border-[#334155] rounded-lg p-4 flex items-center justify-between">
                            <span className="text-[#94A3B8] font-medium uppercase text-sm">Heart Rate</span>
                            <span className="text-3xl font-mono text-white">{patient.latest.heart_rate} <span className="text-sm text-[#6B7280]">bpm</span></span>
                        </div>
                        <div className="flex-1 bg-[#1E293B] border border-[#334155] rounded-lg p-4 flex items-center justify-between">
                            <span className="text-[#94A3B8] font-medium uppercase text-sm">SpO2</span>
                            <span className="text-3xl font-mono text-white">{patient.latest.spo2} <span className="text-sm text-[#6B7280]">%</span></span>
                        </div>
                        <div className="flex-1 bg-[#1E293B] border border-[#334155] rounded-lg p-4 flex items-center justify-between">
                            <span className="text-[#94A3B8] font-medium uppercase text-sm">Temp</span>
                            <span className="text-3xl font-mono text-white">{patient.latest.temperature} <span className="text-sm text-[#6B7280]">°C</span></span>
                        </div>
                    </div>
                )}

                {/* Synera Trend Chart - MOST IMPORTANT VIZ */}
                <div>
                    <h3 className="text-white font-semibold mb-3">Synera Anomaly Detection & Heart Rate Trend</h3>
                    <TrendChart patientId={patient.patient_id} />
                </div>

                {/* Clinical Assistant — RAG Chatbot */}
                <div>
                    <h3 className="text-white font-semibold mb-3">Clinical Assistant</h3>
                    <ClinicalQueryPanel patient={patient} />
                </div>
            </div>
        </div>
    );
};
