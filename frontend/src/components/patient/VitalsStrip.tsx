import React from 'react';
import { Patient } from '../../types/patient';

export interface VitalsStripProps {
    patient: Patient;
}

export const VitalsStrip: React.FC<VitalsStripProps> = ({ patient }) => {
    // Return early or show placeholder if we don't have latest vitals yet
    if (!patient.latest) {
        return (
            <div className="flex gap-2 mb-8 flex-wrap">
                <span className="text-[#94A3B8] text-sm italic">Waiting for telemetry...</span>
            </div>
        );
    }

    const vitals = [
        {
            label: "Heart Rate", value: patient.latest.heart_rate, unit: "BPM",
            low: 60, high: 100, icon: "❤️"
        },
        {
            label: "SpO2", value: patient.latest.spo2, unit: "%",
            low: 95, high: 100, icon: "🫁"
        },
        {
            label: "Temperature", value: patient.latest.temperature, unit: "°C",
            low: 36.5, high: 37.5, icon: "🌡️"
        },
        {
            label: "Motion", value: patient.latest.motion_score, unit: "",
            low: 0, high: 3, icon: "📱"
        },
        {
            label: "MSE", value: patient.latest.reconstruction_error.toFixed(4), unit: "",
            low: 0, high: 0.00168, icon: "🧠"
        },
    ];

    const getStatusColor = (v: any) => {
        const val = Number(v.value);
        if (isNaN(val)) return "bg-[#334155] border-[#475569] text-[#E2E8F0]"; // Fallback

        if (val < v.low || val > v.high) {
            // Check if it's wildly out of bounds or just slightly
            const range = v.high - v.low;
            const margin = range * 0.1; // 10% margin

            if (val < v.low - margin || val > v.high + margin) {
                // Red - Critical
                return "bg-[#DC2626]/20 border-[#DC2626]/50 text-[#FCA5A5]";
            }
            // Yellow - Warning (Borderline)
            return "bg-[#D97706]/20 border-[#D97706]/50 text-[#FCD34D]";
        }
        // Green - Normal
        return "bg-[#16A34A]/20 border-[#16A34A]/50 text-[#86EFAC]";
    };

    const getDotColor = (v: any) => {
        const val = Number(v.value);
        if (isNaN(val)) return "bg-[#94A3B8]"; // Fallback

        if (val < v.low || val > v.high) {
            const range = v.high - v.low;
            const margin = range * 0.1;

            if (val < v.low - margin || val > v.high + margin) {
                return "bg-[#EF4444]";
            }
            return "bg-[#F59E0B]";
        }
        return "bg-[#22C55E]";
    };

    return (
        <div className="flex gap-2 mb-8 flex-wrap">
            {vitals.map((v, i) => {
                const colors = getStatusColor(v);
                const dotColor = getDotColor(v);

                return (
                    <div key={i} className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${colors} shadow-sm backdrop-blur-sm`}>
                        <span className="text-sm">{v.icon}</span>
                        <span className="text-xs font-semibold uppercase tracking-wider opacity-80">{v.label}:</span>
                        <span className="text-sm font-bold font-mono">
                            {v.value} {v.unit}
                        </span>
                        <div className={`w-2 h-2 rounded-full ${dotColor} drop-shadow-md ml-1`} />
                    </div>
                );
            })}
        </div>
    );
};
