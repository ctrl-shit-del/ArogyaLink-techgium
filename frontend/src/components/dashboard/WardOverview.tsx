import React, { useMemo } from 'react';
import { usePatientStore } from '../../store/patientStore';
import { Activity, AlertTriangle, ShieldCheck } from 'lucide-react';

export const WardOverview: React.FC = () => {
    const patients = usePatientStore((state) => Object.values(state.patients));

    const stats = useMemo(() => {
        let synera = 0;
        let watch = 0;
        let stable = 0;

        patients.forEach(p => {
            if (p.tier === 'SYNERA_STATE') synera++;
            else if (p.tier === 'WATCH') watch++;
            else if (p.tier === 'STABLE') stable++;
        });

        return { synera, watch, stable };
    }, [patients]);

    return (
        <div className="w-full bg-[#1E293B] border-b border-[#334155] p-4 flex items-center justify-between shrink-0 h-20">
            <div className="flex flex-col">
                <h1 className="text-white text-xl font-bold tracking-tight">ArogyaLink</h1>
                <span className="text-[#94A3B8] text-sm">Primary Health Centre Ward</span>
            </div>

            <div className="flex gap-4">
                {/* Synera State */}
                <div className="bg-[#DC2626]/10 border border-[#DC2626]/30 rounded-lg px-4 py-2 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-[#DC2626]/20 flex items-center justify-center text-[#DC2626]">
                        <AlertTriangle size={20} />
                    </div>
                    <div className="flex flex-col">
                        <span className="text-3xl font-bold text-[#F87171] leading-none">{stats.synera}</span>
                        <span className="text-[#94A3B8] text-xs uppercase font-semibold">Critical</span>
                    </div>
                </div>

                {/* Watch State */}
                <div className="bg-[#D97706]/10 border border-[#D97706]/30 rounded-lg px-4 py-2 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-[#D97706]/20 flex items-center justify-center text-[#D97706]">
                        <Activity size={20} />
                    </div>
                    <div className="flex flex-col">
                        <span className="text-3xl font-bold text-[#FBBF24] leading-none">{stats.watch}</span>
                        <span className="text-[#94A3B8] text-xs uppercase font-semibold">Watch</span>
                    </div>
                </div>

                {/* Stable State */}
                <div className="bg-[#16A34A]/10 border border-[#16A34A]/30 rounded-lg px-4 py-2 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-[#16A34A]/20 flex items-center justify-center text-[#16A34A]">
                        <ShieldCheck size={20} />
                    </div>
                    <div className="flex flex-col">
                        <span className="text-3xl font-bold text-[#4ADE80] leading-none">{stats.stable}</span>
                        <span className="text-[#94A3B8] text-xs uppercase font-semibold">Stable</span>
                    </div>
                </div>
            </div>
        </div>
    );
};
