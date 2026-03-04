import React from 'react';
import { ClinicalBrief } from '../../types/alert';

export interface AlertBriefProps {
    brief: ClinicalBrief;
}

export const AlertBrief: React.FC<AlertBriefProps> = ({ brief }) => {
    return (
        <div className="flex flex-col gap-6 w-full text-sm">
            {/* Differential Diagnosis */}
            <div>
                <h4 className="text-[#94A3B8] font-semibold mb-2 uppercase tracking-wider text-xs">Differential Diagnosis</h4>
                <div className="flex flex-wrap gap-2">
                    {brief.differential_diagnosis.map((dx, idx) => {
                        let colorClass = 'bg-[#334155] text-white';
                        if (dx.likelihood === 'Likely') colorClass = 'bg-[#DC2626] text-white';
                        if (dx.likelihood === 'Possible') colorClass = 'bg-[#D97706] text-white';

                        return (
                            <div key={idx} className={`px-2 py-1 rounded text-xs leading-none flex items-center gap-1 ${colorClass}`} title={dx.reasoning}>
                                <span className="font-medium">{dx.condition}</span>
                                <span className="opacity-75 relative top-[0.5px]">- {dx.likelihood}</span>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Recommended Actions */}
            <div>
                <h4 className="text-[#94A3B8] font-semibold mb-2 uppercase tracking-wider text-xs">Recommended Actions</h4>
                <ol className="list-decimal pl-4 space-y-2 text-[#E2E8F0]">
                    {brief.recommended_actions.map((action, idx) => {
                        let badgeColor = 'bg-[#334155]';
                        if (action.priority === 'High') badgeColor = 'bg-[#DC2626]';
                        if (action.priority === 'Medium') badgeColor = 'bg-[#D97706]';

                        return (
                            <li key={idx} className="pl-1">
                                <div className="flex flex-col gap-0.5">
                                    <div className="flex items-start gap-2">
                                        <span className="font-medium">{action.action}</span>
                                        <span className={`text-[10px] px-1.5 py-0.5 rounded leading-none mt-0.5 ${badgeColor}`}>{action.priority}</span>
                                    </div>
                                    <p className="text-[#94A3B8] text-xs">{action.rationale}</p>
                                </div>
                            </li>
                        );
                    })}
                </ol>
            </div>

            {/* Drug Interactions */}
            {brief.drug_interaction_flags && brief.drug_interaction_flags.length > 0 && (
                <div>
                    <h4 className="text-[#94A3B8] font-semibold mb-2 uppercase tracking-wider text-xs">Drug Interactions</h4>
                    <div className="flex flex-col gap-1">
                        {brief.drug_interaction_flags.map((flag, idx) => {
                            if (flag.severity === 'Info') return null; // Only show warnings/critical

                            let alertClass = 'bg-[#D97706]/20 border-[#D97706]/30 text-[#FBBF24]';
                            if (flag.severity === 'Critical') alertClass = 'bg-[#DC2626]/20 border-[#DC2626]/30 text-[#F87171]';

                            return (
                                <div key={idx} className={`p-2 rounded border text-xs flex gap-2 ${alertClass}`}>
                                    <span className="font-bold">{flag.medication}:</span>
                                    <span>{flag.flag}</span>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Sources */}
            <div className="mt-auto pt-4 border-t border-[#334155]">
                <h4 className="text-[#6B7280] font-semibold mb-2 uppercase tracking-wider text-[10px]">Sources</h4>
                <div className="flex flex-wrap gap-1">
                    {brief.sources.map((source, idx) => (
                        <div key={idx} className="bg-[#1E293B] border border-[#334155] text-[#94A3B8] px-1.5 py-0.5 rounded text-[10px]">
                            {source.document} ({source.section})
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
