import React from 'react';
import { useAlertStore } from '../../store/alertStore';
import { usePatientStore } from '../../store/patientStore';
import { AlertBrief } from './AlertBrief';
import { AcknowledgeButton } from './AcknowledgeButton';

export interface AlertCardProps {
    alertId: string;
}

export const AlertCard: React.FC<AlertCardProps> = ({ alertId }) => {
    const alert = useAlertStore((state) => state.alerts.find(a => a.alert_id === alertId));
    const patient = usePatientStore((state) => state.patients[alert?.patient_id || '']);

    if (!alert) return null;

    return (
        <div className="w-full h-full bg-[#1E293B] border-2 border-[#DC2626] rounded-xl overflow-hidden flex flex-col shadow-2xl relative animate-pulse-border">

            {/* Header spanning full width */}
            <div className="bg-[#DC2626]/10 p-4 border-b border-[#DC2626]/20 flex justify-between items-center shrink-0">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-[#DC2626] rounded-full flex items-center justify-center relative">
                        <div className="absolute inset-0 rounded-full border-2 border-[#DC2626] animate-ping opacity-50"></div>
                        <span className="text-white font-bold text-lg">!</span>
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-white leading-none mb-1">{alert.patient_name}</h2>
                        <div className="flex gap-2 text-sm text-[#F87171] font-medium">
                            <span>SYNERA STATE</span>
                            <span>•</span>
                            <span>{alert.drl_priority} PRIORITY</span>
                            {patient && (
                                <>
                                    <span>•</span>
                                    <span className="text-[#94A3B8]">Ward {patient.ward}, Bed {patient.bed}</span>
                                </>
                            )}
                        </div>
                    </div>
                </div>

                <div className="text-right">
                    <div className="text-[#94A3B8] text-sm">Time Triggered</div>
                    <div className="text-white font-mono">{new Date(alert.timestamp).toLocaleTimeString()}</div>
                </div>
            </div>

            {/* Main Content Split */}
            <div className="flex-1 flex min-h-0 overflow-hidden">

                {/* Left Panel: Vitals & Metrics */}
                <div className="w-1/3 p-6 border-r border-[#334155] flex flex-col gap-6 overflow-y-auto shrink-0 bg-[#0F172A]/30">

                    <div>
                        <h3 className="text-[#94A3B8] uppercase text-xs font-semibold tracking-wider mb-3">Trigger Metrics</h3>
                        <div className="bg-[#DC2626]/20 border border-[#DC2626]/30 rounded-lg p-4">
                            <div className="text-[#F87171] text-sm mb-1">{alert.trigger_vital} Deviation</div>
                            <div className="text-3xl font-bold text-white font-mono">{alert.deviation_sigma.toFixed(2)}σ</div>
                            <div className="text-[#94A3B8] text-xs mt-2">MSE: {alert.reconstruction_error.toFixed(4)}</div>
                        </div>
                    </div>

                    {patient?.latest && (
                        <div>
                            <h3 className="text-[#94A3B8] uppercase text-xs font-semibold tracking-wider mb-3">Current Vitals</h3>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-[#1E293B] rounded p-3 border border-[#334155]">
                                    <div className="text-[#94A3B8] text-xs">Heart Rate</div>
                                    <div className="text-xl font-mono text-white">{patient.latest.heart_rate} <span className="text-xs text-[#6B7280]">bpm</span></div>
                                </div>
                                <div className="bg-[#1E293B] rounded p-3 border border-[#334155]">
                                    <div className="text-[#94A3B8] text-xs">SpO2</div>
                                    <div className="text-xl font-mono text-white">{patient.latest.spo2} <span className="text-xs text-[#6B7280]">%</span></div>
                                </div>
                                <div className="bg-[#1E293B] rounded p-3 border border-[#334155]">
                                    <div className="text-[#94A3B8] text-xs">Temp</div>
                                    <div className="text-xl font-mono text-white">{patient.latest.temperature} <span className="text-xs text-[#6B7280]">°C</span></div>
                                </div>
                                <div className="bg-[#1E293B] rounded p-3 border border-[#334155]">
                                    <div className="text-[#94A3B8] text-xs">Motion</div>
                                    <div className="text-xl font-mono text-white">{patient.latest.motion_score.toFixed(1)}</div>
                                </div>
                            </div>
                        </div>
                    )}

                </div>

                {/* Right Panel: Clinical Brief */}
                <div className="w-2/3 p-6 overflow-y-auto">
                    <AlertBrief brief={alert.clinical_brief} />
                </div>
            </div>

            {/* Footer / Action */}
            <div className="shrink-0 p-4 border-t border-[#334155] bg-[#0F172A]/50">
                <AcknowledgeButton alertId={alertId} />
            </div>

            <style>{`
        @keyframes pulse-border {
          0% { border-color: rgba(220, 38, 38, 0.4); box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.2); }
          50% { border-color: rgba(220, 38, 38, 1); box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
          100% { border-color: rgba(220, 38, 38, 0.4); box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }
        .animate-pulse-border {
          animation: pulse-border 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
      `}</style>
        </div>
    );
};
