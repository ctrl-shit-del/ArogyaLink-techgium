import React from 'react';
import { WardOverview } from '../components/dashboard/WardOverview';
import { PatientList } from '../components/dashboard/PatientList';
import { useUiStore } from '../store/uiStore';
import { useAlertStore } from '../store/alertStore';
import { AlertCard } from '../components/alerts/AlertCard';
import { PatientDetail } from './PatientDetail';
import { ShieldCheck, Activity } from 'lucide-react';

export const Dashboard: React.FC = () => {
    const activeAlertId = useUiStore((state) => state.activeAlertId);
    const selectedPatientId = useUiStore((state) => state.selectedPatientId);
    const alerts = useAlertStore((state) => state.alerts);

    // Find an unacknowledged SYNERA_STATE alert to auto-display
    const activeCriticalAlert = React.useMemo(() => {
        return alerts.find(a => !a.acknowledged && a.deviation_sigma > 2);
    }, [alerts]);

    const displayAlertId = activeAlertId || activeCriticalAlert?.alert_id;

    return (
        <div className="flex flex-col h-screen w-full bg-[#0F172A] overflow-hidden">

            {/* Top Bar: Ward Overview (Always Visible) */}
            <WardOverview />

            {/* Main Layout Split */}
            <div className="flex-1 flex min-h-0">

                {/* Left: Patient List (1/3 width) */}
                <div className="w-[380px] shrink-0 h-full border-r border-[#334155] shadow-xl z-10 bg-[#0F172A]">
                    <PatientList />
                </div>

                {/* Right: Context-Dependent Main Area */}
                <div className="flex-1 h-full bg-[#0F172A] relative overflow-hidden">

                    {/* Overlay state: Critical Alert */}
                    {displayAlertId ? (
                        <div className="absolute inset-4 z-50">
                            <AlertCard alertId={displayAlertId} />
                        </div>
                    ) : selectedPatientId ? (
                        /* Selected Patient Detail View */
                        <div className="w-full h-full animate-in fade-in duration-300">
                            <PatientDetail patientId={selectedPatientId} />
                        </div>
                    ) : (
                        /* Empty State */
                        <div className="w-full h-full flex flex-col items-center justify-center text-[#6B7280] p-12 text-center animate-in fade-in duration-500">
                            <ShieldCheck size={64} className="mb-4 text-[#334155] opacity-50" />
                            <h2 className="text-xl font-medium text-[#94A3B8] mb-2">Ward Status Stable</h2>
                            <p className="max-w-md">No critical alerts requiring immediate attention. Select a patient from the list to view their detailed monitoring, medical history, and Synera risk profile.</p>
                        </div>
                    )}

                </div>
            </div>
        </div>
    );
};
