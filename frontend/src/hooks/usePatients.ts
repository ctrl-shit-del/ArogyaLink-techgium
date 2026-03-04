import { useEffect } from 'react';
import { usePatientStore } from '../store/patientStore';
import { Patient } from '../types/patient';

export const usePatients = () => {
    const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
    const setPatients = usePatientStore((state) => state.setPatients);

    useEffect(() => {
        let intervalId: number;

        const fetchPatients = async () => {
            try {
                const response = await fetch(`${apiBase}/api/v1/patients/`);
                if (response.ok) {
                    const basicPatients = await response.json();

                    const detailedPatients: Patient[] = await Promise.all(
                        basicPatients.map(async (p: any): Promise<Patient> => {
                            const detailRes = await fetch(`${apiBase}/api/v1/patients/${p.patient_id}`);
                            const detail = await detailRes.json();

                            const vitalsRes = await fetch(`${apiBase}/api/v1/patients/${p.patient_id}/vitals?limit=1`);
                            const vitalsHist = await vitalsRes.json();
                            const latestVitals = vitalsHist.length > 0 ? vitalsHist[0] : null;

                            // Fetch the authoritative live state from the engine's state manager
                            // This ensures tier is correct after browser refresh even if no new WS events fired
                            let initialTier: Patient['tier'] = 'STABLE';
                            try {
                                const stateRes = await fetch(`${apiBase}/api/v1/patients/${p.patient_id}/state`);
                                if (stateRes.ok) {
                                    const stateData = await stateRes.json();
                                    // Map backend state values to frontend Tier type
                                    const stateMap: Record<string, Patient['tier']> = {
                                        'SYNERA_STATE': 'SYNERA_STATE',
                                        'WATCH': 'WATCH',
                                        'EXERTION': 'EXERTION',
                                        'STABLE': 'STABLE',
                                        'ARTIFACT': 'ARTIFACT',
                                    };
                                    initialTier = stateMap[stateData.state] ?? 'STABLE';
                                }
                            } catch (_) { /* fallback to STABLE */ }

                            let age = 0;
                            if (detail.dob) {
                                const diff = Date.now() - new Date(detail.dob).getTime();
                                age = Math.abs(new Date(diff).getUTCFullYear() - 1970);
                            }

                            return {
                                patient_id: detail.patient_id,
                                name: detail.name,
                                age: age,
                                gender: detail.gender || 'Unknown',
                                ward: detail.ward || p.ward,
                                bed: detail.bed_number || p.bed_number,
                                tier: initialTier,
                                baseline_hr: detail.baseline_hr_mean || 80,
                                baseline_mse_mean: detail.baseline_mse_mean || 0.0005,
                                baseline_mse_std: detail.baseline_mse_std || 0.0001,
                                latest: latestVitals,
                                diagnosed_conditions: detail.diagnosed_conditions || [],
                                current_medications: detail.current_medications || [],
                                genomic_risk_cardiac: 'Unknown',
                                genomic_risk_sepsis: 'Unknown',
                                genomic_risk_respiratory: 'Unknown',
                            };
                        })
                    );
                    setPatients(detailedPatients);
                } else {
                    console.error('Failed to fetch patients:', response.statusText);
                }
            } catch (error) {
                console.error('Error fetching patients:', error);
            }
        };

        fetchPatients();
        intervalId = window.setInterval(fetchPatients, 15000);

        return () => clearInterval(intervalId);
    }, [apiBase, setPatients]);
};
