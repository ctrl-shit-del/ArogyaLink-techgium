import { create } from 'zustand';
import { Patient } from '../types/patient';

interface PatientStore {
    patients: Record<string, Patient>;
    updatePatient: (id: string, patch: Partial<Patient>) => void;
    setPatients: (list: Patient[]) => void;
}

export const usePatientStore = create<PatientStore>((set) => ({
    patients: {},
    updatePatient: (id, patch) =>
        set((state) => ({
            patients: {
                ...state.patients,
                [id]: {
                    ...state.patients[id],
                    ...patch,
                },
            },
        })),
    setPatients: (list) => {
        set((state) => {
            const newPatients = { ...state.patients };
            list.forEach(patient => {
                const existing = newPatients[patient.patient_id];
                newPatients[patient.patient_id] = {
                    ...patient,
                    // Never overwrite a live WebSocket tier with the polling default 'STABLE'
                    tier: existing?.tier ?? patient.tier ?? 'STABLE',
                    // Always prefer the freshest vitals from the API
                    latest: patient.latest ?? existing?.latest,
                };
            });
            return { patients: newPatients };
        });
    },
}));
