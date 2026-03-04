import { create } from 'zustand';

interface UiStore {
    activeAlertId: string | null;
    selectedPatientId: string | null;
    setActiveAlert: (id: string | null) => void;
    setSelectedPatient: (id: string | null) => void;
}

export const useUiStore = create<UiStore>((set) => ({
    activeAlertId: null,
    selectedPatientId: null,
    setActiveAlert: (id) => set({ activeAlertId: id }),
    setSelectedPatient: (id) => set({ selectedPatientId: id }),
}));
