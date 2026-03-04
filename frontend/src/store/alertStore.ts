import { create } from 'zustand';
import { Alert } from '../types/alert';

interface AlertStore {
    alerts: Alert[];
    addAlert: (alert: Alert) => void;
    setAlerts: (alerts: Alert[]) => void;
    acknowledge: (id: string) => void;
}

export const useAlertStore = create<AlertStore>((set) => ({
    alerts: [],
    setAlerts: (alerts) => set({ alerts }),
    addAlert: (alert) =>
        set((state) => ({
            alerts: [alert, ...state.alerts],
        })),
    acknowledge: (id) =>
        set((state) => ({
            alerts: state.alerts.map((alert) =>
                alert.alert_id === id ? { ...alert, acknowledged: true } : alert
            ),
        })),
}));
