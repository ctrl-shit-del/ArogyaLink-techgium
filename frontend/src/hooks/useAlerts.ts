import { useEffect } from 'react';
import { useAlertStore } from '../store/alertStore';
import { Alert } from '../types/alert';

export const useAlerts = () => {
    const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
    const setAlerts = useAlertStore((state) => state.setAlerts);

    useEffect(() => {
        const fetchAlerts = async () => {
            try {
                const response = await fetch(`${apiBase}/api/v1/alerts/?limit=50`);
                if (response.ok) {
                    const data: Alert[] = await response.json();
                    setAlerts(data);
                } else {
                    console.error('Failed to fetch alerts:', response.statusText);
                }
            } catch (error) {
                console.error('Error fetching alerts:', error);
            }
        };

        fetchAlerts();
    }, [apiBase, setAlerts]);
};
