import { useEffect, useRef } from 'react';
import { usePatientStore } from '../store/patientStore';
import { useAlertStore } from '../store/alertStore';

export const useWebSocket = () => {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
    const wsRef = useRef<WebSocket | null>(null);
    const updatePatient = usePatientStore((state) => state.updatePatient);
    const addAlert = useAlertStore((state) => state.addAlert);

    useEffect(() => {
        let reconnectTimer: number;

        const connect = () => {
            if (wsRef.current?.readyState === WebSocket.OPEN) return;

            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket connected');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    const type = data.event_type || data.type;
                    const payload = data.payload || data;

                    if (type === 'STATE_CHANGE') {
                        const tierMap: Record<string, 'SYNERA_STATE' | 'WATCH' | 'STABLE' | 'EXERTION' | 'ARTIFACT'> = {
                            'SYNERA_STATE': 'SYNERA_STATE',
                            'WATCH': 'WATCH',
                            'STABLE': 'STABLE',
                            'EXERTION': 'EXERTION',
                            'ARTIFACT': 'ARTIFACT',
                        };
                        updatePatient(payload.patient_id, {
                            tier: tierMap[payload.new_state] || 'STABLE'
                        });
                    }

                    if (type === 'SYNERA_STATE') {
                        updatePatient(payload.patient_id, {
                            tier: 'SYNERA_STATE'
                        });
                        addAlert(payload);
                    }
                } catch (error) {
                    console.error('WebSocket message parsing error:', error);
                }
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected. Reconnecting in 3s...');
                reconnectTimer = window.setTimeout(connect, 3000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                ws.close();
            };
        };

        connect();

        return () => {
            clearTimeout(reconnectTimer);
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [wsUrl, updatePatient, addAlert]);
};
