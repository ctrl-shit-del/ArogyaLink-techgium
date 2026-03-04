import React, { useState } from 'react';
import { useAlertStore } from '../../store/alertStore';

export interface AcknowledgeButtonProps {
    alertId: string;
}

export const AcknowledgeButton: React.FC<AcknowledgeButtonProps> = ({ alertId }) => {
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);
    const acknowledge = useAlertStore((state) => state.acknowledge);
    const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

    const handleAcknowledge = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${apiBase}/api/v1/alerts/${alertId}/acknowledge`, {
                method: 'POST',
            });
            if (response.ok) {
                acknowledge(alertId);
                setSuccess(true);
            } else {
                console.error('Failed to acknowledge alert');
            }
        } catch (error) {
            console.error('Error acknowledging alert:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <button
            onClick={handleAcknowledge}
            disabled={loading || success}
            className={`w-full py-3 px-4 rounded-md font-medium text-sm transition-colors ${success
                    ? 'bg-[#16A34A] text-white cursor-default'
                    : 'bg-[#DC2626] hover:bg-[#B91C1C] text-white cursor-pointer active:scale-[0.98]'
                }`}
        >
            {loading ? 'Acknowledging...' : success ? 'Acknowledged ✓' : 'Acknowledge Alert'}
        </button>
    );
};
