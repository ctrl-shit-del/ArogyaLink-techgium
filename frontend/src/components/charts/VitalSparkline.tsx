import React, { useEffect, useState } from 'react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';

export interface VitalSparklineProps {
    patientId: string;
    isSynera: boolean;
}

export const VitalSparkline: React.FC<VitalSparklineProps> = ({ patientId, isSynera }) => {
    const [data, setData] = useState<{ hr: number }[]>([]);
    const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

    useEffect(() => {
        let mounted = true;
        const fetchLatestVitals = async () => {
            try {
                const response = await fetch(`${apiBase}/api/v1/patients/${patientId}/vitals?limit=10`);
                if (response.ok && mounted) {
                    const vitals = await response.json();
                    const hrData = (vitals.reverse ? vitals.reverse() : vitals).map((v: any) => ({ hr: v.heart_rate }));
                    setData(hrData);
                }
            } catch (error) {
                console.error('Sparkline fetch error:', error);
            }
        };

        if (patientId) {
            fetchLatestVitals();
            const interval = setInterval(fetchLatestVitals, 15000); // Check occasionally
            return () => {
                mounted = false;
                clearInterval(interval);
            };
        }
    }, [patientId, apiBase]);

    const color = isSynera ? '#DC2626' : '#16A34A';

    if (!data.length) {
        return <div className="w-[80px] h-[30px] bg-[#1E293B] animate-pulse rounded"></div>;
    }

    return (
        <div style={{ width: 80, height: 30 }}>
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <YAxis domain={['dataMin - 5', 'dataMax + 5']} hide />
                    <Line
                        type="monotone"
                        dataKey="hr"
                        stroke={color}
                        strokeWidth={1.5}
                        dot={false}
                        isAnimationActive={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};
