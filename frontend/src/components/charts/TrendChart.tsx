import React, { useEffect, useState } from 'react';
import {
    ComposedChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import { usePatientStore } from '../../store/patientStore';

export interface TrendChartProps {
    patientId: string;
}

export const TrendChart: React.FC<TrendChartProps> = ({ patientId }) => {
    const [data, setData] = useState<any[]>([]);
    const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
    const patient = usePatientStore((state) => state.patients[patientId]);
    const baselineMseMean = patient?.baseline_mse_mean || 0.0005;

    useEffect(() => {
        const fetchVitals = async () => {
            try {
                const response = await fetch(`${apiBase}/api/v1/patients/${patientId}/vitals?limit=100`);
                if (response.ok) {
                    const vitals = await response.json();
                    // Transform if needed; assuming array is ordered oldest to newest, else reverse it.
                    // Vitals might come newest first based on 'limit=100' so we might need to reverse.
                    const chartData = (vitals.reverse ? vitals.reverse() : vitals).map((v: any) => {
                        const date = new Date(v.recorded_at);
                        return {
                            time: `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}`,
                            heart_rate: v.heart_rate,
                            reconstruction_error: v.reconstruction_error,
                            is_synera_state: v.deviation_sigma > 2, // Assuming threshold loosely mapped for the reference line
                            ...v,
                        };
                    });
                    setData(chartData);
                }
            } catch (error) {
                console.error('Failed to fetch trend chart data:', error);
            }
        };

        if (patientId) {
            fetchVitals();
            const interval = setInterval(fetchVitals, 5000);
            return () => clearInterval(interval);
        }
    }, [patientId, apiBase]);

    const calibrationReadings = data.slice(0, 15);
    const baselineMseArr = calibrationReadings
        .filter((r) => r.reconstruction_error != null)
        .map((r) => r.reconstruction_error);

    const dynamicBaselineMean = baselineMseArr.length > 0
        ? baselineMseArr.reduce((a, b) => a + b, 0) / baselineMseArr.length
        : baselineMseMean;

    // Compute std of calibration window so the threshold is always above personal normal
    const dynamicBaselineStd = baselineMseArr.length > 1
        ? Math.sqrt(baselineMseArr.map(x => (x - dynamicBaselineMean) ** 2).reduce((a, b) => a + b, 0) / baselineMseArr.length)
        : (patient?.baseline_mse_std || 0.0001);

    // Per-patient Synera threshold: mean + 2.5σ (same formula as rule engine)
    const syneraThreshold = dynamicBaselineMean + 2.5 * dynamicBaselineStd;

    const mseValues = data
        .map((r) => r.reconstruction_error)
        .filter(Boolean);
    const maxMse = Math.max(...mseValues, syneraThreshold * 3);
    const mseDomain = [0, Math.ceil(maxMse * 1.2 * 10000) / 10000];

    const displayData = data.slice(-20);

    return (
        <div className="w-full h-[450px] bg-[#1E293B] rounded-lg p-4 border border-[#334155] flex flex-col gap-4">
            {/* Top Chart: Heart Rate */}
            <ResponsiveContainer width="100%" height="50%">
                <ComposedChart data={displayData} syncId="patientTrend">
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="time" stroke="#94A3B8" fontSize={12} tickLine={false} axisLine={false} hide={true} />
                    <YAxis
                        yAxisId="left"
                        domain={[60, 160]}
                        stroke="#94A3B8"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                    />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', color: '#F8FAFC' }}
                        itemStyle={{ color: '#E2E8F0' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px', color: '#94A3B8' }} verticalAlign="top" align="right" />

                    <ReferenceLine y={120} yAxisId="left" stroke="#6B7280" strokeDasharray="3 3" label={{ position: 'top', value: 'Static threshold (120)', fill: '#6B7280', fontSize: 10 }} />

                    <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="heart_rate"
                        stroke="#2563EB"
                        name="Heart Rate"
                        dot={false}
                        strokeWidth={2}
                    />
                </ComposedChart>
            </ResponsiveContainer>

            {/* Bottom Chart: MSE Error */}
            <ResponsiveContainer width="100%" height="50%">
                <ComposedChart data={displayData} syncId="patientTrend">
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="time" stroke="#94A3B8" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis
                        yAxisId="right"
                        orientation="left"
                        domain={mseDomain}
                        allowDataOverflow={true}
                        stroke="#94A3B8"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(val) => val.toFixed(4)}
                    />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', color: '#F8FAFC' }}
                        itemStyle={{ color: '#E2E8F0' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px', color: '#94A3B8' }} verticalAlign="top" align="right" />

                    <ReferenceLine y={syneraThreshold} yAxisId="right" stroke="#DC2626" strokeDasharray="3 3" label={{ position: 'top', value: `Synera threshold (${syneraThreshold.toFixed(4)})`, fill: '#DC2626', fontSize: 10 }} />
                    <ReferenceLine
                        yAxisId="right"
                        y={dynamicBaselineMean}
                        stroke="#16A34A"
                        strokeDasharray="2 2"
                        label={{ value: `Personal normal: ${dynamicBaselineMean.toFixed(4)}`, fill: "#16A34A", fontSize: 10 }}
                    />

                    {displayData.map((entry, index) => {
                        if (entry.is_synera_state && (!displayData[index - 1] || !displayData[index - 1].is_synera_state)) {
                            return (
                                <ReferenceLine
                                    key={`event-${index}`}
                                    x={entry.time}
                                    stroke="#DC2626"
                                    strokeDasharray="3 3"
                                />
                            );
                        }
                        return null;
                    })}

                    <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="reconstruction_error"
                        stroke="#DC2626"
                        name="MSE error"
                        dot={false}
                        strokeWidth={2}
                    />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
};
