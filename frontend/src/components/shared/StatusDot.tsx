import React from 'react';
import { Tier } from '../../types/patient';

export interface StatusDotProps {
    tier: Tier;
    className?: string;
    size?: 'sm' | 'md' | 'lg';
}

const TIER_COLORS: Record<Tier, string> = {
    SYNERA_STATE: 'bg-[#DC2626]',
    WATCH: 'bg-[#D97706]',
    EXERTION: 'bg-[#2563EB]',
    STABLE: 'bg-[#16A34A]',
    ARTIFACT: 'bg-[#6B7280]',
};

export const StatusDot: React.FC<StatusDotProps> = ({ tier, className = '', size = 'md' }) => {
    const isSynera = tier === 'SYNERA_STATE';
    const colorClass = TIER_COLORS[tier] || TIER_COLORS.ARTIFACT;

    const sizeClass = {
        sm: 'w-2 h-2',
        md: 'w-2.5 h-2.5',
        lg: 'w-3 h-3',
    }[size];

    return (
        <div className={`relative ${sizeClass} ${className}`}>
            {isSynera && (
                <span className={`absolute inset-0 rounded-full ${colorClass} opacity-75 animate-ping`} />
            )}
            <span className={`relative inline-flex rounded-full w-full h-full ${colorClass}`} />
        </div>
    );
};
