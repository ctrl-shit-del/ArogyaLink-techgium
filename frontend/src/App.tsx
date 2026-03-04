import React, { useEffect } from 'react';
import { Dashboard } from './pages/Dashboard';
import { useWebSocket } from './hooks/useWebSocket';
import { usePatients } from './hooks/usePatients';
import { useAlerts } from './hooks/useAlerts';

function App() {
    // Initialize data fetching and websocket hooks at the root level
    usePatients();
    useAlerts();
    useWebSocket();

    return (
        <div className="min-h-screen bg-[#0F172A] text-[#F8FAFC] font-sans">
            <Dashboard />
        </div>
    );
}

export default App;
