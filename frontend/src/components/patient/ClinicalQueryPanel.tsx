import React, { useState, useRef, useEffect } from 'react';
import { Patient } from '../../types/patient';
import { AlertBrief } from '../alerts/AlertBrief';
import { Bot, Send, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { ClinicalBrief } from '../../types/alert';

interface QueryEntry {
    question: string;
    brief: ClinicalBrief | null;
    error: string | null;
    loading: boolean;
}

interface ClinicalQueryPanelProps {
    patient: Patient;
}

const SUGGESTED_QUERIES = [
    'What is the sepsis risk given current vitals?',
    'Are there any drug interactions to check?',
    'What does the MSE elevation indicate?',
    'What are the recommended immediate actions?',
];

export const ClinicalQueryPanel: React.FC<ClinicalQueryPanelProps> = ({ patient }) => {
    const [question, setQuestion] = useState('');
    const [history, setHistory] = useState<QueryEntry[]>([]);
    const [expanded, setExpanded] = useState<number | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const bottomRef = useRef<HTMLDivElement>(null);
    const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [history]);

    const submit = async (q: string) => {
        const queryText = q.trim();
        if (!queryText) return;

        const entry: QueryEntry = { question: queryText, brief: null, error: null, loading: true };
        setHistory(prev => [...prev, entry]);
        setQuestion('');
        setExpanded(null);

        try {
            const res = await fetch(`${apiBase}/api/v1/rag/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    patient_id: patient.patient_id,
                    question: queryText,
                    vitals_snapshot: patient.latest || {},
                }),
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            setHistory(prev => {
                const next = [...prev];
                const idx = next.length - 1;
                next[idx] = { ...next[idx], brief: data as ClinicalBrief, loading: false };
                return next;
            });
            setExpanded(history.length); // auto-expand latest
        } catch (err: any) {
            setHistory(prev => {
                const next = [...prev];
                const idx = next.length - 1;
                next[idx] = { ...next[idx], error: err.message || 'Query failed', loading: false };
                return next;
            });
        }
    };

    const onKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submit(question);
        }
    };

    return (
        <div className="w-full bg-[#1E293B] rounded-lg border border-[#334155] flex flex-col overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-2 px-4 py-3 border-b border-[#334155] shrink-0">
                <Bot size={16} className="text-[#2563EB]" />
                <span className="text-white font-semibold text-sm">Clinical Assistant</span>
                <span className="ml-auto text-[#6B7280] text-xs">RAG · Groq LLM · pgvector</span>
            </div>

            {/* History */}
            <div className="flex-1 overflow-y-auto max-h-96 flex flex-col gap-3 p-4">
                {history.length === 0 && (
                    <div className="flex flex-col gap-3">
                        <p className="text-[#6B7280] text-xs text-center py-2">
                            Ask a clinical question about <span className="text-white font-medium">{patient.name}</span>
                        </p>
                        <div className="grid grid-cols-2 gap-2">
                            {SUGGESTED_QUERIES.map((sq, i) => (
                                <button
                                    key={i}
                                    onClick={() => submit(sq)}
                                    className="text-left text-xs text-[#94A3B8] bg-[#0F172A] border border-[#334155] hover:border-[#2563EB] hover:text-white rounded-lg px-3 py-2 transition-colors"
                                >
                                    {sq}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {history.map((entry, idx) => (
                    <div key={idx} className="flex flex-col gap-2">
                        {/* Question bubble */}
                        <div className="flex justify-end">
                            <div className="bg-[#2563EB]/20 border border-[#2563EB]/30 text-[#93C5FD] text-xs rounded-lg px-3 py-2 max-w-[80%]">
                                {entry.question}
                            </div>
                        </div>

                        {/* Response */}
                        {entry.loading && (
                            <div className="flex items-center gap-2 text-[#6B7280] text-xs">
                                <Loader2 size={14} className="animate-spin text-[#2563EB]" />
                                Querying knowledge base...
                            </div>
                        )}
                        {entry.error && (
                            <div className="text-[#F87171] text-xs bg-[#DC2626]/10 border border-[#DC2626]/20 rounded-lg px-3 py-2">
                                {entry.error}
                            </div>
                        )}
                        {entry.brief && (
                            <div className="bg-[#0F172A] border border-[#334155] rounded-lg overflow-hidden">
                                <button
                                    onClick={() => setExpanded(expanded === idx ? null : idx)}
                                    className="w-full flex items-center justify-between px-3 py-2 text-xs text-[#94A3B8] hover:text-white transition-colors"
                                >
                                    <span className="font-medium">
                                        {(entry.brief.differential_diagnosis?.[0]?.condition || 'Clinical response ready').slice(0, 60)}...
                                    </span>
                                    {expanded === idx ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                </button>
                                {expanded === idx && (
                                    <div className="px-3 pb-4 border-t border-[#334155]">
                                        <AlertBrief brief={entry.brief} />
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div className="border-t border-[#334155] p-3 flex gap-2 shrink-0">
                <input
                    ref={inputRef}
                    type="text"
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    onKeyDown={onKeyDown}
                    placeholder="Ask a clinical question..."
                    className="flex-1 bg-[#0F172A] border border-[#334155] focus:border-[#2563EB] text-white text-sm rounded-lg px-3 py-2 outline-none placeholder-[#6B7280] transition-colors"
                />
                <button
                    onClick={() => submit(question)}
                    disabled={!question.trim() || history.some(h => h.loading)}
                    className="flex items-center gap-1.5 bg-[#2563EB] hover:bg-[#1D4ED8] disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs rounded-lg px-3 py-2 transition-colors shrink-0"
                >
                    <Send size={14} />
                    Ask
                </button>
            </div>
        </div>
    );
};
