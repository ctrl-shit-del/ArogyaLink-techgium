# Data flow
1. Wearable → MQTT (synera/patient/{id}/vitals). 2. Backend subscriber → rule A/B/C. 3. On SYNERA_STATE → RAG service → ClinicalBrief → WebSocket + DB.
