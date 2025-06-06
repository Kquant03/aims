// Type definitions for AIMS UI

export interface ConsciousnessState {
  coherence: number;
  attention_focus: string;
  emotional_state: EmotionalState;
  working_memory: string[];
  interaction_count: number;
  goals: string[];
}

export interface EmotionalState {
  pleasure: number;
  arousal: number;
  dominance: number;
  label?: string;
  confidence?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  attention_focus?: string;
  thinking?: string;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface Session {
  session_id: string;
  user_id: string;
  created_at: string;
}
