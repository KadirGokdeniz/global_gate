// ✅ Güncellenmiş types/index.ts

export interface Message {
  id: string;
  question: string;
  answer: string;
  timestamp: Date;
  provider: string;
  model: string;
  sources?: Source[];
  session_id?: string;
  stats?: ResponseStats;
  performance?: PerformanceStats;
  airline_preference?: string;
  language: string;
  cot_enabled?: boolean;   // ✅ Yeni alan
  reasoning?: string;      // ✅ Yeni alan - CoT düşünme süreci
}

export interface Source {
  airline: string;
  source: string;
  content_preview: string;
  updated_date?: string;
  url?: string;
  similarity_score: number;
}

export interface ResponseStats {
  total_retrieved: number;
  avg_similarity: number;
  context_quality: 'high' | 'medium' | 'low';
}

export interface PerformanceStats {
  response_time: number;
  processing_time: number;
  retrieval_time: number;
}

export interface QuickQuestion {
  title: string;
  desc: string;
}

export interface QuickQuestionCategory {
  [category: string]: QuickQuestion[];
}

export interface APIResponse {
  success: boolean;
  session_id?: string;
  answer?: string;
  sources?: Source[];
  model?: string;
  provider?: string;
  stats?: ResponseStats;
  performance?: PerformanceStats;
  airline_preference?: string;
  language?: string;
  error?: string;
  cot_enabled?: boolean;   // ✅ Yeni alan
  reasoning?: string;      // ✅ Yeni alan
}

export interface APIConnection {
  success: boolean;
  url?: string;
  error?: string;
  models_ready?: boolean;
}

export type Language = 'en' | 'tr';
export type Provider = 'OpenAI' | 'Claude';

// ✅ "all" seçeneği kaldırıldı - Sadece thy ve pegasus
export type AirlinePreference = 'thy' | 'pegasus';

export type FeedbackType = 'helpful' | 'not_helpful' | 'too_slow' | 'incorrect';

export interface VoiceRecordingState {
  isRecording: boolean;
  isProcessing: boolean;
  transcript: string;
  error?: string;
  volume?: number;
}

export interface SessionStats {
  totalQueries: number;
  satisfactionRate: number;
  helpfulCount: number;
  totalFeedback: number;
}