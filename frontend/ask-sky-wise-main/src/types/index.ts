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
}

export interface Source {
  source: string;
  similarity_score: number;
  airline: string;
  content?: string;
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
}

export interface APIConnection {
  success: boolean;
  url?: string;
  error?: string;
  models_ready?: boolean;
}

export type Language = 'en' | 'tr';
export type Provider = 'OpenAI' | 'Claude';
export type AirlinePreference = 'all' | 'thy' | 'pegasus';
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