import {
  APIResponse,
  APIConnection,
  Provider,
  AirlinePreference,
  FeedbackType,
  Language,
} from '@/types';

// ═══════════════════════════════════════════════════════════════════
// Konfigürasyon & build-time guard
// ═══════════════════════════════════════════════════════════════════

const PRODUCTION_URL = import.meta.env.VITE_API_URL as string | undefined;
const IS_PROD = import.meta.env.PROD;

// ✅ Build-time guard: production'da VITE_API_URL YOKSA hemen patla.
// Bu hata deploy'u engeller, sessizce localhost'a düşmeyi önler.
if (IS_PROD && !PRODUCTION_URL) {
  throw new Error(
    '[config] VITE_API_URL must be set in production. ' +
      'Set it in your .env.production file or CI/CD environment.',
  );
}

// ✅ HTTPS guard: production'da http:// URL'lere izin verme.
// Browser zaten mixed content'i bloklayacak ama açık hata mesajı daha iyi.
if (IS_PROD && PRODUCTION_URL && !PRODUCTION_URL.startsWith('https://')) {
  throw new Error(
    `[config] VITE_API_URL must use https:// in production. Got: ${PRODUCTION_URL}`,
  );
}

// ✅ Endpoint listesi: prod'da TEK URL, dev'de fallback'ler
const API_ENDPOINTS: string[] = PRODUCTION_URL
  ? [PRODUCTION_URL.replace(/\/$/, '')] // trailing slash temizle
  : [
      'http://localhost:8000',
      'http://127.0.0.1:8000',
      'http://localhost:8080',
      'http://127.0.0.1:8080',
    ];

// Production'da console.log'ları sessize al
const log = import.meta.env.DEV ? console.log : () => {};
const logError = console.error;

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

/**
 * AbortController tabanlı timeout. fetch'i belirli süre sonra iptal eder.
 * Başarılı/başarısız her durumda timer temizlenir — leak yok.
 */
async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(input, {
      ...init,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Hata mesajını kullanıcıya göstermek için güvenli string'e çevir.
 */
function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    if (error.name === 'AbortError') return 'Request timeout';
    return error.message;
  }
  return 'Unknown error';
}

// ═══════════════════════════════════════════════════════════════════
// APIService
// ═══════════════════════════════════════════════════════════════════

class APIService {
  private baseUrl: string | null = null;

  // Zaman aşımı sabitleri — tek yerde tanımlı
  private static readonly HEALTH_TIMEOUT_MS = 5000;
  private static readonly QUERY_TIMEOUT_MS = 60000; // LLM yanıtları uzun olabilir
  private static readonly STT_TIMEOUT_MS = 45000;
  private static readonly TTS_TIMEOUT_MS = 30000;
  private static readonly FEEDBACK_TIMEOUT_MS = 10000;

  /**
   * Sağlıklı bir API endpoint'i bul ve `baseUrl`'e kaydet.
   * Prod'da sadece VITE_API_URL denenir; dev'de birkaç localhost varyantı.
   */
  async findWorkingAPI(): Promise<APIConnection> {
    for (const endpoint of API_ENDPOINTS) {
      // Her endpoint için önce /health, o yoksa / dene
      for (const path of ['/health', '/']) {
        try {
          const response = await fetchWithTimeout(
            `${endpoint}${path}`,
            { method: 'GET' },
            APIService.HEALTH_TIMEOUT_MS,
          );

          if (response.ok) {
            const data = await response.json().catch(() => ({}));
            this.baseUrl = endpoint;
            log(`✅ API: Connected to ${endpoint}${path}`);
            return {
              success: true,
              url: endpoint,
              // Açık niyet: backend bu alanı döndürmezse true varsay
              models_ready: data.models_ready ?? true,
            };
          }
          // response.ok değilse sıradaki path'i dene
        } catch (error) {
          log(`API: ${endpoint}${path} failed:`, errorMessage(error));
          // Timeout veya network hatası → sıradaki path/endpoint
        }
      }
    }

    return {
      success: false,
      error: IS_PROD
        ? 'Unable to reach API server'
        : 'No API endpoint available (tried localhost:8000/8080)',
    };
  }

  async queryAirlinePolicy(
    question: string,
    provider: Provider,
    model: string,
    airlinePreference: AirlinePreference,
    language: Language,
    enableCoT: boolean = false,
  ): Promise<APIResponse> {
    if (!this.baseUrl) {
      return { success: false, error: 'API not connected' };
    }

    try {
      const endpoint = provider === 'OpenAI' ? '/chat/openai' : '/chat/claude';

      const params = new URLSearchParams({
        question,
        model,
        language,
        max_results: '3',
        similarity_threshold: '0.3',
        enable_cot: enableCoT.toString(),
      });

      const airlineMap: Record<AirlinePreference, string> = {
        thy: 'turkish_airlines',
        pegasus: 'pegasus',
      };
      params.append('airline_preference', airlineMap[airlinePreference]);

      const response = await fetchWithTimeout(
        `${this.baseUrl}${endpoint}?${params}`,
        {
          method: 'GET',
          headers: { Accept: 'application/json' },
        },
        APIService.QUERY_TIMEOUT_MS,
      );

      if (!response.ok) {
        return { success: false, error: `API Error: ${response.status}` };
      }

      const data = await response.json();
      if (!data.success) {
        return { success: false, error: data.error || 'Processing failed' };
      }

      return {
        success: true,
        session_id: data.session_id,
        answer: data.answer,
        sources: data.sources || [],
        model: data.model_used || model,
        provider,
        stats: data.stats || {},
        performance: data.performance || {},
        airline_preference: data.airline_preference,
        language: data.language || language,
        cot_enabled: enableCoT,
        reasoning: data.reasoning,
      };
    } catch (error) {
      logError('API query error:', error);
      return { success: false, error: errorMessage(error) };
    }
  }

  async sendFeedback(
    question: string,
    answer: string,
    feedbackType: FeedbackType,
    provider: string,
    model: string,
  ): Promise<boolean> {
    if (!this.baseUrl) return false;

    try {
      const response = await fetchWithTimeout(
        `${this.baseUrl}/feedback`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question,
            answer,
            feedback_type: feedbackType,
            provider,
            model,
          }),
        },
        APIService.FEEDBACK_TIMEOUT_MS,
      );
      return response.ok;
    } catch (error) {
      logError('Feedback error:', errorMessage(error));
      return false;
    }
  }

  async convertTextToSpeech(
    text: string,
    language: Language,
  ): Promise<string | null> {
    if (!this.baseUrl) return null;

    try {
      const languageMap: Record<Language, string> = {
        en: 'en-US',
        tr: 'tr-TR',
      };
      const params = new URLSearchParams({
        text: text.trim(),
        language: languageMap[language],
      });

      const response = await fetchWithTimeout(
        `${this.baseUrl}/speech/synthesize?${params}`,
        {
          method: 'POST',
          headers: { Accept: 'audio/mpeg, audio/wav, audio/*' },
        },
        APIService.TTS_TIMEOUT_MS,
      );

      if (!response.ok) return null;

      const blob = await response.blob();
      if (blob.size === 0) return null;

      return URL.createObjectURL(blob);
    } catch (error) {
      logError('TTS error:', errorMessage(error));
      return null;
    }
  }

  async convertSpeechToText(
    audioBlob: Blob,
    language: Language,
  ): Promise<string | null> {
    if (!this.baseUrl) return null;

    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'recording.webm');

      const languageMap: Record<Language, string> = { en: 'en', tr: 'tr' };

      const response = await fetchWithTimeout(
        `${this.baseUrl}/speech/transcribe?language=${languageMap[language]}`,
        { method: 'POST', body: formData },
        APIService.STT_TIMEOUT_MS,
      );

      if (!response.ok) return null;

      const data = await response.json();
      return data.success && data.transcript ? data.transcript : null;
    } catch (error) {
      logError('STT error:', errorMessage(error));
      return null;
    }
  }

  async checkSpeechHealth(): Promise<{
    tts: boolean;
    stt: boolean;
    details?: unknown;
  }> {
    if (!this.baseUrl) return { tts: false, stt: false };

    try {
      const response = await fetchWithTimeout(
        `${this.baseUrl}/speech/health`,
        {},
        APIService.HEALTH_TIMEOUT_MS,
      );
      if (!response.ok) return { tts: false, stt: false };

      const data = await response.json();
      return {
        tts: data.services?.polly?.status === 'ready',
        stt: data.services?.assemblyai?.status === 'ready',
        details: data,
      };
    } catch {
      return { tts: false, stt: false };
    }
  }

  reconnect() {
    this.baseUrl = null;
    return this.findWorkingAPI();
  }

  // Test veya debug için yararlı
  getBaseUrl(): string | null {
    return this.baseUrl;
  }
}

export const apiService = new APIService();