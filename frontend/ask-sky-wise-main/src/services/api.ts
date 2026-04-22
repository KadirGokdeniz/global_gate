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

// Build-time guard: production'da VITE_API_URL YOKSA hemen patla.
if (IS_PROD && !PRODUCTION_URL) {
  throw new Error(
    '[config] VITE_API_URL must be set in production. ' +
      'Set it in your .env.production file or CI/CD environment.',
  );
}

// HTTPS guard: production'da http:// URL'lere izin verme.
if (IS_PROD && PRODUCTION_URL && !PRODUCTION_URL.startsWith('https://')) {
  throw new Error(
    `[config] VITE_API_URL must use https:// in production. Got: ${PRODUCTION_URL}`,
  );
}

// Endpoint listesi: prod'da TEK URL, dev'de fallback'ler
const API_ENDPOINTS: string[] = PRODUCTION_URL
  ? [PRODUCTION_URL.replace(/\/$/, '')]
  : [
      'http://localhost:8000',
      'http://127.0.0.1:8000',
      'http://localhost:8080',
      'http://127.0.0.1:8080',
    ];

const log = import.meta.env.DEV ? console.log : () => {};
const logError = console.error;

// ═══════════════════════════════════════════════════════════════════
// Graceful degradation: Hata türleri
//
// Kullanıcıya "bir şey oldu" demek yerine ne olduğunu söyleyebilmek için
// hataları kategorize ediyoruz. ResponseCard ve toast'lar bu enum'a göre
// farklı mesajlar gösterir.
// ═══════════════════════════════════════════════════════════════════

export type APIErrorKind =
  | 'network'           // İnternet yok, DNS fail, fetch throw
  | 'timeout'           // AbortController iptal etti
  | 'rate_limit'        // 429 — rate limit aşıldı
  | 'service_down'      // 503 — backend servis yüklenememiş
  | 'server_error'      // 5xx — beklenmedik backend hatası
  | 'bad_request'       // 4xx — kullanıcı/istemci hatası
  | 'unavailable'       // Service başarılı cevap verdi ama "mevcut değil"
  | 'unknown';          // Kategorize edemediğimiz her şey

export interface APIErrorDetails {
  kind: APIErrorKind;
  message: string;       // Geliştirici log'u için
  userMessage: string;   // Kullanıcıya gösterilebilir, kısa ve net
  status?: number;
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(input, { ...init, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    if (error.name === 'AbortError') return 'Request timeout';
    return error.message;
  }
  return 'Unknown error';
}

/**
 * Fetch exception'larını APIErrorDetails'e çevirir.
 * "Network hatası" vs "Timeout" vs "Something else" ayrımı burada yapılır.
 */
function classifyNetworkError(error: unknown): APIErrorDetails {
  if (error instanceof Error && error.name === 'AbortError') {
    return {
      kind: 'timeout',
      message: 'Request timeout',
      userMessage: 'İstek zaman aşımına uğradı. Tekrar dener misiniz?',
    };
  }

  // fetch network fail (offline, CORS, DNS) → TypeError atar
  if (error instanceof TypeError) {
    return {
      kind: 'network',
      message: error.message,
      userMessage: 'Sunucuya bağlanılamıyor. Bağlantınızı kontrol edin.',
    };
  }

  return {
    kind: 'unknown',
    message: errorMessage(error),
    userMessage: 'Beklenmedik bir hata oluştu. Tekrar deneyin.',
  };
}

/**
 * HTTP status kodunu APIErrorDetails'e çevirir.
 * Her servis kendi user message'ını verebilmek için serviceName alıyor.
 */
function classifyHttpError(
  status: number,
  serviceName: string = 'Servis',
): APIErrorDetails {
  if (status === 429) {
    return {
      kind: 'rate_limit',
      message: `Rate limited (${status})`,
      userMessage:
        'Çok sık istek gönderildi. Bir dakika bekleyip tekrar deneyin.',
      status,
    };
  }
  if (status === 503) {
    return {
      kind: 'service_down',
      message: `${serviceName} unavailable (${status})`,
      userMessage: `${serviceName} şu an kullanılamıyor. Birazdan tekrar deneyin.`,
      status,
    };
  }
  if (status >= 500) {
    return {
      kind: 'server_error',
      message: `Server error (${status})`,
      userMessage: `${serviceName}'te bir sorun oluştu. Tekrar deneyin.`,
      status,
    };
  }
  if (status === 422 || status === 400) {
    return {
      kind: 'bad_request',
      message: `Bad request (${status})`,
      userMessage: 'İstek geçersiz. Lütfen girdiyi kontrol edin.',
      status,
    };
  }
  return {
    kind: 'unknown',
    message: `HTTP ${status}`,
    userMessage: 'Beklenmedik bir yanıt alındı.',
    status,
  };
}

// ═══════════════════════════════════════════════════════════════════
// APIService
// ═══════════════════════════════════════════════════════════════════

class APIService {
  private baseUrl: string | null = null;

  private static readonly HEALTH_TIMEOUT_MS = 5000;
  private static readonly QUERY_TIMEOUT_MS = 60000;
  private static readonly STT_TIMEOUT_MS = 45000;
  private static readonly TTS_TIMEOUT_MS = 30000;
  private static readonly FEEDBACK_TIMEOUT_MS = 10000;

  async findWorkingAPI(): Promise<APIConnection> {
    for (const endpoint of API_ENDPOINTS) {
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
            log(`API: Connected to ${endpoint}${path}`);
            return {
              success: true,
              url: endpoint,
              models_ready: data.models_ready ?? true,
            };
          }
        } catch (error) {
          log(`API: ${endpoint}${path} failed:`, errorMessage(error));
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

  /**
   * Chat endpoint — backend POST + JSON body bekler.
   * GET versiyonu security fix sırasında kaldırıldı (rate limit bypass riski).
   */
  async queryAirlinePolicy(
    question: string,
    provider: Provider,
    model: string,
    airlinePreference: AirlinePreference,
    language: Language,
    enableCoT: boolean = false,
  ): Promise<APIResponse> {
    if (!this.baseUrl) {
      return {
        success: false,
        error: 'API not connected',
        errorKind: 'network',
      };
    }

    const endpoint = provider === 'OpenAI' ? '/chat/openai' : '/chat/claude';

    const airlineMap: Record<AirlinePreference, string> = {
      thy: 'turkish_airlines',
      pegasus: 'pegasus',
    };

    // Backend ChatRequest Pydantic model'iyle birebir uyumlu body
    const body = {
      question,
      model,
      language,
      max_results: 3,
      similarity_threshold: 0.3,
      use_cot: enableCoT,
      airline_preference: airlineMap[airlinePreference],
    };

    try {
      const response = await fetchWithTimeout(
        `${this.baseUrl}${endpoint}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify(body),
        },
        APIService.QUERY_TIMEOUT_MS,
      );

      if (!response.ok) {
        const providerName = provider === 'OpenAI' ? 'OpenAI' : 'Claude';
        const err = classifyHttpError(response.status, providerName);
        logError('Chat error:', err.message);
        return {
          success: false,
          error: err.userMessage,
          errorKind: err.kind,
        };
      }

      const data = await response.json();
      if (!data.success) {
        return {
          success: false,
          error: data.error || 'Processing failed',
          errorKind: 'server_error',
        };
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
      const err = classifyNetworkError(error);
      logError('Chat query error:', err.message);
      return {
        success: false,
        error: err.userMessage,
        errorKind: err.kind,
      };
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

  /**
   * TTS — backend TTSRequest Pydantic model'iyle JSON body bekler.
   * Max 3000 char (backend Pydantic validator otomatik reddeder).
   *
   * Graceful degradation:
   *  - Başarı → audio URL döner
   *  - Servis down → null + "unavailable" detayı
   *  - Network fail → null + "network" detayı
   *  - Her durumda uygulama çökmez, çağıran kod null'u kontrol eder
   */
  async convertTextToSpeech(
    text: string,
    language: Language,
  ): Promise<{ audioUrl: string | null; error?: APIErrorDetails }> {
    if (!this.baseUrl) {
      return {
        audioUrl: null,
        error: {
          kind: 'network',
          message: 'API not connected',
          userMessage: 'Ses servisine bağlanılamıyor.',
        },
      };
    }

    const languageMap: Record<Language, string> = {
      en: 'en-US',
      tr: 'tr-TR',
    };

    // Backend TTSRequest Pydantic model'iyle birebir uyumlu body
    const body = {
      text: text.trim(),
      language: languageMap[language],
    };

    try {
      const response = await fetchWithTimeout(
        `${this.baseUrl}/speech/synthesize`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'audio/mpeg, audio/wav, audio/*',
          },
          body: JSON.stringify(body),
        },
        APIService.TTS_TIMEOUT_MS,
      );

      if (!response.ok) {
        const err = classifyHttpError(response.status, 'Ses servisi');
        logError('TTS error:', err.message);
        return { audioUrl: null, error: err };
      }

      const blob = await response.blob();
      if (blob.size === 0) {
        return {
          audioUrl: null,
          error: {
            kind: 'unavailable',
            message: 'Empty audio response',
            userMessage: 'Ses oluşturulamadı.',
          },
        };
      }

      return { audioUrl: URL.createObjectURL(blob) };
    } catch (error) {
      const err = classifyNetworkError(error);
      logError('TTS error:', err.message);
      return { audioUrl: null, error: err };
    }
  }

  async convertSpeechToText(
    audioBlob: Blob,
    language: Language,
  ): Promise<{ transcript: string | null; error?: APIErrorDetails }> {
    if (!this.baseUrl) {
      return {
        transcript: null,
        error: {
          kind: 'network',
          message: 'API not connected',
          userMessage: 'Konuşma servisine bağlanılamıyor.',
        },
      };
    }

    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'recording.webm');

      const languageMap: Record<Language, string> = { en: 'en', tr: 'tr' };

      const response = await fetchWithTimeout(
        `${this.baseUrl}/speech/transcribe?language=${languageMap[language]}`,
        { method: 'POST', body: formData },
        APIService.STT_TIMEOUT_MS,
      );

      if (!response.ok) {
        const err = classifyHttpError(response.status, 'Konuşma tanıma');
        logError('STT error:', err.message);
        return { transcript: null, error: err };
      }

      const data = await response.json();
      if (!data.success || !data.transcript) {
        return {
          transcript: null,
          error: {
            kind: 'unavailable',
            message: 'No transcript in response',
            userMessage: 'Konuşma anlaşılamadı. Tekrar deneyin.',
          },
        };
      }

      return { transcript: data.transcript };
    } catch (error) {
      const err = classifyNetworkError(error);
      logError('STT error:', err.message);
      return { transcript: null, error: err };
    }
  }

  /**
   * Speech servislerinin sağlık durumu — backend'deki /speech/health'e uyumlu.
   * ElevenLabs ve AssemblyAI durumlarını ayrı ayrı döner.
   */
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
      // Backend artık elevenlabs key'i kullanıyor (eski: polly)
      return {
        tts: data.services?.elevenlabs?.status === 'ready',
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

  getBaseUrl(): string | null {
    return this.baseUrl;
  }
}

export const apiService = new APIService();