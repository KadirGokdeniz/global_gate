import { APIResponse, APIConnection, Provider, AirlinePreference, FeedbackType, Language } from '@/types';

const API_ENDPOINTS = [
  'http://localhost:8000',
  'http://127.0.0.1:8000',
  'http://localhost:8080',
  'http://127.0.0.1:8080'
];

class APIService {
  private baseUrl: string | null = null;

  async findWorkingAPI(): Promise<APIConnection> {
    for (const endpoint of API_ENDPOINTS) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        let response;
        try {
          response = await fetch(`${endpoint}/health`, {
            method: 'GET',
            signal: controller.signal
          });
        } catch (healthError) {
          response = await fetch(`${endpoint}/`, {
            method: 'GET',
            signal: controller.signal
          });
        }
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
          const data = await response.json();
          this.baseUrl = endpoint;
          return {
            success: true,
            url: endpoint,
            models_ready: data.models_ready || true
          };
        }
      } catch (error) {
        continue;
      }
    }
    
    return {
      success: false,
      error: 'No API endpoint available'
    };
  }

  // ✅ CoT (Chain of Thought) parametresi eklendi
  async queryAirlinePolicy(
    question: string,
    provider: Provider,
    model: string,
    airlinePreference: AirlinePreference,
    language: Language,
    enableCoT: boolean = false  // ✅ Yeni parametre
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
        enable_cot: enableCoT.toString()  // ✅ CoT parametresi
      });

      // ✅ "all" seçeneği artık kullanılmıyor - her zaman belirli havayolu
      const airlineMap: Record<string, string> = {
        'thy': 'turkish_airlines',
        'pegasus': 'pegasus'
      };
      params.append('airline_preference', airlineMap[airlinePreference] || airlinePreference);

      const response = await fetch(`${this.baseUrl}${endpoint}?${params}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
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
            cot_enabled: enableCoT,  // ✅ CoT durumunu response'a ekle
            reasoning: data.reasoning  // ✅ CoT reasoning (varsa)
          };
        } else {
          return { success: false, error: data.error || 'Processing failed' };
        }
      } else {
        return { success: false, error: `API Error: ${response.status}` };
      }
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          return { success: false, error: 'Request timeout (30s)' };
        }
        return { success: false, error: `Connection error: ${error.message}` };
      }
      return { success: false, error: 'Unknown error occurred' };
    }
  }

  async sendFeedback(
    question: string,
    answer: string,
    feedbackType: FeedbackType,
    provider: string,
    model: string
  ): Promise<boolean> {
    console.log('🚀 sendFeedback called with:', {
      baseUrl: this.baseUrl,
      feedbackType,
      provider,
      model,
      questionLength: question.length,
      answerLength: answer.length
    });

    if (!this.baseUrl) {
      console.error('❌ Feedback failed: API not connected');
      return false;
    }

    try {
      const feedbackData = {
        question,
        answer,
        feedback_type: feedbackType,
        provider,
        model
      };

      console.log('📤 Sending feedback request:', feedbackData);

      const response = await fetch(`${this.baseUrl}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData)
      });

      console.log('📥 Feedback response status:', response.status, response.statusText);

      if (response.ok) {
        const responseData = await response.json();
        console.log('✅ Feedback success:', responseData);
        return true;
      } else {
        const errorText = await response.text();
        console.error('❌ Feedback HTTP error:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText
        });
        return false;
      }
    } catch (error) {
      console.error('❌ Feedback network error:', error);
      if (error instanceof Error) {
        console.error('Error details:', {
          name: error.name,
          message: error.message,
          stack: error.stack
        });
      }
      return false;
    }
  }

  async convertTextToSpeech(text: string, language: Language): Promise<string | null> {
    console.log('🔊 API: TTS request started', { textLength: text.length, language });
    
    if (!this.baseUrl) {
      console.error('❌ API: No base URL for TTS');
      return null;
    }

    try {
      const languageMap = {
        'en': 'en-US',
        'tr': 'tr-TR'
      };

      const params = new URLSearchParams({
        text: text.trim(),
        language: languageMap[language] || 'tr-TR'
      });

      console.log('📤 API: Sending TTS request with params:', Object.fromEntries(params));

      const response = await fetch(`${this.baseUrl}/speech/synthesize?${params}`, {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg, audio/wav, audio/*'
        }
      });

      console.log('📥 API: TTS response status:', response.status, response.statusText);

      if (response.ok) {
        const blob = await response.blob();
        console.log('✅ API: TTS blob received:', blob.size, 'bytes');
        return URL.createObjectURL(blob);
      } else {
        const errorText = await response.text();
        console.error('❌ API: TTS HTTP error:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText.substring(0, 200)
        });
        return null;
      }
    } catch (error) {
      console.error('❌ API: TTS network error:', error);
      return null;
    }
  }

  async convertSpeechToText(audioBlob: Blob, language: Language): Promise<string | null> {
    if (!this.baseUrl) {
      return null;
    }

    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'recording.webm');
      
      const languageMap = {
        'en': 'en',
        'tr': 'tr'
      };

      const response = await fetch(`${this.baseUrl}/speech/transcribe?language=${languageMap[language] || 'tr'}`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        console.log('STT Response:', data);
        
        if (data.success && data.transcript) {
          return data.transcript;
        } else {
          console.error('STT failed:', data.error || 'No transcript');
          return null;
        }
      } else {
        console.error('STT HTTP Error:', response.status, response.statusText);
        return null;
      }
    } catch (error) {
      console.error('STT error:', error);
      return null;
    }
  }

  async checkSpeechHealth(): Promise<{ tts: boolean, stt: boolean, details?: any }> {
    if (!this.baseUrl) {
      return { tts: false, stt: false };
    }

    try {
      const response = await fetch(`${this.baseUrl}/speech/health`);
      if (response.ok) {
        const data = await response.json();
        return {
          tts: data.services?.polly?.status === 'ready',
          stt: data.services?.assemblyai?.status === 'ready',
          details: data
        };
      }
      return { tts: false, stt: false };
    } catch (error) {
      console.error('Speech health check error:', error);
      return { tts: false, stt: false };
    }
  }

  reconnect() {
    this.baseUrl = null;
    return this.findWorkingAPI();
  }
}

export const apiService = new APIService();