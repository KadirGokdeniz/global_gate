import { useVoiceRecording } from '@/hooks/useVoiceRecording';
import { Button } from '@/components/ui/button';
import { Mic, MicOff, Loader2, Volume2, VolumeX } from 'lucide-react';
import { useEffect } from 'react';

interface VoiceInputProps {
  onTranscript: (transcript: string) => void;
  language: 'en' | 'tr';
}

// UI için kullanılan eşik — hook ile aynı değer (sadece görsel gösterim için)
const VOLUME_DISPLAY_THRESHOLD = 0.01;

export const VoiceInput = ({ onTranscript, language }: VoiceInputProps) => {
  const {
    isRecording,
    isProcessing,
    transcript,
    error,
    volume,
    silenceDetected,
    recordingDuration,
    autoStoppedReason,
    startRecording,
    stopRecording,
    reset,
  } = useVoiceRecording(language);

  // Transcript geldiğinde parent'a ilet ve hook'u sıfırla
  useEffect(() => {
    if (transcript) {
      onTranscript(transcript);
      reset();
    }
  }, [transcript, onTranscript, reset]);

  const handleToggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const formatDuration = (ms: number) => `${Math.floor(ms / 1000)}s`;

  const getButtonContent = () => {
    if (isProcessing) {
      return (
        <>
          <Loader2 className="w-5 h-5 animate-spin" />
          {language === 'en' ? 'Processing...' : 'İşleniyor...'}
        </>
      );
    }
    if (isRecording) {
      return (
        <>
          <MicOff className="w-5 h-5" />
          {language === 'en' ? 'Stop Recording' : 'Kaydı Durdur'}
        </>
      );
    }
    return (
      <>
        <Mic className="w-5 h-5" />
        {language === 'en' ? 'Start Recording' : 'Kayda Başla'}
      </>
    );
  };

  return (
    <div className="flex flex-col items-center gap-4 max-w-md mx-auto">
      <div className="relative">
        <Button
          onClick={handleToggleRecording}
          disabled={isProcessing}
          variant={isRecording ? 'destructive' : 'default'}
          size="lg"
          aria-label={
            isRecording
              ? language === 'en'
                ? 'Stop recording'
                : 'Kaydı durdur'
              : language === 'en'
                ? 'Start recording'
                : 'Kayda başla'
          }
          aria-pressed={isRecording}
          className={`
            relative overflow-hidden transition-all duration-300 h-16 px-8
            ${
              isRecording
                ? 'voice-recording animate-pulse bg-red-500 hover:bg-red-600'
                : 'bg-primary hover:bg-primary/90'
            }
            ${isProcessing ? 'opacity-70' : ''}
            rounded-full shadow-xl
          `}
        >
          {getButtonContent()}
        </Button>

        {/* Volume indicator */}
        {isRecording && (
          <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 flex flex-col items-center gap-1">
            <div
              className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden shadow-inner"
              role="progressbar"
              aria-label={language === 'en' ? 'Voice volume' : 'Ses seviyesi'}
              aria-valuenow={Math.round(volume * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className={`h-full transition-all duration-150 rounded-full shadow-sm ${
                  volume > VOLUME_DISPLAY_THRESHOLD
                    ? 'bg-green-500'
                    : 'bg-red-300'
                }`}
                style={{ width: `${Math.min(100, volume * 100)}%` }}
              />
            </div>
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              {volume > VOLUME_DISPLAY_THRESHOLD ? (
                <Volume2 className="w-3 h-3 text-green-500" />
              ) : (
                <VolumeX className="w-3 h-3 text-red-400" />
              )}
              <span>{(volume * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Recording status */}
      {isRecording && (
        <div className="flex flex-col items-center gap-2 text-sm">
          <div
            className="flex items-center gap-3 text-muted-foreground bg-muted/30 px-4 py-2 rounded-full"
            role="status"
            aria-live="polite"
          >
            <div className="voice-pulse w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            <span className="font-medium">
              {language === 'en'
                ? 'Listening... Speak now'
                : 'Dinleniyor... Konuşun'}
            </span>
            <span className="text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded">
              {formatDuration(recordingDuration)}
            </span>
          </div>

          {silenceDetected && (
            <div
              className="flex items-center gap-2 text-orange-600 dark:text-orange-400 bg-orange-50 dark:bg-orange-900/20 px-3 py-2 rounded-lg border border-orange-200 dark:border-orange-800"
              role="status"
              aria-live="polite"
            >
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse" />
              <span className="text-xs font-medium">
                {language === 'en'
                  ? 'Silence detected — auto-stopping soon...'
                  : 'Sessizlik algılandı — yakında duracak...'}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          className="w-full text-center p-3 bg-destructive/10 rounded-lg border border-destructive/20"
          role="alert"
        >
          <p className="text-sm text-destructive font-medium">{error}</p>
        </div>
      )}

      {/* Processing */}
      {isProcessing && (
        <div
          className="w-full text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
          role="status"
          aria-live="polite"
        >
          <p className="text-sm text-blue-700 dark:text-blue-300 font-medium">
            {language === 'en'
              ? 'Converting speech to text...'
              : 'Konuşma metne dönüştürülüyor...'}
          </p>
        </div>
      )}

      {/* Auto-stop bildirimi (son kayıttan) */}
      {autoStoppedReason && !isRecording && !isProcessing && (
        <div className="w-full text-center p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
          <p className="text-sm text-yellow-700 dark:text-yellow-300 font-medium">
            {autoStoppedReason === 'silence'
              ? language === 'en'
                ? 'Auto-stopped due to silence'
                : 'Sessizlik nedeniyle otomatik durduruldu'
              : language === 'en'
                ? 'Auto-stopped (max duration reached)'
                : 'Maksimum süreye ulaşıldı, durduruldu'}
          </p>
        </div>
      )}

      {/* Tip */}
      {!isRecording && !isProcessing && !autoStoppedReason && !error && (
        <div className="w-full text-center text-xs text-muted-foreground bg-muted/20 px-3 py-2 rounded-lg">
          <span>
            {language === 'en'
              ? '💡 Tip: Recording auto-stops after 2s of silence or 30s max'
              : '💡 İpucu: 2sn sessizlik veya 30sn sonra otomatik durur'}
          </span>
        </div>
      )}
    </div>
  );
};