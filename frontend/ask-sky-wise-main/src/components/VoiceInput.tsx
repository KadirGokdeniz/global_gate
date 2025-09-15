import { useVoiceRecording } from '@/hooks/useVoiceRecording';
import { Button } from '@/components/ui/button';
import { Mic, MicOff, Loader2, Volume2, VolumeX } from 'lucide-react';
import { useEffect, useState, useRef } from 'react';

interface VoiceInputProps {
  onTranscript: (transcript: string) => void;
  language: 'en' | 'tr';
}

export const VoiceInput = ({ onTranscript, language }: VoiceInputProps) => {
  const {
    isRecording,
    isProcessing,
    transcript,
    error,
    volume = 0,
    startRecording,
    stopRecording,
    reset
  } = useVoiceRecording(language);

  // ✅ NEW: Real-time transcript display
  const [realtimeTranscript, setRealtimeTranscript] = useState('');
  const [isAutoStopping, setIsAutoStopping] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [silenceDetected, setSilenceDetected] = useState(false);
  
  // ✅ NEW: Timers for auto-stop functionality
  const silenceTimeoutRef = useRef<NodeJS.Timeout>();
  const durationTimerRef = useRef<NodeJS.Timeout>();
  const maxRecordingTimeRef = useRef<NodeJS.Timeout>();

  // ✅ NEW: Voice activity detection parameters
  const SILENCE_THRESHOLD = 0.01; // Volume threshold for silence
  const SILENCE_DURATION = 2000; // 2 seconds of silence before auto-stop
  const MAX_RECORDING_TIME = 30000; // 30 seconds max recording
  const MIN_RECORDING_TIME = 1000; // 1 second minimum before auto-stop

  useEffect(() => {
    if (transcript) {
      onTranscript(transcript);
      reset();
      setRealtimeTranscript('');
      setRecordingDuration(0);
    }
  }, [transcript, onTranscript, reset]);

  // ✅ NEW: Recording duration timer
  useEffect(() => {
    if (isRecording) {
      const startTime = Date.now();
      durationTimerRef.current = setInterval(() => {
        setRecordingDuration(Date.now() - startTime);
      }, 100);

      // Maximum recording time safety
      maxRecordingTimeRef.current = setTimeout(() => {
        console.log('📢 Voice: Max recording time reached, auto-stopping');
        handleAutoStop('max_time');
      }, MAX_RECORDING_TIME);

      return () => {
        if (durationTimerRef.current) clearInterval(durationTimerRef.current);
        if (maxRecordingTimeRef.current) clearTimeout(maxRecordingTimeRef.current);
      };
    } else {
      setRecordingDuration(0);
      if (durationTimerRef.current) clearInterval(durationTimerRef.current);
      if (maxRecordingTimeRef.current) clearTimeout(maxRecordingTimeRef.current);
    }
  }, [isRecording]);

  // ✅ NEW: Voice activity detection and auto-stop  
  useEffect(() => {
    if (isRecording && recordingDuration > MIN_RECORDING_TIME) {
      
      if (volume < SILENCE_THRESHOLD) {
        // Silence detected - only start countdown if not already started
        if (!silenceDetected) {
          setSilenceDetected(true);
          console.log('🔇 Voice: Silence detected, starting countdown...');
          
          // Clear any existing timeout
          if (silenceTimeoutRef.current) {
            clearTimeout(silenceTimeoutRef.current);
          }
          
          silenceTimeoutRef.current = setTimeout(() => {
            console.log('⏹️ Voice: Silence timeout reached, auto-stopping');
            handleAutoStop('silence');
          }, SILENCE_DURATION);
        }
      } else {
        // Voice activity detected
        if (silenceDetected) {
          console.log('🔊 Voice: Voice detected, resetting silence timer');
          setSilenceDetected(false);
          if (silenceTimeoutRef.current) {
            clearTimeout(silenceTimeoutRef.current);
          }
        }
      }
    } else {
      setSilenceDetected(false);
      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current);
      }
    }

    return () => {
      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current);
      }
    };
  }, [volume, isRecording, recordingDuration, silenceDetected]); // silenceDetected dependency eklendi

  // ✅ NEW: Auto-stop handler
  const handleAutoStop = (reason: 'silence' | 'max_time') => {
    console.log(`🛑 Voice: Auto-stopping due to ${reason}`);
    setIsAutoStopping(true);
    stopRecording();
    setTimeout(() => setIsAutoStopping(false), 1000);
  };

  const handleToggleRecording = () => {
    if (isRecording) {
      console.log('🛑 Voice: Manual stop');
      stopRecording();
    } else {
      console.log('🎤 Voice: Manual start');
      setRealtimeTranscript('');
      setSilenceDetected(false);
      startRecording();
    }
  };

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

  // ✅ NEW: Format duration for display
  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    return `${seconds}s`;
  };

  // ✅ NEW: Calculate remaining time for silence countdown
  const getSilenceCountdown = () => {
    if (!silenceDetected || !isRecording) return 0;
    return Math.ceil(SILENCE_DURATION / 1000); // Always show max for now
  };

  return (
    <div className="flex flex-col items-center gap-4 max-w-md mx-auto">
      <div className="relative">
        <Button 
          onClick={handleToggleRecording}
          disabled={isProcessing}
          variant={isRecording ? "destructive" : "default"}
          size="lg"
          className={`
            relative overflow-hidden transition-all duration-300 h-16 px-8
            ${isRecording ? 'voice-recording animate-pulse bg-red-500 hover:bg-red-600' : 'bg-primary hover:bg-primary/90'}
            ${isProcessing ? 'opacity-70' : ''}
            ${isAutoStopping ? 'ring-4 ring-yellow-400' : ''}
            rounded-full shadow-xl
          `}
        >
          {getButtonContent()}
        </Button>
        
        {/* ✅ ENHANCED: Voice level indicator with better styling */}
        {isRecording && (
          <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 flex flex-col items-center gap-1">
            <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden shadow-inner">
              <div 
                className={`h-full transition-all duration-150 rounded-full shadow-sm ${
                  volume > SILENCE_THRESHOLD ? 'bg-green-500' : 'bg-red-300'
                }`}
                style={{ width: `${Math.min(100, volume * 100)}%` }}
              />
            </div>
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              {volume > SILENCE_THRESHOLD ? (
                <Volume2 className="w-3 h-3 text-green-500" />
              ) : (
                <VolumeX className="w-3 h-3 text-red-400" />
              )}
              <span>{(volume * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* ✅ NEW: Recording status with duration and auto-stop countdown */}
      {isRecording && (
        <div className="flex flex-col items-center gap-2 text-sm">
          <div className="flex items-center gap-3 text-muted-foreground bg-muted/30 px-4 py-2 rounded-full">
            <div className="voice-pulse w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
            <span className="font-medium">
              {language === 'en' ? 'Listening... Speak now' : 'Dinleniyor... Konuşun'}
            </span>
            <span className="text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded">
              {formatDuration(recordingDuration)}
            </span>
          </div>

          {/* ✅ NEW: Silence detection feedback */}
          {silenceDetected && recordingDuration > MIN_RECORDING_TIME && (
            <div className="flex items-center gap-2 text-orange-600 dark:text-orange-400 bg-orange-50 dark:bg-orange-900/20 px-3 py-2 rounded-lg border border-orange-200 dark:border-orange-800">
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
              <span className="text-xs font-medium">
                {language === 'en' 
                  ? `Auto-stopping in ${getSilenceCountdown()}s...` 
                  : `${getSilenceCountdown()}s içinde duracak...`}
              </span>
            </div>
          )}
        </div>
      )}

      {/* ✅ NEW: Real-time transcript preview */}
      {realtimeTranscript && (
        <div className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="text-xs text-blue-600 dark:text-blue-400 mb-1 font-medium">
            {language === 'en' ? 'Live Transcript:' : 'Canlı Metin:'}
          </div>
          <div className="text-sm text-blue-800 dark:text-blue-200 italic">
            "{realtimeTranscript}"
          </div>
        </div>
      )}

      {/* Status messages */}
      {error && (
        <div className="w-full text-center p-3 bg-destructive/10 rounded-lg border border-destructive/20">
          <p className="text-sm text-destructive font-medium">
            {error}
          </p>
        </div>
      )}
      
      {isProcessing && (
        <div className="w-full text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <p className="text-sm text-blue-700 dark:text-blue-300 font-medium">
            {language === 'en' ? 'Converting speech to text...' : 'Konuşma metne dönüştürülüyor...'}
          </p>
        </div>
      )}

      {/* ✅ NEW: Auto-stop confirmation */}
      {isAutoStopping && (
        <div className="w-full text-center p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
          <p className="text-sm text-yellow-700 dark:text-yellow-300 font-medium">
            {language === 'en' ? 'Auto-stopped recording' : 'Kayıt otomatik durduruldu'}
          </p>
        </div>
      )}

      {/* ✅ NEW: Recording tips */}
      {!isRecording && !isProcessing && (
        <div className="w-full text-center text-xs text-muted-foreground bg-muted/20 px-3 py-2 rounded-lg">
          {language === 'en' ? (
            <span>
              💡 Tip: Recording auto-stops after 2s of silence or 30s max
            </span>
          ) : (
            <span>
              💡 İpucu: 2sn sessizlik veya 30sn sonra otomatik durur
            </span>
          )}
        </div>
      )}
    </div>
  );
};