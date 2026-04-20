import { useState, useCallback, useRef, useEffect } from 'react';
import { VoiceRecordingState } from '@/types';

// ✅ Voice activity sabitleri — TEK YERDE tanımlı.
// VoiceInput.tsx artık kendi sabitlerini kullanmıyor.
const SILENCE_THRESHOLD = 0.01;
const SILENCE_DURATION_MS = 2000;
const MIN_RECORDING_TIME_MS = 1000;
const MAX_RECORDING_TIME_MS = 30000;
const VOLUME_HISTORY_SIZE = 5;

// Production'da gereksiz log gürültüsünü kes; hataları her zaman göster.
const log = import.meta.env.DEV ? console.log : () => {};
const logError = console.error;

type StopReason = 'silence' | 'max_time';

export const useVoiceRecording = (language: 'en' | 'tr' = 'en') => {
  const [state, setState] = useState<VoiceRecordingState>({
    isRecording: false,
    isProcessing: false,
    transcript: '',
    volume: 0,
    silenceDetected: false,
    recordingDuration: 0,
  });

  // ─── Refs ──────────────────────────────────────────────────────────────
  // Stale closure problemini çözen kritik ref:
  // requestAnimationFrame döngüsünde state.isRecording güncel olmadığı için
  // döngüyü sürdürüp sürdürmeyeceğimize bu ref üzerinden karar veriyoruz.
  const isRecordingRef = useRef(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const volumeHistoryRef = useRef<number[]>([]);
  const silenceStartTimeRef = useRef<number | null>(null);
  const recordingStartTimeRef = useRef<number>(0);

  // stopRecording'i updateVolume içinden çağırırken closure yakalamasını
  // engellemek için ref üzerinden çağırıyoruz. Her render'da güncellenir.
  const stopRecordingRef = useRef<(reason?: StopReason) => void>(() => {});

  // ─── Cleanup ──────────────────────────────────────────────────────────
  const cleanupAudioResources = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {
        // AudioContext zaten kapalıysa fırlatan hatayı yutuyoruz
      });
      audioContextRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    analyserRef.current = null;
    volumeHistoryRef.current = [];
    silenceStartTimeRef.current = null;
  }, []);

  // ─── Volume monitoring & silence detection ────────────────────────────
  // useCallback DEĞİL — her rerender'da yeniden yaratılsın istemiyoruz,
  // ref'ler üzerinden çalıştığı için stabil bir fonksiyon yeterli.
  const updateVolume = () => {
    if (!isRecordingRef.current || !analyserRef.current) {
      return;
    }

    // Time-domain data ile RMS hesapla — voice activity için frequency
    // data'sından daha doğru sonuç veriyor.
    const dataArray = new Uint8Array(analyserRef.current.fftSize);
    analyserRef.current.getByteTimeDomainData(dataArray);

    let sumSquares = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const sample = (dataArray[i] - 128) / 128;
      sumSquares += sample * sample;
    }
    const rms = Math.sqrt(sumSquares / dataArray.length);
    const instantVolume = Math.min(1, rms * 10);

    // Sliding window smoothing — 5 frame ortalaması
    volumeHistoryRef.current.push(instantVolume);
    if (volumeHistoryRef.current.length > VOLUME_HISTORY_SIZE) {
      volumeHistoryRef.current.shift();
    }
    const smoothedVolume =
      volumeHistoryRef.current.reduce((a, b) => a + b, 0) /
      volumeHistoryRef.current.length;

    const now = Date.now();
    const recordingDuration = now - recordingStartTimeRef.current;

    // Silence detection (sadece min süre geçtikten sonra)
    let silenceDetected = false;
    if (recordingDuration > MIN_RECORDING_TIME_MS) {
      if (smoothedVolume < SILENCE_THRESHOLD) {
        if (silenceStartTimeRef.current === null) {
          silenceStartTimeRef.current = now;
          log('🔇 Voice Hook: Silence started');
        }
        silenceDetected = true;

        const silenceDuration = now - silenceStartTimeRef.current;
        if (silenceDuration >= SILENCE_DURATION_MS) {
          log('⏹️ Voice Hook: Silence threshold reached, auto-stopping');
          stopRecordingRef.current('silence');
          return;
        }
      } else if (silenceStartTimeRef.current !== null) {
        log('🔊 Voice Hook: Voice detected, resetting silence timer');
        silenceStartTimeRef.current = null;
      }
    }

    // Max recording time guard
    if (recordingDuration > MAX_RECORDING_TIME_MS) {
      log('📢 Voice Hook: Max recording time reached');
      stopRecordingRef.current('max_time');
      return;
    }

    // Gereksiz re-render'dan kaçınmak için anlamlı değişiklikler olmadıkça
    // setState çağırma. 60fps'de her frame setState absurd olur.
    setState((prev) => {
      const volumeDelta = Math.abs(prev.volume - smoothedVolume);
      const durationDelta = Math.abs(prev.recordingDuration - recordingDuration);
      const silenceChanged = prev.silenceDetected !== silenceDetected;

      if (volumeDelta < 0.01 && durationDelta < 100 && !silenceChanged) {
        return prev;
      }

      return {
        ...prev,
        volume: smoothedVolume,
        silenceDetected,
        recordingDuration,
      };
    });

    animationFrameRef.current = requestAnimationFrame(updateVolume);
  };

  // ─── STT (speech-to-text) ─────────────────────────────────────────────
  const processAudio = useCallback(
    async (audioBlob: Blob) => {
      log('🔄 Voice Hook: Processing audio blob...', audioBlob.size, 'bytes');

      try {
        if (audioBlob.size === 0) {
          throw new Error('Audio blob is empty');
        }

        // Circular import'u önlemek için dinamik import
        const { apiService } = await import('@/services/api');
        const transcript = await apiService.convertSpeechToText(
          audioBlob,
          language,
        );

        if (transcript && transcript.trim().length > 0) {
          log('✅ Voice Hook: Transcript received');
          setState((prev) => ({
            ...prev,
            isProcessing: false,
            transcript: transcript.trim(),
          }));
        } else {
          setState((prev) => ({
            ...prev,
            isProcessing: false,
            error:
              language === 'en'
                ? 'No speech detected. Please try speaking louder.'
                : 'Konuşma algılanamadı. Lütfen daha yüksek sesle konuşun.',
          }));
        }
      } catch (error) {
        logError('❌ Voice Hook: Processing error:', error);
        setState((prev) => ({
          ...prev,
          isProcessing: false,
          error:
            language === 'en'
              ? 'Failed to process audio. Please try again.'
              : 'Ses işlenemedi. Lütfen tekrar deneyin.',
        }));
      }
    },
    [language],
  );

  // ─── Stop ─────────────────────────────────────────────────────────────
  const stopRecording = useCallback((reason?: StopReason) => {
    if (!isRecordingRef.current) {
      log('Voice Hook: stopRecording called but not recording, ignoring');
      return;
    }

    log('🛑 Voice Hook: Stopping recording, reason:', reason || 'manual');

    // Ref'i hemen false yap ki updateVolume döngüsü bir sonraki frame'de
    // kendini durdursun.
    isRecordingRef.current = false;

    try {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== 'inactive'
      ) {
        // onstop handler processAudio'yu tetikleyecek
        mediaRecorderRef.current.stop();
      }
    } catch (err) {
      logError('Voice Hook: MediaRecorder stop error:', err);
    }

    setState((prev) => ({
      ...prev,
      isRecording: false,
      isProcessing: true,
      silenceDetected: false,
      autoStoppedReason: reason,
    }));

    cleanupAudioResources();
  }, [cleanupAudioResources]);

  // Her render'da ref'i güncelle — updateVolume güncel closure'ı çağırabilsin
  stopRecordingRef.current = stopRecording;

  // ─── Start ────────────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    if (isRecordingRef.current) {
      log('Voice Hook: Already recording, ignoring start');
      return;
    }

    try {
      log('🎤 Voice Hook: Starting recording...');

      // Önceki bir hatayı veya otomatik durdurma nedenini temizle
      setState((prev) => ({
        ...prev,
        error: undefined,
        transcript: '',
        autoStoppedReason: undefined,
      }));

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100,
        },
      });

      streamRef.current = stream;
      recordingStartTimeRef.current = Date.now();
      silenceStartTimeRef.current = null;
      volumeHistoryRef.current = [];

      // AudioContext setup
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.3;
      source.connect(analyser);
      analyserRef.current = analyser;

      // MediaRecorder setup — tarayıcı uyumluluğu için MIME type fallback
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : MediaRecorder.isTypeSupported('audio/mp4')
            ? 'audio/mp4'
            : '';

      const mediaRecorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined,
      );
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mimeType || 'audio/webm',
        });
        if (audioBlob.size > 0) {
          processAudio(audioBlob);
        } else {
          setState((prev) => ({
            ...prev,
            isProcessing: false,
            error:
              language === 'en'
                ? 'No audio data captured'
                : 'Ses verisi yakalanamadı',
          }));
        }
      };

      mediaRecorder.onerror = (event) => {
        logError('❌ Voice Hook: MediaRecorder error:', event);
        isRecordingRef.current = false;
        cleanupAudioResources();
        setState((prev) => ({
          ...prev,
          error: 'Recording error occurred',
          isRecording: false,
        }));
      };

      mediaRecorder.start(1000);

      // ÖNEMLİ: ref'i state'ten önce set ediyoruz ki updateVolume'un ilk
      // frame'i isRecordingRef.current === true görsün ve döngüyü sürdürsün.
      isRecordingRef.current = true;

      setState((prev) => ({
        ...prev,
        isRecording: true,
        volume: 0,
        silenceDetected: false,
        recordingDuration: 0,
      }));

      updateVolume();

      log('✅ Voice Hook: Recording started');
    } catch (error) {
      logError('❌ Voice Hook: Failed to start recording:', error);
      isRecordingRef.current = false;
      cleanupAudioResources();
      setState((prev) => ({
        ...prev,
        error:
          language === 'en'
            ? 'Microphone access denied or unavailable'
            : 'Mikrofon erişimi reddedildi veya kullanılamıyor',
        isRecording: false,
      }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language, processAudio, cleanupAudioResources]);

  // ─── Reset ────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    log('🔄 Voice Hook: Resetting');

    if (isRecordingRef.current) {
      isRecordingRef.current = false;
      try {
        if (
          mediaRecorderRef.current &&
          mediaRecorderRef.current.state !== 'inactive'
        ) {
          mediaRecorderRef.current.stop();
        }
      } catch {
        // ignore
      }
    }

    cleanupAudioResources();
    recordingStartTimeRef.current = 0;

    setState({
      isRecording: false,
      isProcessing: false,
      transcript: '',
      volume: 0,
      silenceDetected: false,
      recordingDuration: 0,
    });
  }, [cleanupAudioResources]);

  // ─── Unmount cleanup ──────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      log('🧹 Voice Hook: Cleaning up on unmount');
      isRecordingRef.current = false;
      try {
        if (
          mediaRecorderRef.current &&
          mediaRecorderRef.current.state === 'recording'
        ) {
          mediaRecorderRef.current.stop();
        }
      } catch {
        // ignore
      }
      cleanupAudioResources();
    };
  }, [cleanupAudioResources]);

  return {
    ...state,
    startRecording,
    // Dışarıdan manuel çağrı için reason argümanını expose etmiyoruz
    stopRecording: () => stopRecording(),
    reset,
  };
};