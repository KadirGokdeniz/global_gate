import { useState, useCallback, useRef, useEffect } from 'react';
import { VoiceRecordingState } from '@/types';

export const useVoiceRecording = (language: 'en' | 'tr' = 'en') => {
  const [state, setState] = useState<VoiceRecordingState>({
    isRecording: false,
    isProcessing: false,
    transcript: '',
    volume: 0
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const volumeAnalyzerRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number>();
  
  // ✅ NEW: Enhanced volume detection refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const lastVolumeUpdateRef = useRef<number>(0);
  const volumeHistoryRef = useRef<number[]>([]);
  
  // ✅ NEW: Auto-stop functionality refs
  const silenceStartTimeRef = useRef<number | null>(null);
  const recordingStartTimeRef = useRef<number>(0);

  // ✅ ENHANCED: Better volume calculation using time domain data
  const updateVolume = useCallback(() => {
    if (!volumeAnalyzerRef.current) return;

    // Use time domain data for more accurate voice activity detection
    const dataArray = new Uint8Array(volumeAnalyzerRef.current.fftSize);
    volumeAnalyzerRef.current.getByteTimeDomainData(dataArray);
    
    // Calculate RMS (Root Mean Square) for better volume representation
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const sample = (dataArray[i] - 128) / 128; // Normalize to -1 to 1
      sum += sample * sample;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    
    // Apply smoothing and convert to 0-1 range
    const currentVolume = Math.min(1, rms * 10); // Amplify and cap at 1
    
    // ✅ NEW: Volume smoothing with history
    volumeHistoryRef.current.push(currentVolume);
    if (volumeHistoryRef.current.length > 5) {
      volumeHistoryRef.current.shift();
    }
    
    const smoothedVolume = volumeHistoryRef.current.reduce((a, b) => a + b, 0) / volumeHistoryRef.current.length;
    
    setState(prev => ({ ...prev, volume: smoothedVolume }));
    lastVolumeUpdateRef.current = Date.now();
    
    // ✅ NEW: Auto-stop logic based on silence detection
    const SILENCE_THRESHOLD = 0.01;
    const SILENCE_DURATION = 2000; // 2 seconds
    const MIN_RECORDING_TIME = 1000; // 1 second minimum
    const MAX_RECORDING_TIME = 30000; // 30 seconds maximum
    
    const recordingDuration = Date.now() - recordingStartTimeRef.current;
    
    // Check for maximum recording time
    if (recordingDuration > MAX_RECORDING_TIME) {
      console.log('📢 Voice Hook: Max recording time reached, stopping...');
      stopRecording();
      return;
    }
    
    // Only check silence after minimum recording time
    if (recordingDuration > MIN_RECORDING_TIME) {
      if (smoothedVolume < SILENCE_THRESHOLD) {
        if (silenceStartTimeRef.current === null) {
          silenceStartTimeRef.current = Date.now();
          console.log('🔇 Voice Hook: Silence started');
        } else {
          const silenceDuration = Date.now() - silenceStartTimeRef.current;
          if (silenceDuration >= SILENCE_DURATION) {
            console.log('⏹️ Voice Hook: Silence duration exceeded, auto-stopping...');
            stopRecording();
            return;
          }
        }
      } else {
        // Voice detected, reset silence timer
        if (silenceStartTimeRef.current !== null) {
          console.log('🔊 Voice Hook: Voice detected, resetting silence timer');
          silenceStartTimeRef.current = null;
        }
      }
    }
    
    if (state.isRecording) {
      animationFrameRef.current = requestAnimationFrame(updateVolume);
    }
  }, [state.isRecording]);

  const startRecording = useCallback(async () => {
    try {
      console.log('🎤 Voice Hook: Starting recording...');
      setState(prev => ({ ...prev, error: undefined }));
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true, // ✅ NEW: Better audio quality
          sampleRate: 44100
        } 
      });
      
      streamRef.current = stream;
      recordingStartTimeRef.current = Date.now();
      silenceStartTimeRef.current = null;
      volumeHistoryRef.current = [];

      // ✅ ENHANCED: Better audio context setup
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      
      // ✅ ENHANCED: Better analyzer settings for voice detection
      analyser.fftSize = 1024; // Higher resolution for better voice detection
      analyser.smoothingTimeConstant = 0.3; // Less smoothing for more responsive detection
      
      source.connect(analyser);
      volumeAnalyzerRef.current = analyser;

      // Start volume monitoring
      updateVolume();

      // ✅ ENHANCED: Better MediaRecorder setup
      const options: MediaRecorderOptions = {};
      
      // Try different MIME types for better compatibility
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        options.mimeType = 'audio/webm;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/webm')) {
        options.mimeType = 'audio/webm';
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        options.mimeType = 'audio/mp4';
      }
      
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
          console.log(`📦 Voice Hook: Audio chunk received: ${event.data.size} bytes`);
        }
      };

      mediaRecorder.onstop = () => {
        console.log('🛑 Voice Hook: MediaRecorder stopped, processing audio...');
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: options.mimeType || 'audio/webm' 
        });
        console.log(`📦 Voice Hook: Final audio blob: ${audioBlob.size} bytes`);
        processAudio(audioBlob);
      };

      mediaRecorder.onerror = (event) => {
        console.error('❌ Voice Hook: MediaRecorder error:', event);
        setState(prev => ({ 
          ...prev, 
          error: 'Recording error occurred',
          isRecording: false 
        }));
      };

      // ✅ NEW: Record in smaller chunks for better processing
      mediaRecorder.start(1000); // 1 second chunks
      setState(prev => ({ ...prev, isRecording: true }));
      
      console.log('✅ Voice Hook: Recording started successfully');
      
    } catch (error) {
      console.error('❌ Voice Hook: Failed to start recording:', error);
      setState(prev => ({ 
        ...prev, 
        error: 'Microphone access denied or unavailable',
        isRecording: false 
      }));
    }
  }, [updateVolume]);

  const stopRecording = useCallback(() => {
    console.log('🛑 Voice Hook: Stopping recording...');
    
    if (mediaRecorderRef.current && state.isRecording) {
      mediaRecorderRef.current.stop();
      setState(prev => ({ ...prev, isRecording: false, isProcessing: true }));
      
      // Stop volume monitoring
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      // ✅ ENHANCED: Proper cleanup
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      
      // Stop media stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
          console.log('🔇 Voice Hook: Audio track stopped');
        });
        streamRef.current = null;
      }
      
      // Reset timers
      silenceStartTimeRef.current = null;
      volumeHistoryRef.current = [];
      
      console.log('✅ Voice Hook: Recording stopped successfully');
    }
  }, [state.isRecording]);

  // ✅ ENHANCED: Better audio processing with more error handling
  const processAudio = useCallback(async (audioBlob: Blob) => {
    console.log('🔄 Voice Hook: Processing audio blob...');
    
    try {
      if (audioBlob.size === 0) {
        throw new Error('Audio blob is empty');
      }
      
      // Use the API service for speech-to-text conversion
      const { apiService } = await import('@/services/api');
      console.log('📡 Voice Hook: Calling STT API...');
      
      const transcript = await apiService.convertSpeechToText(audioBlob, language);
      
      if (transcript && transcript.trim().length > 0) {
        console.log('✅ Voice Hook: Transcript received:', transcript.substring(0, 50) + '...');
        setState(prev => ({ 
          ...prev, 
          isProcessing: false,
          transcript: transcript.trim()
        }));
      } else {
        console.log('⚠️ Voice Hook: No transcript received');
        setState(prev => ({ 
          ...prev, 
          isProcessing: false,
          error: language === 'en' ? 
            'No speech detected. Please try speaking louder.' :
            'Konuşma algılanamadı. Lütfen daha yüksek sesle konuşun.'
        }));
      }
      
    } catch (error) {
      console.error('❌ Voice Hook: Processing error:', error);
      setState(prev => ({ 
        ...prev, 
        isProcessing: false,
        error: language === 'en' ? 
          'Failed to process audio. Please try again.' :
          'Ses işlenemedi. Lütfen tekrar deneyin.'
      }));
    }
  }, [language]);

  const reset = useCallback(() => {
    console.log('🔄 Voice Hook: Resetting...');
    
    setState({
      isRecording: false,
      isProcessing: false,
      transcript: '',
      volume: 0
    });
    
    // Clean up any ongoing recording
    if (mediaRecorderRef.current && state.isRecording) {
      mediaRecorderRef.current.stop();
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    // Reset refs
    silenceStartTimeRef.current = null;
    volumeHistoryRef.current = [];
    recordingStartTimeRef.current = 0;
    
    console.log('✅ Voice Hook: Reset completed');
  }, [state.isRecording]);

  // ✅ ENHANCED: Better cleanup on unmount
  useEffect(() => {
    return () => {
      console.log('🧹 Voice Hook: Cleaning up on unmount...');
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  return {
    ...state,
    startRecording,
    stopRecording,
    reset
  };
};