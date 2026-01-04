// ✅ ResponseCard.tsx - TTS Control mekanizması eklendi
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ThumbsUp, ThumbsDown, Clock, AlertTriangle, Volume2, VolumeX, Play, Pause, Square, ExternalLink } from 'lucide-react';
import { Message, FeedbackType, Language } from '@/types';
import { useState, useRef, useEffect } from 'react';

interface ResponseCardProps {
  message: Message;
  language: Language;
  onFeedback: (messageId: string, feedback: FeedbackType) => void;
  onPlayAudio?: (text: string) => Promise<string | null>;
  feedbackGiven?: FeedbackType | null;
}

export const ResponseCard = ({ 
  message, 
  language, 
  onFeedback, 
  onPlayAudio,
  feedbackGiven 
}: ResponseCardProps) => {
  // ✅ FIX 1: Audio state management genişletildi
  const [audioState, setAudioState] = useState<'idle' | 'loading' | 'playing' | 'paused' | 'error'>('idle');
  const [audioProgress, setAudioProgress] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [feedbackLoading, setFeedbackLoading] = useState<FeedbackType | null>(null);
  const [expandedSources, setExpandedSources] = useState<number[]>([]);
  const toggleSource = (index: number) => {
    setExpandedSources(prev =>
      prev.includes(index)
        ? prev.filter(i => i !== index)
        : [...prev, index]
    );
  };
  // ✅ FIX 2: Audio element referansı saklanıyor
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // ✅ FIX 3: Component unmount'ta cleanup
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
        audioRef.current = null;
      }
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  // ✅ FIX 4: Progress tracking fonksiyonu
  const startProgressTracking = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }
    
    progressIntervalRef.current = setInterval(() => {
      if (audioRef.current) {
        const currentTime = audioRef.current.currentTime;
        const duration = audioRef.current.duration;
        
        if (duration > 0) {
          setAudioProgress((currentTime / duration) * 100);
        }
      }
    }, 100);
  };

  const stopProgressTracking = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  // ✅ FIX 5: Enhanced audio control fonksiyonları
  const handlePlayAudio = async () => {
    console.log('🔊 TTS: Play button clicked, current state:', audioState);
    
    if (!onPlayAudio) {
      console.error('❌ TTS: onPlayAudio function not provided');
      return;
    }

    try {
      // Eğer zaten bir audio varsa ve paused durumundaysa, resume et
      if (audioRef.current && audioState === 'paused') {
        console.log('▶️ TTS: Resuming paused audio');
        audioRef.current.play();
        setAudioState('playing');
        startProgressTracking();
        return;
      }

      // Yeni audio için loading state'i
      setAudioState('loading');
      console.log('🔊 TTS: Requesting new audio URL...');
      
      const audioUrl = await onPlayAudio(message.answer);
      
      if (!audioUrl) {
        console.error('❌ TTS: No audio URL returned');
        setAudioState('error');
        return;
      }
      
      console.log('✅ TTS: Audio URL received, creating audio element');
      
      // Eski audio'yu temizle
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
      }
      
      // Yeni audio element oluştur
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      
      // ✅ FIX 6: Comprehensive audio event listeners
      audio.addEventListener('loadstart', () => {
        console.log('🔊 TTS: Audio loading started');
        setAudioState('loading');
      });
      
      audio.addEventListener('canplay', () => {
        console.log('✅ TTS: Audio can play');
      });
      
      audio.addEventListener('loadedmetadata', () => {
        console.log('📊 TTS: Audio metadata loaded, duration:', audio.duration);
        setAudioDuration(audio.duration);
        setAudioProgress(0);
      });
      
      audio.addEventListener('play', () => {
        console.log('▶️ TTS: Audio playback started');
        setAudioState('playing');
        startProgressTracking();
      });
      
      audio.addEventListener('pause', () => {
        console.log('⏸️ TTS: Audio paused');
        setAudioState('paused');
        stopProgressTracking();
      });
      
      audio.addEventListener('ended', () => {
        console.log('⏹️ TTS: Audio playback ended');
        setAudioState('idle');
        setAudioProgress(0);
        stopProgressTracking();
      });
      
      audio.addEventListener('error', (e) => {
        console.error('❌ TTS: Audio error:', e);
        console.error('Audio error details:', {
          code: audio.error?.code,
          message: audio.error?.message
        });
        setAudioState('error');
        stopProgressTracking();
      });
      
      // Audio'yu oynat
      console.log('🔊 TTS: Starting playback...');
      const playPromise = audio.play();
      
      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            console.log('✅ TTS: Audio play promise resolved');
          })
          .catch((error) => {
            console.error('❌ TTS: Audio play promise rejected:', error);
            setAudioState('error');
            
            if (error.name === 'NotAllowedError') {
              console.error('🚫 TTS: Browser blocked audio - user interaction required');
            }
          });
      }
      
    } catch (error) {
      console.error('❌ TTS: General error in handlePlayAudio:', error);
      setAudioState('error');
    }
  };

  // ✅ FIX 7: Pause fonksiyonu
  const handlePauseAudio = () => {
    console.log('⏸️ TTS: Pause button clicked');
    if (audioRef.current && audioState === 'playing') {
      audioRef.current.pause();
    }
  };

  // ✅ FIX 8: Stop fonksiyonu
  const handleStopAudio = () => {
    console.log('⏹️ TTS: Stop button clicked');
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setAudioState('idle');
      setAudioProgress(0);
      stopProgressTracking();
    }
  };

  // ✅ FIX 9: Progress seek fonksiyonu
  const handleSeekAudio = (percentage: number) => {
    if (audioRef.current && audioDuration > 0) {
      const newTime = (percentage / 100) * audioDuration;
      audioRef.current.currentTime = newTime;
      setAudioProgress(percentage);
    }
  };

  // ✅ FIX 10: Format time helper
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleFeedbackClick = async (type: FeedbackType) => {
    console.log('📝 Feedback button clicked:', { messageId: message.id, type, feedbackGiven });
    
    try {
      setFeedbackLoading(type);
      await onFeedback(message.id, type);
      console.log('✅ Feedback sent successfully:', type);
    } catch (error) {
      console.error('❌ Feedback error:', error);
    } finally {
      setFeedbackLoading(null);
    }
  };

  const getFeedbackButton = (type: FeedbackType, icon: React.ReactNode, label: string) => {
    const isSelected = feedbackGiven === type;
    const isDisabled = (feedbackGiven !== null && feedbackGiven !== undefined) && feedbackGiven !== type;
    const isLoading = feedbackLoading === type;
    
    return (
      <Button
        variant={isSelected ? "default" : "outline"}
        size="sm"
        onClick={() => handleFeedbackClick(type)}
        disabled={isDisabled || isLoading}
        className={`text-xs transition-all duration-200 ${
          isSelected ? 'ring-2 ring-primary/50' : ''
        } ${isLoading ? 'opacity-50' : ''}`}
      >
        {isLoading ? (
          <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : (
          icon
        )}
        <span className="ml-2">{label}</span>
        {isSelected && (
          <div className="ml-1 w-2 h-2 bg-current rounded-full opacity-60" />
        )}
      </Button>
    );
  };

  return (
    <div className="w-full max-w-4xl mx-auto animate-fade-in-up">
      {/* Question Display */}
      <div className="mb-4">
        <h2 className="text-xl font-medium text-muted-foreground">
          <span className="text-primary font-semibold">Q:</span> {message.question}
        </h2>
      </div>

      {/* Answer Card */}
      <Card className="glass-card shadow-xl border-0">
        <CardContent className="p-8">
          {/* Answer Content */}
          <div className="flex items-start justify-between gap-4 mb-6">
            <div className="flex-1">
              <h3 className="text-2xl font-semibold mb-4 text-foreground">
                {language === 'en' ? 'Answer' : 'Cevap'}
              </h3>
              <p className="text-lg leading-relaxed text-foreground/90">
                {message.answer}
              </p>
            </div>
          </div>

          {/* ✅ FIX 11: Enhanced Audio Controls Panel */}
          {onPlayAudio && (
            <div className="mb-6 p-4 bg-muted/20 rounded-xl border border-muted/50">
              <div className="flex items-center justify-between gap-4 mb-3">
                <h4 className="font-medium flex items-center gap-2">
                  <Volume2 className="w-4 h-4" />
                  {language === 'en' ? 'Audio Player' : 'Ses Oynatıcı'}
                </h4>
                
                {/* Audio Control Buttons */}
                <div className="flex items-center gap-2">
                  {audioState === 'idle' || audioState === 'error' ? (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handlePlayAudio}
                      disabled={audioState === 'loading'}
                      className="h-10 px-4"
                    >
                      {audioState === 'loading' ? (
                        <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <Play className="w-4 h-4" />
                      )}
                      <span className="ml-2">
                        {language === 'en' ? 'Play' : 'Oynat'}
                      </span>
                    </Button>
                  ) : (
                    <>
                      {audioState === 'playing' ? (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handlePauseAudio}
                          className="h-10 px-4"
                        >
                          <Pause className="w-4 h-4" />
                          <span className="ml-2">
                            {language === 'en' ? 'Pause' : 'Duraklat'}
                          </span>
                        </Button>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handlePlayAudio}
                          className="h-10 px-4"
                        >
                          <Play className="w-4 h-4" />
                          <span className="ml-2">
                            {language === 'en' ? 'Resume' : 'Devam'}
                          </span>
                        </Button>
                      )}
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleStopAudio}
                        className="h-10 px-4"
                      >
                        <Square className="w-4 h-4" />
                        <span className="ml-2">
                          {language === 'en' ? 'Stop' : 'Dur'}
                        </span>
                      </Button>
                    </>
                  )}
                </div>
              </div>

              {/* ✅ FIX 12: Progress Bar */}
              {audioState !== 'idle' && audioState !== 'error' && (
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-muted-foreground min-w-[40px]">
                      {formatTime((audioProgress / 100) * audioDuration)}
                    </span>
                    
                    {/* Interactive Progress Bar */}
                    <div 
                      className="flex-1 h-2 bg-muted rounded-full overflow-hidden cursor-pointer"
                      onClick={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const percentage = ((e.clientX - rect.left) / rect.width) * 100;
                        handleSeekAudio(Math.max(0, Math.min(100, percentage)));
                      }}
                    >
                      <div 
                        className="h-full bg-primary transition-all duration-150 ease-out rounded-full"
                        style={{ width: `${audioProgress}%` }}
                      />
                    </div>
                    
                    <span className="text-xs text-muted-foreground min-w-[40px]">
                      {formatTime(audioDuration)}
                    </span>
                  </div>
                  
                  {/* Audio State Indicator */}
                  <div className="flex items-center justify-center">
                    <div className={`flex items-center gap-2 text-xs px-3 py-1 rounded-full ${
                      audioState === 'playing' ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-300' :
                      audioState === 'paused' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-300' :
                      audioState === 'loading' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300' :
                      'bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-300'
                    }`}>
                      <div className={`w-2 h-2 rounded-full ${
                        audioState === 'playing' ? 'bg-green-500 animate-pulse' :
                        audioState === 'paused' ? 'bg-yellow-500' :
                        audioState === 'loading' ? 'bg-blue-500 animate-pulse' :
                        'bg-red-500'
                      }`} />
                      <span className="font-medium capitalize">
                        {audioState === 'playing' ? (language === 'en' ? 'Playing' : 'Oynatılıyor') :
                         audioState === 'paused' ? (language === 'en' ? 'Paused' : 'Duraklatıldı') :
                         audioState === 'loading' ? (language === 'en' ? 'Loading' : 'Yükleniyor') :
                         (language === 'en' ? 'Error' : 'Hata')}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Error State */}
              {audioState === 'error' && (
                <div className="text-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                  <div className="flex items-center justify-center gap-2 text-red-700 dark:text-red-300">
                    <VolumeX className="w-4 h-4" />
                    <span className="text-sm font-medium">
                      {language === 'en' ? 'Audio playback failed' : 'Ses oynatma başarısız'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Rest of the existing content - Metadata, Statistics, Feedback, etc. */}
          {/* Metadata */}
          <div className="flex flex-wrap gap-2 mb-6">
            <Badge variant="secondary" className="px-3 py-1">
              {message.provider} • {message.model}
            </Badge>
            {message.airline_preference && (
              <Badge variant="outline" className="px-3 py-1">
                {message.airline_preference === 'thy' ? '🇹🇷 Turkish Airlines' : 
                 message.airline_preference === 'pegasus' ? '✈️ Pegasus Airlines' : 
                 '🌐 All Airlines'}
              </Badge>
            )}
            {message.sources && (
              <Badge variant="outline" className="px-3 py-1">
                {message.sources.length} {language === 'en' ? 'sources' : 'kaynak'}
              </Badge>
            )}
          </div>

          {/* Statistics */}
          {message.stats && (
            <div className="grid grid-cols-3 gap-6 p-4 bg-muted/30 rounded-xl mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-primary">{message.stats.total_retrieved}</div>
                <div className="text-sm text-muted-foreground">
                  {language === 'en' ? 'Sources Found' : 'Bulunan Kaynak'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-accent">
                  {(message.stats.avg_similarity * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-muted-foreground">
                  {language === 'en' ? 'Avg Similarity' : 'Ort. Benzerlik'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600 capitalize">
                  {language === 'tr' ? 
                    (message.stats.context_quality === 'high' ? 'Yüksek' :
                     message.stats.context_quality === 'medium' ? 'Orta' : 'Düşük') :
                    message.stats.context_quality}
                </div>
                <div className="text-sm text-muted-foreground">
                  {language === 'en' ? 'Quality' : 'Kalite'}
                </div>
              </div>
            </div>
          )}

          {/* Feedback Section */}
          <div className="space-y-4">
            <div className="text-center">
              <h4 className="text-sm font-medium text-muted-foreground mb-3">
                {language === 'en' ? 'How was this response?' : 'Bu yanıt nasıldı?'}
              </h4>
            </div>
            
            <div className="flex flex-wrap gap-3 justify-center">
              {getFeedbackButton(
                'helpful',
                <ThumbsUp className="w-4 h-4" />,
                language === 'en' ? 'Helpful' : 'Yardımcı'
              )}
              {getFeedbackButton(
                'not_helpful',
                <ThumbsDown className="w-4 h-4" />,
                language === 'en' ? 'Not Helpful' : 'Faydalı Değil'
              )}
              {getFeedbackButton(
                'too_slow',
                <Clock className="w-4 h-4" />,
                language === 'en' ? 'Too Slow' : 'Çok Yavaş'
              )}
              {getFeedbackButton(
                'incorrect',
                <AlertTriangle className="w-4 h-4" />,
                language === 'en' ? 'Incorrect' : 'Yanlış'
              )}
            </div>

            {/* Feedback Status */}
            {feedbackGiven && (
              <div className="mt-4 text-center">
                <div className="inline-flex items-center gap-2 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 px-4 py-2 rounded-full border border-green-200 dark:border-green-800">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm font-medium">
                    {language === 'en' ? 'Thank you for your feedback!' : 'Geri bildiriminiz için teşekkürler!'}
                  </span>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Sources Section */}
      {message.sources && message.sources.length > 0 && (
        <Card className="mt-6 border-0 shadow-lg">
          <CardContent className="p-6">
            <h4 className="font-semibold mb-4 flex items-center gap-2">
              <ExternalLink className="w-4 h-4" />
              {language === 'en' ? `Sources (${message.sources.length})` : `Kaynaklar (${message.sources.length})`}
            </h4>
            <div className="grid gap-3">
              {message.sources.map((source, index) => (
                <div key={index} className="p-3 bg-muted/20 rounded-lg border-l-4 border-primary/50">
                  <div className="flex justify-between items-start gap-2 mb-1">
                    <span className="font-medium text-sm">{source.source}</span>
                    <Badge variant="outline" className="text-xs">
                      {(source.similarity_score * 100).toFixed(1)}% {language === 'en' ? 'match' : 'eşleşme'}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {source.airline}
                    <button
                      onClick={() => toggleSource(index)}
                      className="mt-2 text-xs text-primary underline"
                    >
                      {expandedSources.includes(index)
                        ? 'Hide source text'
                        : 'Show source text'}
                    </button>
                    {expandedSources.includes(index) && (
                    <div className="mt-3 p-3 text-sm bg-slate-100 dark:bg-slate-800 rounded-md whitespace-pre-wrap">
                      {source.content_full ||
                        source.content_preview ||
                        'No source text available.'}
                    </div>
                      )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};