import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ThumbsUp, ThumbsDown, Clock, AlertTriangle, Volume2, ExternalLink } from 'lucide-react';
import { Message, FeedbackType, Language } from '@/types';
import { useState } from 'react';

interface ResponseCardProps {
  message: Message;
  language: Language;
  onFeedback: (messageId: string, feedback: FeedbackType) => void;
  onPlayAudio?: (text: string) => void;
  feedbackGiven?: FeedbackType | null;
}

export const ResponseCard = ({ 
  message, 
  language, 
  onFeedback, 
  onPlayAudio,
  feedbackGiven 
}: ResponseCardProps) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState<FeedbackType | null>(null);

  // ✅ DÜZELTILMIŞ: Enhanced audio handling with comprehensive debugging
  const handlePlayAudio = async () => {
    console.log('🔊 TTS: Audio button clicked');
    
    if (!onPlayAudio) {
      console.error('❌ TTS: onPlayAudio function not provided');
      return;
    }

    try {
      setIsPlaying(true);
      console.log('🔊 TTS: Requesting audio URL...');
      
      const audioUrl = await onPlayAudio(message.answer);
      
      if (!audioUrl) {
        console.error('❌ TTS: No audio URL returned');
        return;
      }
      
      console.log('✅ TTS: Audio URL received:', audioUrl.substring(0, 50) + '...');
      
      // Create and configure audio
      const audio = new Audio(audioUrl);
      
      // Add event listeners for debugging
      audio.addEventListener('loadstart', () => {
        console.log('🔊 TTS: Audio loading started');
      });
      
      audio.addEventListener('canplay', () => {
        console.log('✅ TTS: Audio can play');
      });
      
      audio.addEventListener('play', () => {
        console.log('▶️ TTS: Audio playback started');
      });
      
      audio.addEventListener('ended', () => {
        console.log('⏹️ TTS: Audio playback ended');
        setIsPlaying(false);
      });
      
      audio.addEventListener('error', (e) => {
        console.error('❌ TTS: Audio error:', e);
        console.error('Audio error details:', {
          code: audio.error?.code,
          message: audio.error?.message
        });
        setIsPlaying(false);
      });
      
      // Attempt to play
      console.log('🔊 TTS: Attempting to play audio...');
      const playPromise = audio.play();
      
      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            console.log('✅ TTS: Audio play promise resolved');
            // Set timeout to reset playing state if ended event doesn't fire
            setTimeout(() => {
              if (isPlaying) {
                console.log('⏱️ TTS: Timeout - resetting playing state');
                setIsPlaying(false);
              }
            }, 10000); // 10 second safety timeout
          })
          .catch((error) => {
            console.error('❌ TTS: Audio play promise rejected:', error);
            setIsPlaying(false);
            
            // Check for common browser audio policy issues
            if (error.name === 'NotAllowedError') {
              console.error('🚫 TTS: Browser blocked audio - user interaction may be required');
            }
          });
      }
      
    } catch (error) {
      console.error('❌ TTS: General error in handlePlayAudio:', error);
      setIsPlaying(false);
    }
  };

  // ✅ DÜZELTILDI: Debug ve loading state eklenmiş feedback handler
  const handleFeedbackClick = async (type: FeedbackType) => {
    console.log('🔔 Feedback button clicked:', { messageId: message.id, type, feedbackGiven });
    
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
    const isDisabled = feedbackGiven !== null && feedbackGiven !== type;
    const isLoading = feedbackLoading === type;
    
    return (
      <Button
        variant={isSelected ? "default" : "outline"}
        size="sm"
        onClick={() => {
          // BASIT TEST: Bu log görünüyor mu?
          console.log('BASIC TEST: Button clicked!', type);
          handleFeedbackClick(type);
        }}
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
            
            {/* Audio Button */}
            {onPlayAudio && (
              <Button
                variant="outline"
                size="lg"
                onClick={handlePlayAudio}
                disabled={isPlaying}
                className="shrink-0 h-12 w-12 rounded-full"
                title={language === 'en' ? 'Listen to answer' : 'Cevabı dinle'}
              >
                <Volume2 className={`w-5 h-5 ${isPlaying ? 'animate-pulse' : ''}`} />
              </Button>
            )}
          </div>

          {/* Metadata */}
          <div className="flex flex-wrap gap-2 mb-6">
            <Badge variant="secondary" className="px-3 py-1">
              {message.provider} • {message.model}
            </Badge>
            {message.airline_preference && (
              <Badge variant="outline" className="px-3 py-1">
                {message.airline_preference === 'thy' ? '🇹🇷 Turkish Airlines' : 
                 message.airline_preference === 'pegasus' ? '✈️ Pegasus Airlines' : 
                 '🌍 All Airlines'}
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

          {/* ✅ GELİŞTİRİLDİ: Enhanced Feedback Section */}
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

            {/* ✅ DÜZELTILDI: Enhanced Feedback Status */}
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

            {/* ✅ YENİ: Debug Info (sadece development için) */}
            {process.env.NODE_ENV === 'development' && (
              <div className="mt-2 text-center">
                <details className="text-xs text-muted-foreground">
                  <summary className="cursor-pointer">Debug Info</summary>
                  <div className="mt-2 p-2 bg-muted/20 rounded text-left">
                    <div>Message ID: {message.id}</div>
                    <div>Feedback Given: {feedbackGiven || 'none'}</div>
                    <div>Loading: {feedbackLoading || 'none'}</div>
                  </div>
                </details>
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
              {message.sources.slice(0, 5).map((source, index) => (
                <div key={index} className="p-3 bg-muted/20 rounded-lg border-l-4 border-primary/50">
                  <div className="flex justify-between items-start gap-2 mb-1">
                    <span className="font-medium text-sm">{source.source}</span>
                    <Badge variant="outline" className="text-xs">
                      {(source.similarity_score * 100).toFixed(1)}% {language === 'en' ? 'match' : 'eşleşme'}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {source.airline}
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