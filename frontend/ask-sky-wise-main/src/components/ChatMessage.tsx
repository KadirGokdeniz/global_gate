import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ThumbsUp, ThumbsDown, Clock, AlertTriangle, Volume2 } from 'lucide-react';
import { Message, FeedbackType, Language } from '@/types';
import { useState } from 'react';

interface ChatMessageProps {
  message: Message;
  language: Language;
  onFeedback: (messageId: string, feedback: FeedbackType) => void;
  onPlayAudio?: (text: string) => void;
  feedbackGiven?: FeedbackType | null;
}

export const ChatMessage = ({ 
  message, 
  language, 
  onFeedback, 
  onPlayAudio,
  feedbackGiven 
}: ChatMessageProps) => {
  const [isPlaying, setIsPlaying] = useState(false);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString(language === 'tr' ? 'tr-TR' : 'en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handlePlayAudio = () => {
    if (onPlayAudio) {
      setIsPlaying(true);
      onPlayAudio(message.answer);
      // Reset playing state after a delay (you might want to track actual audio duration)
      setTimeout(() => setIsPlaying(false), 3000);
    }
  };

  const getFeedbackButton = (type: FeedbackType, icon: React.ReactNode, label: string) => (
    <Button
      variant={feedbackGiven === type ? "default" : "outline"}
      size="sm"
      onClick={() => onFeedback(message.id, type)}
      disabled={feedbackGiven !== null && feedbackGiven !== type}
      className="text-xs h-8"
    >
      {icon}
      {label}
    </Button>
  );

  return (
    <Card className="glass-card animate-fade-in-up">
      <CardContent className="p-6 space-y-4">
        {/* Question */}
        <div className="border-l-4 border-primary pl-4">
          <p className="font-medium text-foreground">
            <span className="text-primary font-semibold">Q:</span> {message.question}
          </p>
        </div>

        {/* Answer */}
        <div className="border-l-4 border-accent pl-4">
          <div className="flex items-start justify-between gap-4">
            <p className="text-foreground flex-1">
              <span className="text-accent font-semibold">A:</span> {message.answer}
            </p>
            {onPlayAudio && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handlePlayAudio}
                disabled={isPlaying}
                className="shrink-0"
                title={language === 'en' ? 'Listen to answer' : 'Cevabı dinle'}
              >
                <Volume2 className={`w-4 h-4 ${isPlaying ? 'animate-pulse' : ''}`} />
              </Button>
            )}
          </div>
        </div>

        {/* Metadata */}
        <div className="flex flex-wrap gap-2 text-xs">
          <Badge variant="secondary">
            {message.provider} {message.model}
          </Badge>
          {message.sources && (
            <Badge variant="outline">
              {message.sources.length} {language === 'en' ? 'sources' : 'kaynak'}
            </Badge>
          )}
          <Badge variant="outline">
            {formatTime(message.timestamp)}
          </Badge>
          {message.airline_preference && (
            <Badge variant="outline">
              {message.airline_preference === 'thy' ? '🇹🇷 THY' : 
               message.airline_preference === 'pegasus' ? '✈️ Pegasus' : 
               '🌍 All Airlines'}
            </Badge>
          )}
        </div>

        {/* Session ID */}
        {message.session_id && (
          <div className={`text-xs font-mono p-2 rounded border-l-2 ${
            language === 'tr' ? 'bg-red-50 border-red-300 text-red-800' : 'bg-blue-50 border-blue-300 text-blue-800'
          }`}>
            {language === 'en' ? 'Session ID:' : 'Oturum ID:'} {message.session_id.substring(0, 16)}...
          </div>
        )}

        {/* Statistics */}
        {message.stats && (
          <div className="grid grid-cols-3 gap-4 p-3 bg-muted/50 rounded-lg">
            <div className="text-center">
              <div className="text-lg font-semibold">{message.stats.total_retrieved}</div>
              <div className="text-xs text-muted-foreground">
                {language === 'en' ? 'Sources' : 'Kaynak'}
              </div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold">
                {(message.stats.avg_similarity * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">
                {language === 'en' ? 'Similarity' : 'Benzerlik'}
              </div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold capitalize">
                {language === 'tr' ? 
                  (message.stats.context_quality === 'high' ? 'Yüksek' :
                   message.stats.context_quality === 'medium' ? 'Orta' : 'Düşük') :
                  message.stats.context_quality}
              </div>
              <div className="text-xs text-muted-foreground">
                {language === 'en' ? 'Quality' : 'Kalite'}
              </div>
            </div>
          </div>
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="space-y-2">
            <h4 className="font-medium text-sm">
              {language === 'en' ? 'Sources:' : 'Kaynaklar:'}
            </h4>
            <div className="space-y-1">
              {message.sources.slice(0, 3).map((source, index) => (
                <div key={index} className="text-sm p-2 bg-muted/30 rounded border-l-2 border-muted">
                  <div className="flex justify-between items-start gap-2">
                    <span className="font-medium">{source.source}</span>
                    <Badge variant="outline" className="text-xs">
                      {(source.similarity_score * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {source.airline}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Feedback Section */}
        <div className="border-t pt-4">
          <h4 className="font-medium text-sm mb-3">
            {language === 'en' ? 'Feedback' : 'Geri Bildirim'}
          </h4>
          <div className="flex flex-wrap gap-2">
            {getFeedbackButton(
              'helpful',
              <ThumbsUp className="w-3 h-3 mr-1" />,
              language === 'en' ? 'Helpful' : 'Yardımcı'
            )}
            {getFeedbackButton(
              'not_helpful',
              <ThumbsDown className="w-3 h-3 mr-1" />,
              language === 'en' ? 'Not Helpful' : 'Değil'
            )}
            {getFeedbackButton(
              'too_slow',
              <Clock className="w-3 h-3 mr-1" />,
              language === 'en' ? 'Too Slow' : 'Yavaş'
            )}
            {getFeedbackButton(
              'incorrect',
              <AlertTriangle className="w-3 h-3 mr-1" />,
              language === 'en' ? 'Wrong Info' : 'Yanlış'
            )}
          </div>
          
          {/* Feedback status */}
          {feedbackGiven && (
            <div className="mt-2 text-sm text-muted-foreground">
              {language === 'en' ? 
                (feedbackGiven === 'helpful' ? '✅ You found this helpful' :
                 feedbackGiven === 'not_helpful' ? '⚠️ You marked this as not helpful' :
                 feedbackGiven === 'too_slow' ? '⏱️ You reported this was too slow' :
                 '❌ You reported incorrect information') :
                (feedbackGiven === 'helpful' ? '✅ Bu yanıtın yardımcı olduğunu belirttiniz' :
                 feedbackGiven === 'not_helpful' ? '⚠️ Bu yanıtın yardımcı olmadığını belirttiniz' :
                 feedbackGiven === 'too_slow' ? '⏱️ Bu yanıtın çok yavaş olduğunu belirttiniz' :
                 '❌ Yanlış bilgi olduğunu bildirdiniz')
              }
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};