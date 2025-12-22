import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { VoiceInput } from './VoiceInput';
import { Loader2, Search, Mic, MicOff, Brain } from 'lucide-react';
import { Language } from '@/types';

interface SearchBoxProps {
  language: Language;
  t: (key: string) => string;
  onSearch: (question: string) => void;
  isLoading: boolean;
  enableCoT?: boolean;  // ✅ Yeni prop
  onCoTChange?: (enabled: boolean) => void;  // ✅ Yeni prop
}

export const SearchBox = ({ 
  language, 
  t, 
  onSearch, 
  isLoading,
  enableCoT = false,
  onCoTChange
}: SearchBoxProps) => {
  const [question, setQuestion] = useState('');
  const [showVoiceInput, setShowVoiceInput] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;
    
    onSearch(question.trim());
    setQuestion('');
  };

  const handleVoiceTranscript = (transcript: string) => {
    setQuestion(transcript);
    setShowVoiceInput(false);
    setIsListening(false);
    if (transcript.trim()) {
      onSearch(transcript.trim());
    }
  };

  const toggleVoiceInput = () => {
    setShowVoiceInput(!showVoiceInput);
    setIsListening(!showVoiceInput);
  };

  const toggleCoT = () => {
    if (onCoTChange) {
      onCoTChange(!enableCoT);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-3">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center group">
          {/* Search Input */}
          <div className="relative flex-1">
            <Input
              ref={inputRef}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={language === 'en' ? 
                'Ask about airline policies...' : 
                'Havayolu politikaları hakkında sorun...'}
              disabled={isLoading}
              className="h-16 pl-6 pr-44 text-lg border-2 border-border/20 hover:border-primary/30 focus:border-primary shadow-sm hover:shadow-md focus:shadow-lg rounded-full bg-background transition-all duration-200"
            />
            
            {/* CoT, Voice & Search Buttons Inside Input */}
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center gap-1.5">
              {/* ✅ CoT Toggle Button */}
              {onCoTChange && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={toggleCoT}
                  disabled={isLoading}
                  className={`h-10 px-3 rounded-full transition-all duration-200 flex items-center gap-1.5 ${
                    enableCoT 
                      ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 hover:bg-purple-200 dark:hover:bg-purple-900/50' 
                      : 'hover:bg-muted/50 text-muted-foreground'
                  }`}
                  title={language === 'en' ? 'Chain of Thought reasoning' : 'Düşünce Zinciri akıl yürütme'}
                >
                  <Brain className={`w-4 h-4 ${enableCoT ? 'text-purple-600 dark:text-purple-400' : ''}`} />
                  <span className={`text-xs font-medium hidden sm:inline ${enableCoT ? 'text-purple-600 dark:text-purple-400' : ''}`}>
                    CoT
                  </span>
                </Button>
              )}

              {/* Voice Button */}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={toggleVoiceInput}
                disabled={isLoading}
                className={`h-10 w-10 rounded-full transition-all duration-200 ${
                  isListening ? 'bg-red-50 dark:bg-red-900/30 text-red-600 hover:bg-red-100' : 'hover:bg-muted/50'
                }`}
                title={language === 'en' ? 'Voice input' : 'Sesli giriş'}
              >
                {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </Button>
              
              {/* Search Button */}
              <Button 
                type="submit"
                size="sm"
                disabled={!question.trim() || isLoading}
                className="h-10 w-10 rounded-full bg-primary hover:bg-primary/90 disabled:opacity-50 shadow-sm"
                title={language === 'en' ? 'Search' : 'Ara'}
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Search className="w-5 h-5" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </form>

      {/* ✅ CoT Info Banner - Sadece aktifken göster */}
      {enableCoT && (
        <div className="flex items-center justify-center gap-2 text-xs text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-900/20 rounded-full px-4 py-1.5 mx-auto w-fit animate-fade-in">
          <Brain className="w-3 h-3" />
          <span>
            {language === 'en' 
              ? 'Chain of Thought enabled - AI will show reasoning steps' 
              : 'Düşünce Zinciri aktif - AI akıl yürütme adımlarını gösterecek'}
          </span>
        </div>
      )}

      {/* Voice Input Panel */}
      {showVoiceInput && (
        <div className="bg-background border border-border/50 rounded-xl p-6 shadow-sm animate-fade-in-up">
          <VoiceInput 
            onTranscript={handleVoiceTranscript}
            language={language}
          />
        </div>
      )}
    </div>
  );
};