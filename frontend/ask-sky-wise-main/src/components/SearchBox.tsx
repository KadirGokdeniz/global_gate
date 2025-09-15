import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { VoiceInput } from './VoiceInput';
import { Loader2, Search, Mic, MicOff } from 'lucide-react';
import { Language } from '@/types';

interface SearchBoxProps {
  language: Language;
  t: (key: string) => string;
  onSearch: (question: string) => void;
  isLoading: boolean;
}

export const SearchBox = ({ language, t, onSearch, isLoading }: SearchBoxProps) => {
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
    // Auto-submit after getting transcript
    if (transcript.trim()) {
      onSearch(transcript.trim());
    }
  };

  const toggleVoiceInput = () => {
    setShowVoiceInput(!showVoiceInput);
    setIsListening(!showVoiceInput);
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-4">
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
              className="h-16 pl-6 pr-32 text-lg border-2 border-border/20 hover:border-primary/30 focus:border-primary shadow-sm hover:shadow-md focus:shadow-lg rounded-full bg-background transition-all duration-200"
            />
            
            {/* Voice & Search Buttons Inside Input */}
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={toggleVoiceInput}
                disabled={isLoading}
                className={`h-10 w-10 rounded-full transition-all duration-200 ${
                  isListening ? 'bg-red-50 text-red-600 hover:bg-red-100' : 'hover:bg-muted/50'
                }`}
                title={language === 'en' ? 'Voice input' : 'Sesli giriş'}
              >
                {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </Button>
              
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

      {/* Voice Input Panel - Cleaner Design */}
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