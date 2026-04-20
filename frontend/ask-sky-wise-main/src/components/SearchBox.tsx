import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { VoiceInput } from './VoiceInput';
import { Loader2, ArrowUp, Mic, MicOff, Brain } from 'lucide-react';
import { Language } from '@/types';

interface SearchBoxProps {
  language: Language;
  t: (key: string) => string;
  onSearch: (question: string) => void;
  isLoading: boolean;
  enableCoT?: boolean;
  onCoTChange?: (enabled: boolean) => void;
}

export const SearchBox = ({
  language,
  onSearch,
  isLoading,
  enableCoT = false,
  onCoTChange,
}: SearchBoxProps) => {
  const [question, setQuestion] = useState('');
  const [showVoicePanel, setShowVoicePanel] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const isEn = language === 'en';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;
    onSearch(question.trim());
    setQuestion('');
  };

  const handleVoiceTranscript = (transcript: string) => {
    setQuestion(transcript);
    setShowVoicePanel(false);
    if (transcript.trim()) {
      onSearch(transcript.trim());
    }
  };

  return (
    <div className="w-full space-y-2">
      <form onSubmit={handleSubmit}>
        {/* Ana input konteyneri — tek, temiz border; gradient glow yok */}
        <div className="relative flex items-center gap-1 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 focus-within:border-slate-400 dark:focus-within:border-slate-600 transition-colors">
          <Input
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              isEn
                ? 'Ask about baggage, pets, delays...'
                : 'Bagaj, evcil hayvan, gecikme hakkında sor...'
            }
            disabled={isLoading}
            aria-label={isEn ? 'Search query' : 'Arama sorgusu'}
            className="flex-1 h-12 px-4 bg-transparent border-0 shadow-none text-base placeholder:text-slate-400 focus-visible:ring-0 focus-visible:ring-offset-0"
          />

          {/* Actions — subtle, sağ tarafa hizalı */}
          <div className="flex items-center gap-0.5 pr-1.5">
            {/* CoT Toggle — sadece ikon + küçük vurgu */}
            {onCoTChange && (
              <button
                type="button"
                onClick={() => onCoTChange(!enableCoT)}
                disabled={isLoading}
                aria-pressed={enableCoT}
                aria-label={
                  isEn
                    ? 'Chain of Thought reasoning'
                    : 'Düşünce Zinciri akıl yürütme'
                }
                title={
                  isEn
                    ? 'Chain of Thought reasoning'
                    : 'Düşünce Zinciri akıl yürütme'
                }
                className={`h-9 w-9 flex items-center justify-center rounded-lg transition-colors ${
                  enableCoT
                    ? 'text-violet-600 dark:text-violet-400 bg-violet-50 dark:bg-violet-950/40'
                    : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                }`}
              >
                <Brain className="w-4 h-4" />
              </button>
            )}

            {/* Voice */}
            <button
              type="button"
              onClick={() => setShowVoicePanel(!showVoicePanel)}
              disabled={isLoading}
              aria-pressed={showVoicePanel}
              aria-label={isEn ? 'Voice input' : 'Sesli giriş'}
              title={isEn ? 'Voice input' : 'Sesli giriş'}
              className={`h-9 w-9 flex items-center justify-center rounded-lg transition-colors ${
                showVoicePanel
                  ? 'text-red-600 dark:text-red-500 bg-red-50 dark:bg-red-950/40'
                  : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
              }`}
            >
              {showVoicePanel ? (
                <MicOff className="w-4 h-4" />
              ) : (
                <Mic className="w-4 h-4" />
              )}
            </button>

            {/* Submit — tek belirgin action */}
            <Button
              type="submit"
              size="sm"
              disabled={!question.trim() || isLoading}
              aria-label={isEn ? 'Send' : 'Gönder'}
              className="h-9 w-9 p-0 rounded-lg ml-0.5"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <ArrowUp className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </form>

      {/* CoT aktif hint — sakin bir ipucu satırı */}
      {enableCoT && (
        <div className="flex items-center gap-1.5 px-1 text-xs text-violet-600 dark:text-violet-400">
          <Brain className="w-3 h-3" aria-hidden="true" />
          <span>
            {isEn
              ? 'Chain of Thought is on — the AI will show its reasoning'
              : 'Düşünce Zinciri aktif — AI akıl yürütme adımlarını gösterecek'}
          </span>
        </div>
      )}

      {/* Voice Panel */}
      {showVoicePanel && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
          <VoiceInput
            onTranscript={handleVoiceTranscript}
            language={language}
          />
        </div>
      )}
    </div>
  );
};