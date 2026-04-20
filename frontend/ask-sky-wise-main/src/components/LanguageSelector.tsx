import { Language } from '@/types';

interface LanguageSelectorProps {
  language: Language;
  onLanguageChange: (language: Language) => void;
}

/**
 * Segmented control — navy chrome header üzerinde çalışacak şekilde tasarlandı.
 * Açık metin (chrome-foreground), ince border (white/10),
 * seçili durum altın accent.
 */
export const LanguageSelector = ({
  language,
  onLanguageChange,
}: LanguageSelectorProps) => {
  return (
    <div
      role="group"
      aria-label="Language"
      className="inline-flex items-center gap-0.5 border border-white/10 rounded-lg p-0.5 bg-white/5"
    >
      <button
        type="button"
        onClick={() => onLanguageChange('en')}
        aria-pressed={language === 'en'}
        aria-label="English"
        className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
          language === 'en'
            ? 'bg-accent text-accent-foreground'
            : 'text-chrome-muted hover:text-chrome-foreground'
        }`}
      >
        EN
      </button>
      <button
        type="button"
        onClick={() => onLanguageChange('tr')}
        aria-pressed={language === 'tr'}
        aria-label="Türkçe"
        className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
          language === 'tr'
            ? 'bg-accent text-accent-foreground'
            : 'text-chrome-muted hover:text-chrome-foreground'
        }`}
      >
        TR
      </button>
    </div>
  );
};