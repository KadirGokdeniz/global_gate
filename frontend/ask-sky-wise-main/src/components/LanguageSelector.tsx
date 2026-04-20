import { Language } from '@/types';

interface LanguageSelectorProps {
  language: Language;
  onLanguageChange: (language: Language) => void;
}

/**
 * Segmented control tarzı dil seçici.
 * Header içinde inline kullanılmak üzere tasarlandı — kendi position'u yok.
 * Bayrak emoji yerine ülke kodları: daha minimalist, locale-neutral.
 */
export const LanguageSelector = ({
  language,
  onLanguageChange,
}: LanguageSelectorProps) => {
  return (
    <div
      role="group"
      aria-label="Language"
      className="inline-flex items-center gap-0.5 border border-slate-200 dark:border-slate-800 rounded-lg p-0.5 bg-white dark:bg-slate-900"
    >
      <button
        type="button"
        onClick={() => onLanguageChange('en')}
        aria-pressed={language === 'en'}
        aria-label="English"
        className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
          language === 'en'
            ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
            : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
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
            ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
            : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
        }`}
      >
        TR
      </button>
    </div>
  );
};