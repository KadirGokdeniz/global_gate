import { Language } from '@/types';

interface LanguageSelectorProps {
  language: Language;
  onLanguageChange: (language: Language) => void;
}

export const LanguageSelector = ({ language, onLanguageChange }: LanguageSelectorProps) => {
  return (
    <div className="fixed top-4 right-4 z-50">
      <div className="glass-card rounded-2xl p-2 flex gap-1">
        <button
          onClick={() => onLanguageChange('en')}
          className={`language-flag text-2xl ${language === 'en' ? 'active' : ''}`}
          title="English"
        >
          🇺🇸
        </button>
        <button
          onClick={() => onLanguageChange('tr')}
          className={`language-flag text-2xl ${language === 'tr' ? 'active' : ''}`}
          title="Türkçe"
        >
          🇹🇷
        </button>
      </div>
    </div>
  );
};