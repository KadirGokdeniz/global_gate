import { AirlinePreference, Language } from '@/types';
import { Check } from 'lucide-react';

interface AirlineSelectorProps {
  selectedAirline: AirlinePreference;
  onAirlineSelect: (airline: AirlinePreference) => void;
  language: Language;
}

interface AirlineOption {
  id: AirlinePreference;
  code: string;      // Havayolu kodu — 'TK', 'PC'
  name: string;
  description: string;
  accent: string;    // Sol tarafta ince renkli şerit — kimlik ipucu
}

export const AirlineSelector = ({
  selectedAirline,
  onAirlineSelect,
  language,
}: AirlineSelectorProps) => {
  const isEn = language === 'en';

  const airlines: AirlineOption[] = [
    {
      id: 'thy',
      code: 'TK',
      name: isEn ? 'Turkish Airlines' : 'Türk Hava Yolları',
      description: isEn
        ? '·'
        : '·',
      accent: 'bg-red-500',
    },
    {
      id: 'pegasus',
      code: 'PC',
      name: isEn ? 'Pegasus Airlines' : 'Pegasus Hava Yolları',
      description: isEn
        ? ''
        : '',
      accent: 'bg-amber-500',
    },
  ];

  return (
    <div
      role="radiogroup"
      aria-label={isEn ? 'Select airline' : 'Havayolu seçin'}
      className="grid grid-cols-1 sm:grid-cols-2 gap-3"
    >
      {airlines.map((airline) => {
        const selected = selectedAirline === airline.id;

        return (
          <button
            key={airline.id}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onAirlineSelect(airline.id)}
            className={`group relative flex items-center gap-3 p-4 rounded-lg border text-left transition-colors ${
              selected
                ? 'border-slate-900 dark:border-slate-100 bg-white dark:bg-slate-900'
                : 'border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-slate-300 dark:hover:border-slate-700'
            }`}
          >
            {/* Sol şerit — airline kimliği (ince, baskın değil) */}
            <div
              className={`w-1 self-stretch rounded-full ${airline.accent}`}
              aria-hidden="true"
            />

            {/* Kod badge'i — monospace, enterprise havası */}
            <div
              className={`flex items-center justify-center w-12 h-12 rounded-md text-sm font-mono font-semibold shrink-0 ${
                selected
                  ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                  : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
              }`}
              aria-hidden="true"
            >
              {airline.code}
            </div>

            {/* Label */}
            <div className="flex-1 min-w-0">
              <div className="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">
                {airline.name}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 truncate">
                {airline.description}
              </div>
            </div>

            {/* Seçim göstergesi */}
            {selected && (
              <Check
                className="w-4 h-4 text-slate-900 dark:text-slate-100 shrink-0"
                aria-hidden="true"
              />
            )}
          </button>
        );
      })}
    </div>
  );
};