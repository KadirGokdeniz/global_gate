import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { ChevronDown, Lightbulb, Luggage, PawPrint, Music2, Scale } from 'lucide-react';
import { useState, ReactNode } from 'react';
import { QuickQuestion, Language } from '@/types';

// ═══════════════════════════════════════════════════════════════════
// Kategori tanımları — emoji yerine lucide ikon + stable key
// ═══════════════════════════════════════════════════════════════════

interface Category {
  key: string;               // Dilden bağımsız stable key (state için)
  icon: ReactNode;
  label: { en: string; tr: string };
  questions: {
    title: { en: string; tr: string };
    desc: { en: string; tr: string };
  }[];
}

const CATEGORIES: Category[] = [
  {
    key: 'baggage',
    icon: <Luggage className="w-4 h-4" />,
    label: { en: 'Baggage Policies', tr: 'Bagaj Politikaları' },
    questions: [
      {
        title: {
          en: 'Excess baggage fees',
          tr: 'Fazla bagaj ücretleri',
        },
        desc: {
          en: 'Turkish Airlines and Pegasus data',
          tr: 'THY vs Pegasus verileri',
        },
      },
      {
        title: {
          en: 'Carry-on size limits',
          tr: 'El bagajı boyut sınırları',
        },
        desc: {
          en: 'International flight requirements',
          tr: 'Uluslararası uçuş gereksinimleri',
        },
      },
    ],
  },
  {
    key: 'pets',
    icon: <PawPrint className="w-4 h-4" />,
    label: { en: 'Pet Travel', tr: 'Evcil Hayvan Seyahati' },
    questions: [
      {
        title: {
          en: 'Pet travel requirements',
          tr: 'Evcil hayvan seyahat gereksinimleri',
        },
        desc: {
          en: 'Documents and carrier rules',
          tr: 'Belgeler ve taşıyıcı kuralları',
        },
      },
      {
        title: {
          en: 'Breed restrictions',
          tr: 'Cins kısıtlamaları',
        },
        desc: {
          en: 'Which pets are allowed',
          tr: 'Hangi hayvanlar izinli',
        },
      },
    ],
  },
  {
    key: 'special',
    icon: <Music2 className="w-4 h-4" />,
    label: { en: 'Special Items', tr: 'Özel Eşyalar' },
    questions: [
      {
        title: {
          en: 'Musical instrument transport',
          tr: 'Müzik aleti taşıma',
        },
        desc: {
          en: 'Size limits and special handling',
          tr: 'Boyut sınırları ve özel işlemler',
        },
      },
      {
        title: {
          en: 'Sports equipment rules',
          tr: 'Spor malzemesi kuralları',
        },
        desc: {
          en: 'Golf clubs, skiing gear etc.',
          tr: 'Golf sopası, kayak ekipmanı vb.',
        },
      },
    ],
  },
  {
    key: 'rights',
    icon: <Scale className="w-4 h-4" />,
    label: { en: 'Passenger Rights', tr: 'Yolcu Hakları' },
    questions: [
      {
        title: {
          en: 'Flight delay compensation',
          tr: 'Uçuş gecikme tazminatı',
        },
        desc: {
          en: 'Turkish airline policies',
          tr: 'Türk havayolu politikaları',
        },
      },
      {
        title: {
          en: 'Cancellation rights',
          tr: 'İptal hakları',
        },
        desc: {
          en: 'Refund and rebooking options',
          tr: 'İade ve yeniden rezervasyon seçenekleri',
        },
      },
    ],
  },
];

// ═══════════════════════════════════════════════════════════════════
// Component
// ═══════════════════════════════════════════════════════════════════

interface QuickQuestionsProps {
  language: Language;
  onQuestionSelect: (question: string) => void;
}

export const QuickQuestions = ({
  language,
  onQuestionSelect,
}: QuickQuestionsProps) => {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const isEn = language === 'en';

  return (
    <div className="space-y-3">
      {/* Başlık — büyük card header yerine minimal */}
      <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
        <Lightbulb className="w-4 h-4" aria-hidden="true" />
        <span className="font-medium">
          {isEn ? 'Popular questions' : 'Popüler sorular'}
        </span>
      </div>

      {/* Kategoriler — 2-kolon grid masaüstünde */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {CATEGORIES.map((category) => {
          const isOpen = expandedKey === category.key;

          return (
            <Collapsible
              key={category.key}
              open={isOpen}
              onOpenChange={(open) =>
                setExpandedKey(open ? category.key : null)
              }
              className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden"
            >
              <CollapsibleTrigger asChild>
                <button
                  type="button"
                  className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span
                      className="text-slate-500 dark:text-slate-400"
                      aria-hidden="true"
                    >
                      {category.icon}
                    </span>
                    <span className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                      {category.label[isEn ? 'en' : 'tr']}
                    </span>
                  </div>
                  <ChevronDown
                    className={`w-4 h-4 text-slate-400 shrink-0 transition-transform ${
                      isOpen ? 'rotate-180' : ''
                    }`}
                    aria-hidden="true"
                  />
                </button>
              </CollapsibleTrigger>

              <CollapsibleContent className="border-t border-slate-100 dark:border-slate-800">
                <div className="p-2 space-y-1">
                  {category.questions.map((q, i) => (
                    <Button
                      key={i}
                      variant="ghost"
                      onClick={() => onQuestionSelect(q.title[isEn ? 'en' : 'tr'])}
                      className="w-full h-auto px-3 py-2.5 justify-start flex-col items-start gap-0.5 text-left hover:bg-slate-50 dark:hover:bg-slate-800/50"
                    >
                      <span className="text-sm font-normal text-slate-900 dark:text-slate-100 w-full whitespace-normal">
                        {q.title[isEn ? 'en' : 'tr']}
                      </span>
                      <span className="text-xs font-normal text-slate-500 dark:text-slate-400 w-full whitespace-normal">
                        {q.desc[isEn ? 'en' : 'tr']}
                      </span>
                    </Button>
                  ))}
                </div>
              </CollapsibleContent>
            </Collapsible>
          );
        })}
      </div>
    </div>
  );
};