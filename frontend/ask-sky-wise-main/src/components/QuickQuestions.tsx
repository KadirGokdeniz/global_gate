import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { QuickQuestion, Language } from '@/types';

const QUICK_QUESTIONS: Record<Language, Record<string, QuickQuestion[]>> = {
  en: {
    "✈️ Baggage Policies": [
      {
        title: "Excess baggage fees comparison",
        desc: "Turkish Airlines vs Pegasus"
      },
      {
        title: "Carry-on size limits",
        desc: "International flight requirements"
      }
    ],
    "🐕 Pet Travel": [
      {
        title: "Pet travel requirements",
        desc: "Documents and carrier rules"
      },
      {
        title: "Breed restrictions",
        desc: "Which pets are allowed"
      }
    ],
    "🎵 Special Items": [
      {
        title: "Musical instrument transport",
        desc: "Size limits and special handling"
      },
      {
        title: "Sports equipment rules",
        desc: "Golf clubs, skiing gear etc."
      }
    ],
    "⚖️ Passenger Rights": [
      {
        title: "Flight delay compensation",
        desc: "Turkish airline policies"
      },
      {
        title: "Cancellation rights",
        desc: "Refund and rebooking options"
      }
    ]
  },
  tr: {
    "✈️ Bagaj Politikaları": [
      {
        title: "Fazla bagaj ücretleri karşılaştırması",
        desc: "THY vs Pegasus karşılaştırması"
      },
      {
        title: "El bagajı boyut sınırları",
        desc: "Uluslararası uçuş gereksinimleri"
      }
    ],
    "🐕 Evcil Hayvan Seyahati": [
      {
        title: "Evcil hayvan seyahat gereksinimleri",
        desc: "Belgeler ve taşıyıcı kuralları"
      },
      {
        title: "Cins kısıtlamaları",
        desc: "Hangi hayvanlar izinli"
      }
    ],
    "🎵 Özel Eşyalar": [
      {
        title: "Müzik aleti taşıma",
        desc: "Boyut sınırları ve özel işlemler"
      },
      {
        title: "Spor malzemesi kuralları",
        desc: "Golf sopası, kayak ekipmanı vb."
      }
    ],
    "⚖️ Yolcu Hakları": [
      {
        title: "Uçuş gecikme tazminatı",
        desc: "Türk havayolu politikaları"
      },
      {
        title: "İptal hakları",
        desc: "İade ve yeniden rezervasyon seçenekleri"
      }
    ]
  }
};

interface QuickQuestionsProps {
  language: Language;
  onQuestionSelect: (question: string) => void;
}

export const QuickQuestions = ({ language, onQuestionSelect }: QuickQuestionsProps) => {
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  
  const questions = QUICK_QUESTIONS[language];
  const title = language === 'en' ? 'Popular Questions' : 'Popüler Sorular';

  return (
    <Card className="glass-card animate-fade-in-up">
      <CardHeader>
        <CardTitle className="text-xl font-heading">💡 {title}</CardTitle>
        <CardDescription>
          {language === 'en' 
            ? 'Quick access to common airline policy questions'
            : 'Yaygın havayolu politikası sorularına hızlı erişim'
          }
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {Object.entries(questions).map(([category, categoryQuestions]) => (
          <Collapsible
            key={category}
            open={expandedCategory === category}
            onOpenChange={(open) => setExpandedCategory(open ? category : null)}
          >
            <CollapsibleTrigger asChild>
              <Button 
                variant="ghost" 
                className="w-full justify-between p-4 h-auto font-medium"
              >
                <span className="text-left">{category}</span>
                {expandedCategory === category ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-2 pt-2">
              {categoryQuestions.map((question, index) => (
                <Button
                  key={index}
                  variant="outline"
                  className="w-full text-left h-auto p-4 flex flex-col items-start gap-1 hover:bg-primary/5 transition-all duration-300"
                  onClick={() => onQuestionSelect(question.title)}
                >
                  <span className="font-medium text-sm">{question.title}</span>
                  <span className="text-xs text-muted-foreground">{question.desc}</span>
                </Button>
              ))}
            </CollapsibleContent>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  );
};