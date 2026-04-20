import { Language } from '@/types';

interface SkeletonCardProps {
  language: Language;
  question: string; // Kullanıcının sorduğu soru — gerçek soruyu göster, sahte değil
}

/**
 * Yanıt yüklenirken gösterilen iskelet kart.
 *
 * Tasarım kararları:
 * - Soru gerçek gösterilir (kullanıcı ne sorduğunu görmek ister)
 * - Yanıt alanı shimmer animasyonlu çizgilerle temsil edilir
 * - ResponseCard'ın tam boyutunu taklit eder → layout shift olmaz
 * - "Yapay Zeka düşünüyor..." gibi metin YOK — sadelik premium
 * - Satır uzunlukları farklı → gerçek metin izlenimi
 */
export const SkeletonCard = ({ language, question }: SkeletonCardProps) => {
  const isEn = language === 'en';

  return (
    <article className="w-full max-w-3xl mx-auto animate-fade-in">
      {/* Soru — gerçek metin */}
      <div className="mb-3 px-4">
        <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
          {isEn ? 'Question' : 'Soru'} · {new Intl.DateTimeFormat(isEn ? 'en-US' : 'tr-TR', {
            hour: '2-digit',
            minute: '2-digit',
          }).format(new Date())}
        </div>
        <div className="text-sm text-slate-700 dark:text-slate-300">
          {question}
        </div>
      </div>

      {/* Cevap kartı — shimmer */}
      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        <div className="p-5 sm:p-6">
          {/* Answer label + action buttons placeholder */}
          <div className="flex items-center justify-between mb-4">
            <div className="text-xs font-medium text-slate-500 dark:text-slate-400">
              {isEn ? 'Answer' : 'Cevap'}
            </div>
            {/* Buton placeholder'ları — gerçek butonlarla aynı boyut */}
            <div className="flex items-center gap-1">
              <div className="h-7 w-7 rounded-md bg-slate-100 dark:bg-slate-800 animate-pulse" />
              <div className="h-7 w-7 rounded-md bg-slate-100 dark:bg-slate-800 animate-pulse" />
            </div>
          </div>

          {/* Metin satırları — farklı uzunluklar, gerçek paragraf hissi */}
          <div className="space-y-2.5">
            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-full" />
            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-[92%]" />
            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-[97%]" />
            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-[85%]" />

            {/* Kısa satır — paragraf sonu hissi */}
            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-[60%]" />

            {/* Boşluk */}
            <div className="h-2" />

            {/* İkinci paragraf */}
            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-full" />
            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-[88%]" />
            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-[45%]" />
          </div>
        </div>

        {/* Metadata strip placeholder */}
        <div className="px-5 sm:px-6 py-2.5 bg-slate-50 dark:bg-slate-900/50 border-t border-slate-100 dark:border-slate-800">
          <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-48" />
        </div>

        {/* Feedback strip placeholder */}
        <div className="px-5 sm:px-6 py-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
          <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded animate-pulse w-28" />
          <div className="flex gap-1">
            {[72, 80, 56, 64].map((w, i) => (
              <div
                key={i}
                className="h-7 rounded-md bg-slate-100 dark:bg-slate-800 animate-pulse"
                style={{ width: `${w}px` }}
              />
            ))}
          </div>
        </div>
      </div>
    </article>
  );
};