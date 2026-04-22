import { useLanguage } from '@/hooks/useLanguage';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

/**
 * Privacy Notice — minimal, dürüst.
 * Tam legal document değil ama "bu proje verilerinizi nasıl işliyor"
 * sorusuna net cevap veriyor. Bu bir kişisel proje için doğru ton.
 */
export const PrivacyPage = () => {
  const { language } = useLanguage();
  const isEn = language === 'en';

  return (
    <div className="min-h-screen bg-background">
      {/* Minimal header — main app'teki navy header yerine daha sade */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 sm:px-6 py-4">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            {isEn ? 'Back to Assistant' : 'Asistana dön'}
          </Link>
        </div>
      </header>

      {/* Content */}
      <main className="container mx-auto px-4 sm:px-6 py-12 max-w-2xl">
        <article className="space-y-8">
          {/* Title */}
          <div className="space-y-3">
            <h1 className="text-3xl font-semibold tracking-tight text-foreground">
              {isEn ? 'Privacy Notice' : 'Gizlilik Bildirimi'}
            </h1>
            <p className="text-sm text-muted-foreground">
              {isEn
                ? 'Last updated: April 2026'
                : 'Son güncelleme: Nisan 2026'}
            </p>
          </div>

          {/* About section */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-foreground">
              {isEn ? 'About this project' : 'Bu proje hakkında'}
            </h2>
            <p className="text-[15px] leading-relaxed text-foreground">
              {isEn
                ? 'Airline Assistant is a personal project that allows you to query airline policies using AI. It is not a full commercial product — you should always verify critical information through the airline\'s official channels.'
                : 'Airline Assistant, havayolu politikalarını yapay zeka ile sorgulamanıza olanak tanıyan kişisel bir projedir. Tam bir ticari ürün değildir — kritik bilgileri her zaman havayolunun resmi kaynakları üzerinden doğrulamanızı öneririz.'}
            </p>
          </section>

          {/* Data handling */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-foreground">
              {isEn ? 'How your data is handled' : 'Verileriniz nasıl işlenir'}
            </h2>
            <ul className="space-y-2.5 text-[15px] leading-relaxed text-foreground">
              <li className="flex gap-2">
                <span
                  className="mt-[0.4em] shrink-0 text-accent"
                  aria-hidden="true"
                >
                  ·
                </span>
                <span>
                  {isEn
                    ? 'Your questions and answers are stored in your browser (localStorage) and never leave your device unless you send them.'
                    : 'Sorularınız ve cevaplar tarayıcınızda (localStorage) saklanır ve göndermediğiniz sürece cihazınızdan ayrılmaz.'}
                </span>
              </li>
              <li className="flex gap-2">
                <span
                  className="mt-[0.4em] shrink-0 text-accent"
                  aria-hidden="true"
                >
                  ·
                </span>
                <span>
                  {isEn
                    ? 'When you submit a question, it is sent to our server to retrieve a relevant answer. We do not associate queries with personal identifiers.'
                    : 'Bir soru gönderdiğinizde, ilgili cevabı almak için sunucumuza iletilir. Sorguları kişisel tanımlayıcılarla ilişkilendirmeyiz.'}
                </span>
              </li>
              <li className="flex gap-2">
                <span
                  className="mt-[0.4em] shrink-0 text-accent"
                  aria-hidden="true"
                >
                  ·
                </span>
                <span>
                  {isEn
                    ? 'Voice recordings (if used) are sent only for transcription and are not stored on our servers.'
                    : 'Ses kayıtları (kullanıldığında) yalnızca transkripsiyon için gönderilir ve sunucularımızda saklanmaz.'}
                </span>
              </li>
              <li className="flex gap-2">
                <span
                  className="mt-[0.4em] shrink-0 text-accent"
                  aria-hidden="true"
                >
                  ·
                </span>
                <span>
                  {isEn
                    ? 'Errors and crashes are automatically reported (via Sentry) to help us fix bugs. No personal content is included in these reports.'
                    : 'Hatalar ve çökmeler, sorunları düzeltmemize yardımcı olmak için otomatik olarak bildirilir (Sentry üzerinden). Bu raporlarda kişisel içerik bulunmaz.'}
                </span>
              </li>
            </ul>
          </section>

          {/* Third parties */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-foreground">
              {isEn ? 'Third-party services' : 'Üçüncü taraf hizmetler'}
            </h2>
            <p className="text-[15px] leading-relaxed text-foreground">
              {isEn
                ? 'To provide answers and voice features, this project uses:'
                : 'Cevaplar ve ses özellikleri için bu proje şu servisleri kullanır:'}
            </p>
            <ul className="space-y-1.5 text-[15px] leading-relaxed text-foreground">
              <li className="flex gap-2">
                <span className="mt-[0.4em] shrink-0 text-muted-foreground" aria-hidden="true">·</span>
                <span>
                  <strong className="font-medium">OpenAI</strong> &amp;{' '}
                  <strong className="font-medium">Anthropic</strong>
                  {isEn ? ' — AI-generated answers' : ' — AI cevapları'}
                </span>
              </li>
              <li className="flex gap-2">
                <span className="mt-[0.4em] shrink-0 text-muted-foreground" aria-hidden="true">·</span>
                <span>
                  <strong className="font-medium">AssemblyAI</strong>
                  {isEn ? ' — speech-to-text' : ' — konuşmadan yazıya'}
                </span>
              </li>
              <li className="flex gap-2">
                <span className="mt-[0.4em] shrink-0 text-muted-foreground" aria-hidden="true">·</span>
                <span>
                  <strong className="font-medium">ElevenLabs</strong>
                  {isEn ? ' — text-to-speech' : ' — yazıdan konuşmaya'}
                </span>
              </li>
              <li className="flex gap-2">
                <span className="mt-[0.4em] shrink-0 text-muted-foreground" aria-hidden="true">·</span>
                <span>
                  <strong className="font-medium">Sentry</strong>
                  {isEn ? ' — error tracking' : ' — hata takibi'}
                </span>
              </li>
            </ul>
            <p className="text-[15px] leading-relaxed text-foreground">
              {isEn
                ? 'Each of these providers processes data according to their own privacy policies.'
                : 'Bu sağlayıcıların her biri, veriyi kendi gizlilik politikalarına göre işler.'}
            </p>
          </section>

          {/* Your rights */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-foreground">
              {isEn ? 'Your rights' : 'Haklarınız'}
            </h2>
            <p className="text-[15px] leading-relaxed text-foreground">
              {isEn
                ? 'You can clear all locally stored data anytime by clicking "Clear History" in Settings. To request removal of any server-side logs, contact us below.'
                : 'Ayarlar\'dan "Geçmişi Temizle" seçeneğine tıklayarak yerel olarak depolanan tüm verileri dilediğiniz zaman silebilirsiniz. Sunucu tarafındaki kayıtların silinmesini talep etmek için aşağıdaki iletişim adresini kullanabilirsiniz.'}
            </p>
          </section>

          {/* Contact */}
          <section className="space-y-3 pt-4 border-t border-border">
            <h2 className="text-lg font-semibold text-foreground">
              {isEn ? 'Contact' : 'İletişim'}
            </h2>
            <p className="text-[15px] leading-relaxed text-foreground">
              {isEn
                ? 'For any questions, bug reports, or concerns:'
                : 'Sorularınız, hata bildirimleri veya endişeleriniz için:'}
            </p>
            <a
              href="mailto:kadirqokdeniz@hotmail.com?subject=Airline Assistant - Privacy"
              className="inline-block text-[15px] text-accent hover:underline underline-offset-4"
            >
              kadirqokdeniz@hotmail.com
            </a>
          </section>

          {/* Footnote */}
          <section className="pt-6 border-t border-border">
            <p className="text-xs text-muted-foreground leading-relaxed">
              {isEn
                ? 'This project is under active development. Answers are informational only — for official airline policies, please consult the respective airline\'s own sources.'
                : 'Bu proje aktif geliştirme aşamasındadır. Cevaplar yalnızca bilgilendirme amaçlıdır — resmi havayolu politikaları için ilgili havayolunun kendi kaynaklarına başvurun.'}
            </p>
          </section>
        </article>
      </main>
    </div>
  );
};