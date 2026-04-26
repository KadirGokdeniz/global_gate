import { useState, useEffect, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useLanguage } from '@/hooks/useLanguage';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { usePersistedMessages } from '@/hooks/usePersistedMessages';
import { LanguageSelector } from '@/components/LanguageSelector';
import { SearchBox } from '@/components/SearchBox';
import { ResponseCard } from '@/components/ResponseCard';
import { SkeletonCard } from '@/components/SkeletonCard';
import { AirlineSelector } from '@/components/AirlineSelector';
import { QuickQuestions } from '@/components/QuickQuestions';
import { SettingsPanel } from '@/components/SettingsPanel';
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetOverlay,
} from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import { Settings, Plane } from 'lucide-react';
import { apiService } from '@/services/api';
import type { APIErrorDetails } from '@/services/api';
import { useToast } from '@/hooks/use-toast';
import {
  Message,
  Provider,
  AirlinePreference,
  FeedbackType,
  APIConnection,
  SessionStats,
} from '@/types';

const MODEL_OPTIONS: Record<Provider, string[]> = {
  OpenAI: ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4'],
  Claude: [
    'claude-3-haiku-20240307',
    'claude-3-5-haiku-20241022',
    'claude-sonnet-4-20250514',
  ],
};

const isValidAirline = (v: unknown): v is AirlinePreference =>
  v === 'thy' || v === 'pegasus';
const isValidProvider = (v: unknown): v is Provider =>
  v === 'OpenAI' || v === 'Claude';
const isBoolean = (v: unknown): v is boolean => typeof v === 'boolean';
const isValidFeedback = (v: unknown): boolean =>
  typeof v === 'object' && v !== null;

const Index = () => {
  const { language, t, switchLanguage } = useLanguage();
  const { toast } = useToast();

  const { messages: airlineMessages, addMessage, clearAirline } =
    usePersistedMessages();

  const [selectedAirline, setSelectedAirline] =
    useLocalStorage<AirlinePreference>(
      'airline-assistant:selected-airline',
      'thy',
      isValidAirline,
    );

  const [provider, setProvider] = useLocalStorage<Provider>(
    'airline-assistant:provider',
    'OpenAI',
    isValidProvider,
  );

  const [model, setModel] = useLocalStorage<string>(
    'airline-assistant:model',
    'gpt-4o-mini',
  );

  const [enableCoT, setEnableCoT] = useLocalStorage<boolean>(
    'airline-assistant:cot-enabled',
    false,
    isBoolean,
  );

  const [feedbackGiven, setFeedbackGiven] = useLocalStorage<
    Record<string, FeedbackType>
  >('airline-assistant:feedback', {}, isValidFeedback);

  const [isLoading, setIsLoading] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [apiConnection, setApiConnection] = useState<APIConnection>({
    success: false,
  });
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    if (!MODEL_OPTIONS[provider].includes(model)) {
      setModel(MODEL_OPTIONS[provider][0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const messages = airlineMessages[selectedAirline];
  const reversedMessages = useMemo(() => [...messages].reverse(), [messages]);

  const sessionStats: SessionStats = useMemo(() => {
    const totalQueries =
      airlineMessages.thy.length + airlineMessages.pegasus.length;
    const feedbackValues = Object.values(feedbackGiven);
    const totalFeedback = feedbackValues.length;
    const helpfulCount = feedbackValues.filter((f) => f === 'helpful').length;
    const satisfactionRate =
      totalFeedback > 0 ? (helpfulCount / totalFeedback) * 100 : 0;
    return { totalQueries, satisfactionRate, helpfulCount, totalFeedback };
  }, [airlineMessages, feedbackGiven]);

  useEffect(() => {
    const initializeAPI = async () => {
      const connection = await apiService.findWorkingAPI();
      setApiConnection(connection);
      if (!connection.success) {
        toast({
          title: t('apiFailed'),
          description: connection.error,
          variant: 'destructive',
        });
      }
    };
    initializeAPI();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSendMessage = useCallback(
    async (
      question: string,
      currentProvider: Provider,
      currentModel: string,
      airline: AirlinePreference,
    ) => {
      if (!apiConnection.success) {
        toast({
          title: t('apiFailed'),
          description: t('connectionLost'),
          variant: 'destructive',
        });
        return;
      }

      setIsLoading(true);

      try {
        const response = await apiService.queryAirlinePolicy(
          question,
          currentProvider,
          currentModel,
          airline,
          language,
          enableCoT,
        );

        if (response.success && response.answer) {
          const newMessage: Message = {
            id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            question,
            answer: response.answer,
            timestamp: new Date(),
            provider: currentProvider,
            model: response.model || currentModel,
            sources: response.sources,
            session_id: response.session_id,
            stats: response.stats,
            performance: response.performance,
            airline_preference: response.airline_preference,
            language,
            cot_enabled: enableCoT,
            reasoning: response.reasoning,
          };
          addMessage(airline, newMessage);
        } else {
          // Graceful degradation: error.error artık hata türüne göre
          // kategorize edilmiş user-friendly bir mesaj (api.ts'de düzenlendi)
          toast({
            title: t('apiFailed'),
            description: response.error || 'Unknown error occurred',
            variant: 'destructive',
          });
        }
      } catch (error) {
        toast({
          title: t('apiFailed'),
          description: error instanceof Error ? error.message : 'Network error',
          variant: 'destructive',
        });
      } finally {
        setIsLoading(false);
      }
    },
    [apiConnection.success, language, t, toast, enableCoT, addMessage],
  );

  const handleFeedback = useCallback(
    async (messageId: string, feedback: FeedbackType) => {
      const allMessages = [
        ...airlineMessages.thy,
        ...airlineMessages.pegasus,
      ];
      const message = allMessages.find((m) => m.id === messageId);
      if (!message) return;

      setFeedbackGiven((prev) => ({ ...prev, [messageId]: feedback }));

      try {
        await apiService.sendFeedback(
          message.question,
          message.answer,
          feedback,
          message.provider,
          message.model,
        );

        const feedbackMessages = {
          helpful:
            language === 'en'
              ? 'Thanks for your feedback!'
              : 'Geri bildiriminiz için teşekkürler!',
          not_helpful:
            language === 'en'
              ? "Thanks, we'll improve!"
              : 'Teşekkürler, geliştireceğiz!',
          too_slow:
            language === 'en'
              ? "We'll work on speed!"
              : 'Hızda iyileştirme için çalışacağız!',
          incorrect:
            language === 'en'
              ? "Thanks, we'll review this!"
              : 'Teşekkürler, bu durumu inceleyeceğiz!',
        };

        toast({
          title:
            language === 'en'
              ? 'Feedback Recorded'
              : 'Geri Bildirim Kaydedildi',
          description: feedbackMessages[feedback],
        });
      } catch (error) {
        console.error('Failed to send feedback:', error);
      }
    },
    [airlineMessages, language, toast, setFeedbackGiven],
  );

  /**
   * TTS handler — yeni tip: { audioUrl, error? } döner.
   * ResponseCard bu objeyi kendi içinde işler, toast atmıyoruz artık —
   * error mesajı ResponseCard'ın içinde (getAudioErrorMessage) gösteriliyor.
   * Bu, aynı hata için 2 farklı mesaj çıkmasını engeller.
   */
  const handlePlayAudio = useCallback(
    async (
      text: string,
    ): Promise<{ audioUrl: string | null; error?: APIErrorDetails }> => {
      try {
        return await apiService.convertTextToSpeech(text, language);
      } catch (error) {
        // apiService zaten hataları yakalar, buraya gelmezse emniyet ağı
        console.error('Unexpected TTS error:', error);
        return {
          audioUrl: null,
          error: {
            kind: 'unknown',
            message: error instanceof Error ? error.message : 'Unknown',
            userMessage:
              language === 'en'
                ? 'Audio generation failed'
                : 'Ses üretimi başarısız',
          },
        };
      }
    },
    [language],
  );

  const handleReconnect = useCallback(async () => {
    const connection = await apiService.reconnect();
    setApiConnection(connection);

    if (connection.success) {
      toast({
        title: t('apiConnected'),
        description:
          language === 'en'
            ? 'Successfully reconnected to API'
            : "API'ye başarıyla yeniden bağlandı",
      });
    } else {
      toast({
        title: t('apiFailed'),
        description: connection.error,
        variant: 'destructive',
      });
    }
  }, [t, language, toast]);

  const handleClearHistory = useCallback(() => {
    clearAirline(selectedAirline);
    const currentMessages = airlineMessages[selectedAirline];
    const messageIds = currentMessages.map((m) => m.id);
    setFeedbackGiven((prev) => {
      const newFeedback = { ...prev };
      messageIds.forEach((id) => delete newFeedback[id]);
      return newFeedback;
    });

    toast({
      title: language === 'en' ? 'History Cleared' : 'Geçmiş Temizlendi',
      description:
        language === 'en'
          ? `${selectedAirline === 'thy' ? 'Turkish Airlines' : 'Pegasus'} conversation history cleared.`
          : `${selectedAirline === 'thy' ? 'THY' : 'Pegasus'} konuşma geçmişi temizlendi.`,
    });
  }, [
    language,
    toast,
    selectedAirline,
    airlineMessages,
    clearAirline,
    setFeedbackGiven,
  ]);

  const handleAirlineSelect = useCallback(
    (airline: AirlinePreference) => setSelectedAirline(airline),
    [setSelectedAirline],
  );

  const handleProviderChange = useCallback(
    (newProvider: Provider) => {
      setProvider(newProvider);
      setModel(MODEL_OPTIONS[newProvider][0]);
    },
    [setProvider, setModel],
  );

  const handleSearch = useCallback(
    async (question: string) => {
      setCurrentQuestion(question);
      await handleSendMessage(question, provider, model, selectedAirline);
      setCurrentQuestion('');
    },
    [handleSendMessage, provider, model, selectedAirline],
  );

  return (
    <div className="min-h-screen">
      {/* ─── Minimalist Header ─── */}
      <header className="sticky top-0 z-40 bg-chrome border-b border-chrome-border">
        <div className="container mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-8 h-8 rounded-md bg-white/10 border border-white/10">
                <Plane className="w-4 h-4 text-accent" aria-hidden="true" />
              </div>
              <div className="flex items-baseline gap-2">
                <h1 className="text-sm font-semibold text-chrome-foreground tracking-tight">
                  {language === 'en' ? 'Airline Assistant' : 'Havayolu Asistanı'}
                </h1>
                <span className="hidden sm:inline text-xs text-chrome-muted">
                  {selectedAirline === 'thy' ? 'THY' : 'Pegasus'}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-3">
              <div
                className="hidden sm:flex items-center gap-2 text-xs"
                aria-live="polite"
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    apiConnection.success ? 'bg-emerald-400' : 'bg-slate-500'
                  }`}
                  aria-hidden="true"
                />
                <span className="text-chrome-muted">
                  {apiConnection.success
                    ? language === 'en'
                      ? 'Connected'
                      : 'Bağlı'
                    : language === 'en'
                      ? 'Offline'
                      : 'Çevrimdışı'}
                </span>
              </div>

              <div
                className="hidden sm:block w-px h-5 bg-chrome-border"
                aria-hidden="true"
              />

              <LanguageSelector
                language={language}
                onLanguageChange={switchLanguage}
              />

              <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
                <SheetTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    aria-label={
                      language === 'en' ? 'Open settings' : 'Ayarları aç'
                    }
                    className="h-8 w-8 p-0 text-chrome-muted hover:text-chrome-foreground hover:bg-white/10"
                  >
                    <Settings className="w-4 h-4" />
                  </Button>
                </SheetTrigger>
                <SheetOverlay className="bg-slate-900/40 backdrop-blur-sm" />
                <SheetContent className="w-[88%] sm:max-w-md md:max-w-lg p-0 border-l border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 overflow-y-auto [&>button]:hidden">
                  <SettingsPanel
                    language={language}
                    t={t}
                    provider={provider}
                    model={model}
                    selectedAirline={selectedAirline}
                    onProviderChange={handleProviderChange}
                    onModelChange={setModel}
                    onAirlineChange={handleAirlineSelect}
                    apiConnection={apiConnection}
                    sessionStats={sessionStats}
                    onReconnect={handleReconnect}
                    onClearHistory={handleClearHistory}
                    onClose={() => setSettingsOpen(false)}
                  />
                </SheetContent>
              </Sheet>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 sm:px-6">
        {messages.length === 0 ? (
          /* ─── Landing Page ── */
          <div className="min-h-[calc(100vh-56px)] flex flex-col items-center justify-center py-12 sm:py-20">
            <div className="text-center space-y-5 mb-12 max-w-2xl">
              <h2 className="text-4xl sm:text-5xl md:text-6xl font-semibold tracking-tight text-foreground">
                {language === 'en' ? 'Airline Assistant' : 'Havayolu Asistanı'}
              </h2>

              <div className="flex justify-center" aria-hidden="true">
                <div className="h-px w-16 bg-accent" />
              </div>

              <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
                {language === 'en'
                  ? 'Smart search in airline policies with AI'
                  : 'Yapay zeka destekli havayolu politikaları arama'}
              </p>
            </div>

            <div className="w-full max-w-3xl mb-8">
              <AirlineSelector
                selectedAirline={selectedAirline}
                onAirlineSelect={handleAirlineSelect}
                language={language}
              />
            </div>

            <div className="w-full max-w-2xl mb-12">
              <SearchBox
                language={language}
                t={t}
                onSearch={handleSearch}
                isLoading={isLoading}
                enableCoT={enableCoT}
                onCoTChange={setEnableCoT}
              />
            </div>

            {!isLoading && (
              <div className="w-full max-w-4xl">
                <QuickQuestions
                  language={language}
                  onQuestionSelect={handleSearch}
                />
              </div>
            )}

            {isLoading && currentQuestion && (
              <div className="w-full max-w-3xl">
                <SkeletonCard
                  language={language}
                  question={currentQuestion}
                />
              </div>
            )}
          </div>
        ) : (
          /* ─── Results Page ── */
          <div className="py-6 sm:py-8 space-y-6">
            <div className="max-w-3xl mx-auto">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
                  <span className="font-medium text-slate-900 dark:text-slate-100">
                    {selectedAirline === 'thy'
                      ? language === 'en'
                        ? 'Turkish Airlines'
                        : 'Türk Hava Yolları'
                      : language === 'en'
                        ? 'Pegasus Airlines'
                        : 'Pegasus Hava Yolları'}
                  </span>
                  <span className="text-slate-400 dark:text-slate-600">·</span>
                  <span>
                    {messages.length}{' '}
                    {language === 'en' ? 'messages' : 'mesaj'}
                  </span>
                </div>
              </div>
            </div>

            <div className="max-w-3xl mx-auto">
              <SearchBox
                language={language}
                t={t}
                onSearch={handleSearch}
                isLoading={isLoading}
                enableCoT={enableCoT}
                onCoTChange={setEnableCoT}
              />
            </div>

            <div className="space-y-6">
              {isLoading && currentQuestion && (
                <SkeletonCard
                  language={language}
                  question={currentQuestion}
                />
              )}

              {reversedMessages.map((message, index) => (
                <div
                  key={message.id}
                  className="animate-fade-in-up"
                  style={{ animationDelay: `${index * 0.05}s` }}
                >
                  <ResponseCard
                    message={message}
                    language={language}
                    onFeedback={handleFeedback}
                    onPlayAudio={handlePlayAudio}
                    feedbackGiven={feedbackGiven[message.id]}
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* ─── Footer — privacy + feedback links ─── */}
      <footer className="border-t border-border mt-16">
        <div className="container mx-auto px-4 sm:px-6 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-muted-foreground">
            <div className="flex items-center gap-4">
              <Link
                to="/privacy"
                className="hover:text-foreground transition-colors"
              >
                {language === 'en' ? 'Privacy' : 'Gizlilik'}
              </Link>
              <a
                href="mailto:kadirqokdeniz@hotmail.com?subject=Airline Assistant - Feedback"
                className="hover:text-foreground transition-colors"
              >
                {language === 'en' ? 'Feedback' : 'Geri Bildirim'}
              </a>
            </div>
            <div>
              {language === 'en'
                ? 'Personal project · Not a commercial service'
                : 'Kişisel proje · Ticari bir hizmet değil'}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;