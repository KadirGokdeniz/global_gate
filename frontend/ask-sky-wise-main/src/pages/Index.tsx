import { useState, useEffect, useCallback } from 'react';
import { useLanguage } from '@/hooks/useLanguage';
import { LanguageSelector } from '@/components/LanguageSelector';
import { SearchBox } from '@/components/SearchBox';
import { ResponseCard } from '@/components/ResponseCard';
import { AirlineSelector } from '@/components/AirlineSelector';
import { QuickQuestions } from '@/components/QuickQuestions';
import { SettingsPanel } from '@/components/SettingsPanel';
import { Sheet, SheetContent, SheetTrigger, SheetOverlay } from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import { Settings, Plane } from 'lucide-react';
import { apiService } from '@/services/api';
import { useToast } from '@/hooks/use-toast';
import { 
  Message, 
  Language, 
  Provider, 
  AirlinePreference, 
  FeedbackType, 
  APIConnection,
  SessionStats 
} from '@/types';

const Index = () => {
  const { language, t, switchLanguage } = useLanguage();
  const { toast } = useToast();
  
  // State management
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [apiConnection, setApiConnection] = useState<APIConnection>({ success: false });
  const [feedbackGiven, setFeedbackGiven] = useState<Record<string, FeedbackType>>({});
  const [currentQuestion, setCurrentQuestion] = useState<string>('');
  const [sessionStats, setSessionStats] = useState<SessionStats>({
    totalQueries: 0,
    satisfactionRate: 0,
    helpfulCount: 0,
    totalFeedback: 0
  });

  // Initialize API connection
  useEffect(() => {
    const initializeAPI = async () => {
      const connection = await apiService.findWorkingAPI();
      setApiConnection(connection);
      
      if (!connection.success) {
        toast({
          title: t('apiFailed'),
          description: connection.error,
          variant: "destructive"
        });
      }
    };
    
    initializeAPI();
  }, [toast, t]);

  // Update session stats when messages or feedback changes
  useEffect(() => {
    const totalQueries = messages.length;
    const totalFeedback = Object.keys(feedbackGiven).length;
    const helpfulCount = Object.values(feedbackGiven).filter(f => f === 'helpful').length;
    const satisfactionRate = totalFeedback > 0 ? (helpfulCount / totalFeedback) * 100 : 0;

    setSessionStats({
      totalQueries,
      satisfactionRate,
      helpfulCount,
      totalFeedback
    });
  }, [messages, feedbackGiven]);

  const handleSendMessage = useCallback(async (
    question: string,
    provider: Provider,
    model: string,
    airline: AirlinePreference
  ) => {
    if (!apiConnection.success) {
      toast({
        title: t('apiFailed'),
        description: t('connectionLost'),
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    
    try {
      const response = await apiService.queryAirlinePolicy(
        question,
        provider,
        model,
        airline,
        language
      );

      if (response.success && response.answer) {
        const newMessage: Message = {
          id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          question,
          answer: response.answer,
          timestamp: new Date(),
          provider,
          model: response.model || model,
          sources: response.sources,
          session_id: response.session_id,
          stats: response.stats,
          performance: response.performance,
          airline_preference: response.airline_preference,
          language
        };

        setMessages(prev => [...prev, newMessage]);
        
        toast({
          title: t('analysisComplete'),
          description: language === 'en' ? 
            'Your question has been answered successfully.' : 
            'Sorunuz başarıyla yanıtlandı.'
        });
      } else {
        toast({
          title: t('apiFailed'),
          description: response.error || 'Unknown error occurred',
          variant: "destructive"
        });
      }
    } catch (error) {
      toast({
        title: t('apiFailed'),
        description: error instanceof Error ? error.message : 'Network error',
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  }, [apiConnection.success, language, t, toast]);

  const handleFeedback = useCallback(async (messageId: string, feedback: FeedbackType) => {
    const message = messages.find(m => m.id === messageId);
    if (!message) return;

    // Update local feedback state immediately
    setFeedbackGiven(prev => ({ ...prev, [messageId]: feedback }));

    // Send feedback to API
    try {
      await apiService.sendFeedback(
        message.question,
        message.answer,
        feedback,
        message.provider,
        message.model
      );

      const feedbackMessages = {
        helpful: language === 'en' ? 'Thanks for your feedback!' : 'Geri bildiriminiz için teşekkürler!',
        not_helpful: language === 'en' ? 'Thanks, we\'ll improve!' : 'Teşekkürler, geliştireceğiz!',
        too_slow: language === 'en' ? 'We\'ll work on speed!' : 'Hızda iyileştirme için çalışacağız!',
        incorrect: language === 'en' ? 'Thanks, we\'ll review this!' : 'Teşekkürler, bu durumu inceleyeceğiz!'
      };

      toast({
        title: language === 'en' ? 'Feedback Recorded' : 'Geri Bildirim Kaydedildi',
        description: feedbackMessages[feedback]
      });
    } catch (error) {
      console.error('Failed to send feedback:', error);
    }
  }, [messages, language, toast]);

  // ✅ DÜZELTILMIŞ: TTS function returns audio URL properly
  const handlePlayAudio = useCallback(async (text: string): Promise<string | null> => {
    console.log('🔊 Index: handlePlayAudio called with text length:', text.length);
    
    try {
      console.log('🔊 Index: Requesting TTS from API...');
      const audioUrl = await apiService.convertTextToSpeech(text, language);
      
      if (audioUrl) {
        console.log('✅ Index: TTS URL received:', audioUrl.substring(0, 50) + '...');
        return audioUrl;
      } else {
        console.error('❌ Index: TTS API returned null');
        toast({
          title: language === 'en' ? 'Audio Error' : 'Ses Hatası',
          description: language === 'en' ? 
            'Failed to generate audio' : 
            'Ses oluşturulamadı',
          variant: "destructive"
        });
        return null;
      }
    } catch (error) {
      console.error('❌ Index: TTS error:', error);
      toast({
        title: language === 'en' ? 'Audio Error' : 'Ses Hatası',
        description: language === 'en' ? 
          'Audio generation failed' : 
          'Ses üretimi başarısız',
        variant: "destructive"
      });
      return null;
    }
  }, [language, toast]);

  const handleReconnect = useCallback(async () => {
    const connection = await apiService.reconnect();
    setApiConnection(connection);
    
    if (connection.success) {
      toast({
        title: t('apiConnected'),
        description: language === 'en' ? 'Successfully reconnected to API' : 'API\'ye başarıyla yeniden bağlandı'
      });
    } else {
      toast({
        title: t('apiFailed'),
        description: connection.error,
        variant: "destructive"
      });
    }
  }, [t, language, toast]);

  const handleClearHistory = useCallback(() => {
    setMessages([]);
    setFeedbackGiven({});
    toast({
      title: language === 'en' ? 'History Cleared' : 'Geçmiş Temizlendi',
      description: language === 'en' ? 
        'All conversation history has been cleared.' : 
        'Tüm konuşma geçmişi temizlendi.'
    });
  }, [language, toast]);

  const handleQuestionSelect = useCallback((question: string) => {
    setCurrentQuestion(question);
  }, []);

  // ✅ DÜZELTILDI: Default airline 'thy' olarak değiştirildi ('all' kaldırıldığı için)
  const [selectedAirline, setSelectedAirline] = useState<AirlinePreference>('thy');
  const [provider, setProvider] = useState<Provider>('OpenAI');
  const [model, setModel] = useState('gpt-4o-mini');
  const [settingsOpen, setSettingsOpen] = useState(false);

  const handleAirlineSelect = (airline: AirlinePreference) => {
    setSelectedAirline(airline);
  };

  const handleSearch = async (question: string) => {
    await handleSendMessage(question, provider, model, selectedAirline);
  };

  const modelOptions = {
    OpenAI: ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4'],
    Claude: ['claude-3-haiku-20240307', 'claude-3-5-haiku-20241022', 'claude-sonnet-4-20250514']
  };

  // Update model when provider changes
  useEffect(() => {
    setModel(modelOptions[provider][0]);
  }, [provider]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-700">
      {/* Modern Header with Glassmorphism */}
      <div className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 dark:bg-slate-900/70 border-b border-white/20 shadow-lg shadow-slate-200/20 dark:shadow-slate-900/20">
        <div className="container mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            {/* Enhanced Logo & Title */}
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl blur-lg opacity-30 animate-pulse"></div>
                <div className="relative bg-gradient-to-r from-blue-600 to-indigo-600 p-3 rounded-xl shadow-xl">
                  <Plane className="w-6 h-6 text-white transform rotate-12" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent leading-tight">
                  {language === 'en' ? 'Airline Assistant' : 'Havayolu Asistanı'}
                </h1>
                <p className="text-sm text-slate-500 dark:text-slate-400 font-medium">
                  {language === 'en' ? 'Powered by AI' : 'AI Destekli'}
                </p>
              </div>
            </div>
            
            {/* Enhanced Controls */}
            <div className="flex items-center gap-4">
              {/* API Status Indicator */}
              <div className="hidden md:flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${apiConnection.success ? 'bg-green-500' : 'bg-red-500'} shadow-lg`}>
                  <div className={`w-full h-full rounded-full ${apiConnection.success ? 'bg-green-400' : 'bg-red-400'} animate-ping opacity-75`}></div>
                </div>
                <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
                  {apiConnection.success ? (language === 'en' ? 'Connected' : 'Bağlı') : (language === 'en' ? 'Offline' : 'Çevrimdışı')}
                </span>
              </div>

              {/* NEW: Current Configuration Status */}
              <div className="hidden lg:flex items-center gap-3 bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm rounded-xl px-4 py-2 border border-white/20 shadow-lg">
                {/* Airline Status */}
                <div className="flex items-center gap-2">
                  <span className="text-lg">
                    {selectedAirline === 'thy' ? '🇹🇷' : '✈️'}
                  </span>
                  <span className="text-xs font-medium text-slate-600 dark:text-slate-300">
                    {selectedAirline === 'thy' ? 'THY' : 'Pegasus'}
                  </span>
                </div>
                
                {/* Divider */}
                <div className="w-px h-4 bg-slate-300 dark:bg-slate-600"></div>
                
                {/* AI Provider Status */}
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${
                    provider === 'OpenAI' ? 'bg-green-500' : 'bg-orange-500'
                  }`}></div>
                  <span className="text-xs font-medium text-slate-600 dark:text-slate-300">
                    {provider}
                  </span>
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    {model.includes('gpt') ? model.replace('gpt-', 'GPT-') : 
                     model.includes('claude') ? model.split('-')[1] + '-' + model.split('-')[2] : 
                     model}
                  </span>
                </div>
              </div>

              {/* Mobile: Compact Status */}
              <div className="lg:hidden flex items-center gap-2 bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-white/20">
                <span className="text-sm">
                  {selectedAirline === 'thy' ? '🇹🇷' : 
                   selectedAirline === 'pegasus' ? '✈️' : '🌍'}
                </span>
                <div className={`w-1.5 h-1.5 rounded-full ${
                  provider === 'OpenAI' ? 'bg-green-500' : 'bg-orange-500'
                }`}></div>
              </div>

              <LanguageSelector language={language} onLanguageChange={switchLanguage} />
              
              <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
                <SheetTrigger asChild>
                  <Button 
                    variant="ghost" 
                    size="lg" 
                    className="w-12 h-12 rounded-xl bg-white/50 dark:bg-slate-800/50 hover:bg-white/80 dark:hover:bg-slate-700/80 backdrop-blur-sm border border-white/20 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
                  >
                    <Settings className="w-6 h-6 text-slate-700 dark:text-slate-200" />
                  </Button>
                </SheetTrigger>
                <SheetOverlay className="bg-black/50 backdrop-blur-sm" />
                <SheetContent side="right" className="w-[750px] p-0 border-l-0 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl">
                  <div className="p-8">
                    <SettingsPanel
                      language={language}
                      t={t}
                      provider={provider}
                      model={model}
                      selectedAirline={selectedAirline}
                      onProviderChange={setProvider}
                      onModelChange={setModel}
                      onAirlineChange={handleAirlineSelect}
                      apiConnection={apiConnection}
                      sessionStats={sessionStats}
                      onReconnect={handleReconnect}
                      onClearHistory={handleClearHistory}
                    />
                  </div>
                </SheetContent>
              </Sheet>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content with Modern Styling */}
      <div className="container mx-auto px-6 relative">
        {messages.length === 0 ? (
          /* Enhanced Landing Page */
          <div className="min-h-[calc(100vh-120px)] flex flex-col items-center justify-center px-4 py-20">
            {/* Animated Background Elements */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-indigo-600/20 rounded-full blur-3xl animate-pulse"></div>
              <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-gradient-to-br from-violet-400/20 to-purple-600/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
            </div>

            {/* Enhanced Title Section */}
            <div className="text-center space-y-8 mb-20 max-w-4xl relative z-10">
              <div className="space-y-4">
                <h1 className="text-5xl md:text-7xl font-black bg-gradient-to-r from-slate-800 via-blue-800 to-indigo-800 dark:from-slate-100 dark:via-blue-100 dark:to-indigo-100 bg-clip-text text-transparent leading-tight tracking-tight">
                  {language === 'en' ? 'Airline Assistant' : 'Havayolu Asistanı'}
                </h1>
                <div className="h-1 w-24 bg-gradient-to-r from-blue-600 to-indigo-600 mx-auto rounded-full"></div>
              </div>
              <p className="text-xl md:text-2xl text-slate-600 dark:text-slate-300 leading-relaxed font-medium">
                {language === 'en' ? 
                  'Smart search in airline policies with AI power' : 
                  'Yapay zeka destekli havayolu politikaları arama'}
              </p>
              <div className="flex flex-wrap justify-center gap-4 text-sm">
                <div className="flex items-center gap-2 bg-white/60 dark:bg-slate-800/60 px-4 py-2 rounded-full backdrop-blur-sm border border-white/20">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-slate-700 dark:text-slate-200 font-medium">
                    {language === 'en' ? 'Real-time responses' : 'Anlık yanıtlar'}
                  </span>
                </div>
                <div className="flex items-center gap-2 bg-white/60 dark:bg-slate-800/60 px-4 py-2 rounded-full backdrop-blur-sm border border-white/20">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  <span className="text-slate-700 dark:text-slate-200 font-medium">
                    {language === 'en' ? 'Multiple airlines' : 'Çoklu havayolu'}
                  </span>
                </div>
                <div className="flex items-center gap-2 bg-white/60 dark:bg-slate-800/60 px-4 py-2 rounded-full backdrop-blur-sm border border-white/20">
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                  <span className="text-slate-700 dark:text-slate-200 font-medium">
                    {language === 'en' ? 'Voice support' : 'Ses desteği'}
                  </span>
                </div>
              </div>
            </div>

            {/* ✅ EKLENDI: Airline Selector Landing Page'de */}
            <div className="w-full max-w-4xl mb-12 relative z-10">
              <AirlineSelector
                selectedAirline={selectedAirline}
                onAirlineSelect={handleAirlineSelect}
                language={language}
              />
            </div>

            {/* Enhanced Search Box */}
            <div className="w-full max-w-3xl mb-16 relative z-10">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-indigo-600/20 rounded-3xl blur-xl"></div>
                <div className="relative bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl rounded-3xl p-2 shadow-2xl border border-white/20">
                  <SearchBox
                    language={language}
                    t={t}
                    onSearch={handleSearch}
                    isLoading={isLoading}
                  />
                </div>
              </div>
            </div>

            {/* Enhanced Quick Questions */}
            <div className="w-full max-w-5xl relative z-10">
              <QuickQuestions
                language={language}
                onQuestionSelect={(question) => handleSearch(question)}
              />
            </div>
          </div>
        ) : (
          /* Enhanced Results Page */
          <div className="py-8 space-y-8">
            {/* Compact Search Bar */}
            <div className="max-w-3xl mx-auto px-4 relative z-10">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-indigo-600/10 rounded-2xl blur-lg"></div>
                <div className="relative bg-white/70 dark:bg-slate-800/70 backdrop-blur-lg rounded-2xl p-1 shadow-xl border border-white/20">
                  <SearchBox
                    language={language}
                    t={t}
                    onSearch={handleSearch}
                    isLoading={isLoading}
                  />
                </div>
              </div>
            </div>

            {/* ✅ DÜZELTILDI: Messages reverse - En yeni mesajlar üstte */}
            <div className="space-y-8 px-4 relative z-10">
              {messages.slice().reverse().map((message, index) => (
                <div key={message.id} className="animate-fade-in-up" style={{animationDelay: `${index * 0.1}s`}}>
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
      </div>
    </div>
  );
};

export default Index;