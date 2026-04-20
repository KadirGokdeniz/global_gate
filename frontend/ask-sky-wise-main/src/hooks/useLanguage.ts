import { useCallback } from 'react';
import { Language } from '@/types';
import { useLocalStorage } from './useLocalStorage';

// NOT: "allAirlines", "turkishAirlinesOnly", "pegasusOnly" çevirileri
// kaldırıldı — AirlinePreference artık sadece 'thy' | 'pegasus' ve "All
// Airlines" seçeneği UI'dan çıkarıldığı için bu anahtarlar kullanılmıyordu.
const TRANSLATIONS = {
  en: {
    title: 'Airline Policy Assistant',
    subtitle: 'Get instant answers about airline policies powered by AI',
    askQuestion: 'Ask Your Question',
    questionPlaceholder: 'What would you like to know about airline policies?',
    chooseProvider: 'Choose Provider:',
    chooseModel: 'Model:',
    chooseAirline: 'Choose Airline',
    askButton: 'Ask',
    recentConversations: 'Recent Conversations',
    popularQuestions: 'Popular Questions',
    bagagePolicies: 'Baggage Policies',
    petTravel: 'Pet Travel',
    specialItems: 'Special Items',
    passengerRights: 'Passenger Rights',
    feedback: 'Feedback',
    helpful: '👍 Helpful',
    notHelpful: '👎 Not Helpful',
    tooSlow: '⏱️ Too Slow',
    wrongInfo: '❌ Wrong Info',
    sessionStats: 'Session Stats',
    totalQueries: 'Total Queries',
    satisfaction: 'Satisfaction',
    clearHistory: '🗑️ Clear All History',
    apiConnected: '⚡ API Connected',
    apiFailed: '❌ API Connection Failed',
    reconnect: '🔄 Reconnect',
    analyzing: 'is analyzing airline policies...',
    analysisComplete: '✅ Analysis complete!',
    connectionLost: 'Connection lost',
    requestTimeout: 'Request timeout',
    welcomeMessage: 'Welcome to AI Assistant!',
    welcomeDescription:
      'Ask your first question about airline policies above.',
    features: 'Features:',
    smartSearch: '✅ Smart Policy Search',
    fastResponse: '⚡ Fast Response Times',
    qualityTracking: '📊 Quality Tracking',
    satisfactionTracking: '🎯 User Satisfaction',
    voiceInput: 'Voice Input',
    clickToRecord: 'Click to Record',
    processingAudio: 'Processing audio...',
    voiceQuestion: 'Ask with Voice',
    listenToAnswer: 'Listen to Answer',
    speechDetected: 'Speech detected:',
    speechNotDetected: 'No speech detected',
  },
  tr: {
    title: 'Havayolu Politika Asistanı',
    subtitle: 'Yapay zeka destekli havayolu politikaları danışmanınız',
    askQuestion: 'Sorunuzu Sorun',
    questionPlaceholder:
      'Havayolu politikaları hakkında merak ettiklerinizi yazın...',
    chooseProvider: 'AI Sağlayıcısı Seçin:',
    chooseModel: 'Model:',
    chooseAirline: 'Havayolu Seçin',
    askButton: 'AI Asistanına Sor',
    recentConversations: 'Son Konuşmalar',
    popularQuestions: '💡 Popüler Sorular',
    bagagePolicies: '✈️ Bagaj Politikaları',
    petTravel: '🐕 Evcil Hayvan Seyahati',
    specialItems: '🎵 Özel Eşyalar',
    passengerRights: '⚖️ Yolcu Hakları',
    feedback: '📝 Geri Bildirim',
    helpful: '👍 Yardımcı Oldu',
    notHelpful: '👎 Yardımcı Olmadı',
    tooSlow: '⏱️ Çok Yavaş',
    wrongInfo: '❌ Yanlış Bilgi',
    sessionStats: '📊 Oturum İstatistikleri',
    totalQueries: 'Toplam Sorgu',
    satisfaction: 'Memnuniyet',
    clearHistory: '🗑️ Geçmişi Temizle',
    apiConnected: '⚡ API Bağlandı',
    apiFailed: '❌ API Bağlantısı Başarısız',
    reconnect: '🔄 Yeniden Bağlan',
    analyzing: 'havayolu politikalarını analiz ediyor...',
    analysisComplete: '✅ Analiz tamamlandı!',
    connectionLost: 'Bağlantı kesildi',
    requestTimeout: 'İstek zaman aşımı',
    welcomeMessage: 'Havayolu Politikaları AI Asistanına Hoş Geldiniz!',
    welcomeDescription:
      'Yukarıdan havayolu politikaları hakkında ilk sorunuzu sorun.',
    features: 'Özellikler:',
    smartSearch: '✅ Akıllı Politika Arama',
    fastResponse: '⚡ Hızlı Yanıt Süreleri',
    qualityTracking: '📊 Kalite Takibi',
    satisfactionTracking: '🎯 Memnuniyet Takibi',
    voiceInput: 'Sesli Giriş',
    clickToRecord: 'Kayıt İçin Tıklayın',
    processingAudio: 'Ses işleniyor...',
    voiceQuestion: 'Sesle Sor',
    listenToAnswer: 'Cevabı Dinle',
    speechDetected: 'Konuşma algılandı:',
    speechNotDetected: 'Konuşma algılanmadı',
  },
} as const;

/**
 * Kullanıcının ilk ziyaretinde browser dilinden default belirle.
 * Türkçe tarayıcı → 'tr', diğer → 'en'. Kullanıcı manuel değiştirince
 * bundan sonra localStorage'daki değer kullanılır.
 */
function detectDefaultLanguage(): Language {
  if (typeof navigator === 'undefined') return 'en';
  return navigator.language.toLowerCase().startsWith('tr') ? 'tr' : 'en';
}

// Validator: localStorage'dan gelen değer geçerli bir Language mi?
const isValidLanguage = (v: unknown): v is Language =>
  v === 'en' || v === 'tr';

export const useLanguage = () => {
  const [language, setLanguage] = useLocalStorage<Language>(
    'airline-assistant:language',
    detectDefaultLanguage(),
    isValidLanguage,
  );

  const t = useCallback(
    (key: string): string => {
      const translations = TRANSLATIONS[language] as Record<string, string>;
      return translations[key] || key;
    },
    [language],
  );

  const switchLanguage = useCallback(
    (newLanguage: Language) => {
      setLanguage(newLanguage);
    },
    [setLanguage],
  );

  return { language, t, switchLanguage };
};