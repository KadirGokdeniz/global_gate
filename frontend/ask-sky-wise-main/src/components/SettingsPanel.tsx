// ✅ SettingsPanel.tsx - Genişlik düzeltmeleri
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, Trash2, Zap, AlertCircle, Settings, Plane } from 'lucide-react';
import { Language, Provider, AirlinePreference, APIConnection, SessionStats } from '@/types';

interface SettingsPanelProps {
  language: Language;
  t: (key: string) => string;
  provider: Provider;
  model: string;
  selectedAirline: AirlinePreference;
  onProviderChange: (provider: Provider) => void;
  onModelChange: (model: string) => void;
  onAirlineChange: (airline: AirlinePreference) => void;
  apiConnection: APIConnection;
  sessionStats: SessionStats;
  onReconnect: () => void;
  onClearHistory: () => void;
}

export const SettingsPanel = ({
  language,
  t,
  provider,
  model,
  selectedAirline,
  onProviderChange,
  onModelChange,
  onAirlineChange,
  apiConnection,
  sessionStats,
  onReconnect,
  onClearHistory
}: SettingsPanelProps) => {

  const getConnectionStatus = () => {
    if (apiConnection.success) {
      return apiConnection.models_ready ? (
        <div className="status-success rounded-lg p-4 flex items-center gap-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
          <Zap className="w-5 h-5 text-green-600 dark:text-green-400" />
          <span className="text-sm font-medium text-green-800 dark:text-green-200">{t('apiConnected')}</span>
        </div>
      ) : (
        <div className="status-warning rounded-lg p-4 flex items-center gap-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800">
          <RefreshCw className="w-5 h-5 animate-spin text-yellow-600 dark:text-yellow-400" />
          <span className="text-sm font-medium text-yellow-800 dark:text-yellow-200">{t('apiConnected')} (Loading...)</span>
        </div>
      );
    }
    
    return (
      <div className="status-error rounded-lg p-4 flex items-center gap-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
        <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
        <span className="text-sm font-medium text-red-800 dark:text-red-200">
          {t('apiFailed')}: {apiConnection.error}
        </span>
      </div>
    );
  };

  const modelOptions = {
    OpenAI: ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4'],
    Claude: ['claude-3-haiku-20240307', 'claude-3-5-haiku-20241022', 'claude-sonnet-4-20250514']
  };

  const airlineOptions = [
    { 
      value: 'all' as const, 
      label: language === 'en' ? 'All Airlines' : 'Tüm Havayolları',
      icon: '🌐'
    },
    { 
      value: 'thy' as const, 
      label: language === 'en' ? 'Turkish Airlines' : 'Türk Hava Yolları',
      icon: '🇹🇷'
    },
    { 
      value: 'pegasus' as const, 
      label: language === 'en' ? 'Pegasus Airlines' : 'Pegasus Hava Yolları',
      icon: '✈️'
    }
  ];

  return (
    // ✅ FIX 1: Container genişliği artırıldı ve padding optimize edildi
    <div className="space-y-6 w-full min-w-0 max-w-screen px-2">
      {/* Enhanced Header */}
      <div className="text-center pb-6 border-b border-border/20">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          {language === 'en' ? 'Settings Panel' : 'Ayarlar Paneli'}
        </h2>
        <p className="text-sm text-muted-foreground">
          {language === 'en' ? 'Configure your AI assistant preferences' : 'AI asistan tercihlerinizi yapılandırın'}
        </p>
      </div>

      {/* ✅ FIX 2: Current Configuration - Grid genişletildi */}
      <Card className="border-border/50 bg-gradient-to-r from-blue-50/50 to-indigo-50/50 dark:from-blue-900/10 dark:to-indigo-900/10">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg flex items-center gap-2 text-blue-700 dark:text-blue-300">
            <Settings className="w-5 h-5" />
            {language === 'en' ? 'Current Configuration' : 'Mevcut Yapılandırma'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* ✅ FIX 3: Grid'i tek column yaparak daha fazla yer veriyoruz */}
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-5 border border-white/20">
              <div className="text-sm font-medium text-muted-foreground mb-3">
                {language === 'en' ? 'AI Provider & Model' : 'AI Sağlayıcı & Model'}
              </div>
              <div className="flex items-center gap-3 flex-wrap">
                <Badge variant="secondary" className="text-sm px-3 py-1">
                  {provider}
                </Badge>
                <span className="text-sm text-muted-foreground">→</span>
                {/* ✅ FIX 4: Model ismi için daha fazla yer */}
                <span className="text-sm font-medium break-all">{model}</span>
              </div>
            </div>
            
            <div className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-5 border border-white/20">
              <div className="text-sm font-medium text-muted-foreground mb-3">
                {language === 'en' ? 'Selected Airline' : 'Seçili Havayolu'}
              </div>
              <div className="flex items-center gap-3">
                <span className="text-lg">
                  {selectedAirline === 'all' ? '🌐' : 
                   selectedAirline === 'thy' ? '🇹🇷' : '✈️'}
                </span>
                <span className="text-sm font-medium">
                  {selectedAirline === 'all' ? (language === 'en' ? 'All Airlines' : 'Tüm Havayolları') :
                   selectedAirline === 'thy' ? (language === 'en' ? 'Turkish Airlines' : 'Türk Hava Yolları') :
                   (language === 'en' ? 'Pegasus Airlines' : 'Pegasus Hava Yolları')}
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ✅ FIX 5: Airline Preference - Select genişletildi */}
      <Card className="border-border/50">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg flex items-center gap-2">
            <Plane className="w-5 h-5" />
            {language === 'en' ? 'Airline Preference' : 'Havayolu Tercihi'}
          </CardTitle>
          <p className="text-sm text-muted-foreground mt-2">
            {language === 'en' ? 
              'Choose which airline policies to focus on during searches' : 
              'Aramalarda hangi havayolu politikalarına odaklanılacağını seçin'}
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* ✅ FIX 6: Select height artırıldı ve padding optimize edildi */}
            <Select value={selectedAirline} onValueChange={onAirlineChange}>
              <SelectTrigger className="h-14 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="min-w-[300px]">
                {airlineOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value} className="py-4">
                    <div className="flex items-center gap-3 w-full">
                      <span className="text-lg">{option.icon}</span>
                      <span className="font-medium text-left flex-1">{option.label}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {/* Selection Info */}
            <div className="bg-muted/30 rounded-lg p-4">
              <div className="text-xs font-medium text-muted-foreground mb-2">
                {language === 'en' ? 'SELECTED AIRLINE' : 'SEÇİLİ HAVAYOLU'}
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xl">
                  {airlineOptions.find(a => a.value === selectedAirline)?.icon}
                </span>
                <span className="font-medium text-lg">
                  {airlineOptions.find(a => a.value === selectedAirline)?.label}
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ✅ FIX 7: AI Model Settings - Genişletildi */}
      <Card className="border-border/50">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg flex items-center gap-2">
            <Settings className="w-5 h-5" />
            {language === 'en' ? 'AI Model Settings' : 'AI Model Ayarları'}
          </CardTitle>
          <p className="text-sm text-muted-foreground mt-2">
            {language === 'en' ? 
              'Configure which AI provider and model to use for responses' : 
              'Yanıtlar için hangi AI sağlayıcı ve modelinin kullanılacağını yapılandırın'}
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Provider Selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium block">{t('chooseProvider')}</label>
            <Select value={provider} onValueChange={onProviderChange}>
              <SelectTrigger className="h-14 w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="min-w-[250px]">
                <SelectItem value="OpenAI" className="py-4">
                  <div className="flex items-center gap-3 w-full">
                    <div className="w-8 h-8 rounded bg-green-500 flex items-center justify-center">
                      <span className="text-white text-sm font-bold">AI</span>
                    </div>
                    <span className="font-medium text-left">OpenAI</span>
                  </div>
                </SelectItem>
                <SelectItem value="Claude" className="py-4">
                  <div className="flex items-center gap-3 w-full">
                    <div className="w-8 h-8 rounded bg-orange-500 flex items-center justify-center">
                      <span className="text-white text-sm font-bold">C</span>
                    </div>
                    <span className="font-medium text-left">Claude</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Model Selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium block">{t('chooseModel')}</label>
            <Select value={model} onValueChange={onModelChange}>
              <SelectTrigger className="h-14 w-full">
                <SelectValue />
              </SelectTrigger>
              {/* ✅ FIX 8: SelectContent genişletildi uzun model isimleri için */}
              <SelectContent className="min-w-[350px]">
                {modelOptions[provider].map((modelOption) => (
                  <SelectItem key={modelOption} value={modelOption} className="py-4">
                    <div className="flex flex-col items-start w-full">
                      {/* ✅ FIX 9: Model isimlerini daha readable yaptık */}
                      <span className="font-medium text-left break-all">{modelOption}</span>
                      <span className="text-xs text-muted-foreground text-left">
                        {provider} Model
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* ✅ FIX 10: Model Info genişletildi */}
          <div className="bg-muted/30 rounded-lg p-4">
            <div className="text-xs font-medium text-muted-foreground mb-2">
              {language === 'en' ? 'CURRENT SELECTION' : 'MEVCUT SEÇİM'}
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <Badge variant="outline" className="text-sm px-3 py-1">
                {provider}
              </Badge>
              <span className="text-sm">→</span>
              <span className="text-sm font-medium break-all flex-1">{model}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Connection Status */}
      <Card className="border-border/50">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg flex items-center gap-2">
            🔗 {language === 'en' ? 'Connection Status' : 'Bağlantı Durumu'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {getConnectionStatus()}
          <Button 
            variant="outline" 
            onClick={onReconnect}
            className="w-full h-12 text-base"
            size="sm"
          >
            <RefreshCw className="w-5 h-5 mr-3" />
            {t('reconnect')}
          </Button>
        </CardContent>
      </Card>

      {/* ✅ FIX 11: Session Statistics - Grid optimize edildi */}
      <Card className="border-border/50">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg">📊 {t('sessionStats')}</CardTitle>
          <p className="text-sm text-muted-foreground mt-2">
            {language === 'en' ? 
              'Your current session performance metrics' : 
              'Mevcut oturum performans metrikleri'}
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* ✅ FIX 12: Grid'i tek column yaparak mobil uyumlu hale getirdik */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="text-center p-6 bg-blue-50/50 dark:bg-blue-900/20 rounded-lg border border-blue-200/50 dark:border-blue-800/50">
              <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                {sessionStats.totalQueries}
              </div>
              <div className="text-sm text-blue-700 dark:text-blue-300 font-medium">
                {t('totalQueries')}
              </div>
            </div>
            <div className="text-center p-6 bg-green-50/50 dark:bg-green-900/20 rounded-lg border border-green-200/50 dark:border-green-800/50">
              <div className="text-4xl font-bold text-green-600 dark:text-green-400 mb-2">
                {sessionStats.totalFeedback > 0 
                  ? `${sessionStats.satisfactionRate.toFixed(0)}%`
                  : (language === 'en' ? 'N/A' : 'Yok')
                }
              </div>
              <div className="text-sm text-green-700 dark:text-green-300 font-medium">
                {t('satisfaction')}
              </div>
            </div>
          </div>
          
          {sessionStats.totalFeedback > 0 && (
            <div className="text-center bg-muted/30 rounded-lg p-4">
              <Badge variant="outline" className="text-sm px-4 py-2">
                {sessionStats.helpfulCount} / {sessionStats.totalFeedback} {language === 'en' ? 'helpful responses' : 'yardımcı yanıt'}
              </Badge>
            </div>
          )}

          <Button 
            variant="outline" 
            onClick={onClearHistory}
            className="w-full text-destructive hover:text-destructive h-12 text-base"
            size="sm"
          >
            <Trash2 className="w-5 h-5 mr-3" />
            {t('clearHistory')}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};