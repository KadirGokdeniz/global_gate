import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import {
  RefreshCw,
  Trash2,
  Wifi,
  WifiOff,
  AlertCircle,
  Plane,
  Cpu,
  BarChart3,
  X,
} from 'lucide-react';
import { ReactNode } from 'react';
import {
  Language,
  Provider,
  AirlinePreference,
  APIConnection,
  SessionStats,
} from '@/types';

// ═══════════════════════════════════════════════════════════════════
// Reusable card primitives — tüm settings tek tip görünsün diye
// ═══════════════════════════════════════════════════════════════════

interface SectionProps {
  icon: ReactNode;
  title: string;
  description?: string;
  children: ReactNode;
}

const Section = ({ icon, title, description, children }: SectionProps) => (
  <section className="py-5 border-b border-slate-200 dark:border-slate-800 last:border-0">
    <header className="mb-3">
      <div className="flex items-center gap-2 text-slate-900 dark:text-slate-100">
        <span className="text-slate-500 dark:text-slate-400" aria-hidden="true">
          {icon}
        </span>
        <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
      </div>
      {description && (
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 ml-6">
          {description}
        </p>
      )}
    </header>
    <div className="ml-6">{children}</div>
  </section>
);

interface FieldProps {
  label: string;
  children: ReactNode;
}

const Field = ({ label, children }: FieldProps) => (
  <div className="space-y-1.5">
    <label className="text-xs font-medium text-slate-600 dark:text-slate-400">
      {label}
    </label>
    {children}
  </div>
);

// ═══════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════

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
  onClose?: () => void;   // Sheet'i kapatmak için — mobilde kritik
}

const MODEL_OPTIONS: Record<Provider, string[]> = {
  OpenAI: ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4'],
  Claude: [
    'claude-3-haiku-20240307',
    'claude-3-5-haiku-20241022',
    'claude-sonnet-4-20250514',
  ],
};

export const SettingsPanel = ({
  language,
  provider,
  model,
  selectedAirline,
  onProviderChange,
  onModelChange,
  onAirlineChange,
  apiConnection,
  sessionStats,
  onReconnect,
  onClearHistory,
  onClose,
}: SettingsPanelProps) => {
  const isEn = language === 'en';

  // ─── Connection Status UI helper ─────────────────────────────────
  const connectionBadge = () => {
    if (apiConnection.success) {
      return (
        <div className="flex items-center gap-2 text-sm">
          <Wifi className="w-4 h-4 text-emerald-600 dark:text-emerald-500" />
          <span className="text-slate-900 dark:text-slate-100 font-medium">
            {isEn ? 'Connected' : 'Bağlı'}
          </span>
          {!apiConnection.models_ready && (
            <span className="text-xs text-slate-500 dark:text-slate-400">
              · {isEn ? 'Loading models' : 'Modeller yükleniyor'}
            </span>
          )}
        </div>
      );
    }
    return (
      <div className="flex items-start gap-2 text-sm">
        <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-500 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-slate-900 dark:text-slate-100 font-medium">
            {isEn ? 'Disconnected' : 'Bağlantı yok'}
          </div>
          {apiConnection.error && (
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 break-words">
              {apiConnection.error}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full">
      {/* ─── Header — sticky, close button ile ─────────────────────
          Mobilde kritik: kullanıcının "çıkış yolu" net görünmeli.
          Close button 44x44px (Apple/Google HIG minimum tap target). */}
      <div className="px-5 sm:px-6 py-3 sm:py-4 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 sticky top-0 z-10 flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            {isEn ? 'Settings' : 'Ayarlar'}
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            {isEn
              ? 'Configure your assistant preferences'
              : 'Asistan tercihlerinizi yapılandırın'}
          </p>
        </div>

        {/* Close button — büyük tap target (44x44px mobilde) */}
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label={isEn ? 'Close settings' : 'Ayarları kapat'}
            className="shrink-0 h-11 w-11 flex items-center justify-center rounded-lg text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors -mr-2"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* ─── Body ────────────────────────────────────────────────── */}
      <div className="flex-1 px-6 overflow-y-auto">
        {/* Airline Preference */}
        <Section
          icon={<Plane className="w-4 h-4" />}
          title={isEn ? 'Airline' : 'Havayolu'}
          description={
            isEn
              ? 'Each airline has its own conversation history'
              : 'Her havayolunun kendi konuşma geçmişi vardır'
          }
        >
          <Select value={selectedAirline} onValueChange={onAirlineChange}>
            <SelectTrigger className="h-9 text-sm bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="thy">
                {isEn ? 'Turkish Airlines' : 'Türk Hava Yolları'}
              </SelectItem>
              <SelectItem value="pegasus">
                {isEn ? 'Pegasus Airlines' : 'Pegasus Hava Yolları'}
              </SelectItem>
            </SelectContent>
          </Select>
        </Section>

        {/* AI Model */}
        <Section
          icon={<Cpu className="w-4 h-4" />}
          title={isEn ? 'AI Model' : 'AI Modeli'}
          description={
            isEn
              ? 'Provider and model used to generate responses'
              : 'Yanıtlar için kullanılacak sağlayıcı ve model'
          }
        >
          <div className="grid grid-cols-2 gap-3">
            <Field label={isEn ? 'Provider' : 'Sağlayıcı'}>
              <Select value={provider} onValueChange={onProviderChange}>
                <SelectTrigger className="h-9 text-sm bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="OpenAI">OpenAI</SelectItem>
                  <SelectItem value="Claude">Claude</SelectItem>
                </SelectContent>
              </Select>
            </Field>

            <Field label={isEn ? 'Model' : 'Model'}>
              <Select value={model} onValueChange={onModelChange}>
                <SelectTrigger className="h-9 text-sm bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODEL_OPTIONS[provider].map((m) => (
                    <SelectItem key={m} value={m} className="text-xs font-mono">
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>
        </Section>

        {/* Connection */}
        <Section
          icon={
            apiConnection.success ? (
              <Wifi className="w-4 h-4" />
            ) : (
              <WifiOff className="w-4 h-4" />
            )
          }
          title={isEn ? 'Connection' : 'Bağlantı'}
        >
          <div className="space-y-3">
            {connectionBadge()}
            <Button
              variant="outline"
              size="sm"
              onClick={onReconnect}
              className="h-8 text-xs border-slate-200 dark:border-slate-800"
            >
              <RefreshCw className="w-3 h-3 mr-1.5" />
              {isEn ? 'Reconnect' : 'Yeniden bağlan'}
            </Button>
          </div>
        </Section>

        {/* Session Stats */}
        <Section
          icon={<BarChart3 className="w-4 h-4" />}
          title={isEn ? 'Session' : 'Oturum'}
          description={
            isEn
              ? 'Your activity in this session'
              : 'Bu oturumdaki etkinliğiniz'
          }
        >
          <dl className="space-y-2.5">
            <div className="flex items-baseline justify-between">
              <dt className="text-sm text-slate-600 dark:text-slate-400">
                {isEn ? 'Total queries' : 'Toplam sorgu'}
              </dt>
              <dd className="text-sm font-semibold tabular-nums text-slate-900 dark:text-slate-100">
                {sessionStats.totalQueries}
              </dd>
            </div>
            <div className="flex items-baseline justify-between">
              <dt className="text-sm text-slate-600 dark:text-slate-400">
                {isEn ? 'Satisfaction' : 'Memnuniyet'}
              </dt>
              <dd className="text-sm font-semibold tabular-nums text-slate-900 dark:text-slate-100">
                {sessionStats.totalFeedback > 0
                  ? `${sessionStats.satisfactionRate.toFixed(0)}%`
                  : '—'}
              </dd>
            </div>
            {sessionStats.totalFeedback > 0 && (
              <div className="flex items-baseline justify-between">
                <dt className="text-sm text-slate-600 dark:text-slate-400">
                  {isEn ? 'Helpful responses' : 'Yardımcı yanıtlar'}
                </dt>
                <dd className="text-sm font-semibold tabular-nums text-slate-900 dark:text-slate-100">
                  {sessionStats.helpfulCount} / {sessionStats.totalFeedback}
                </dd>
              </div>
            )}
          </dl>
        </Section>

        {/* Danger Zone */}
        <Section
          icon={<Trash2 className="w-4 h-4" />}
          title={isEn ? 'Data' : 'Veri'}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={onClearHistory}
            className="h-8 text-xs text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-500 dark:hover:text-red-400 dark:hover:bg-red-950/30 border-slate-200 dark:border-slate-800"
          >
            <Trash2 className="w-3 h-3 mr-1.5" />
            {isEn
              ? `Clear ${selectedAirline === 'thy' ? 'THY' : 'Pegasus'} history`
              : `${selectedAirline === 'thy' ? 'THY' : 'Pegasus'} geçmişini temizle`}
          </Button>
        </Section>
      </div>
    </div>
  );
};