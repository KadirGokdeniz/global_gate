import { Component, ErrorInfo, ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  /**
   * Opsiyonel: özel fallback UI. Verilmezse default bir hata ekranı gösterir.
   */
  fallback?: (error: Error, reset: () => void) => ReactNode;
  /**
   * Hata yakalandığında çağrılır. Sentry vb. servislere raporlama için
   * kullanışlı: <ErrorBoundary onError={(err, info) => Sentry.captureException(err)}>
   */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /**
   * Hata ekranı için dil. App kökündeki ErrorBoundary `useLanguage` hook'una
   * erişemez (çünkü kendisi React tree'sinin en üstünde) — o yüzden dil prop
   * olarak veriliyor. Verilmezse browser dili ile 'tr' fallback.
   */
  language?: 'en' | 'tr';
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

// Production'da hassas hata bilgilerini gizle
const IS_DEV = import.meta.env.DEV;

// Tarayıcı dilinden fallback dil tespiti
function detectLanguage(): 'en' | 'tr' {
  if (typeof navigator === 'undefined') return 'en';
  return navigator.language.toLowerCase().startsWith('tr') ? 'tr' : 'en';
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  /**
   * Child component render'ında hata olursa React bunu çağırır.
   * State'i güncelleyerek fallback UI'ı tetikleriz.
   */
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  /**
   * Hata yakalandığında side-effect çalıştırmak için kullanılır (loglama,
   * hata raporlama servisi vs). Render'ı etkilemez.
   */
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Bu console.error production'da bile kalır (vite.config'de pure'a
    // dahil etmedik) — hata raporlama için kritik.
    console.error('ErrorBoundary caught error:', error, errorInfo);

    // Sentry vb. servis entegrasyonu için hook
    this.props.onError?.(error, errorInfo);
  }

  /**
   * Reset — kullanıcı "Tekrar Dene" deyince state'i sıfırlayıp child
   * component'leri remount eder. Eğer hata persistent'sa (örn: bozuk
   * state) kullanıcı yine hata görecek, ama transient hatalardan kurtarır.
   */
  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (!this.state.hasError || !this.state.error) {
      return this.props.children;
    }

    // Özel fallback verilmişse onu kullan
    if (this.props.fallback) {
      return this.props.fallback(this.state.error, this.handleReset);
    }

    const lang = this.props.language ?? detectLanguage();
    const t = (en: string, tr: string) => (lang === 'tr' ? tr : en);

    return (
      <div
        role="alert"
        className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-slate-700"
      >
        <div className="max-w-lg w-full bg-white dark:bg-slate-800 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-red-500 to-orange-500 p-6 text-white">
            <div className="flex items-center gap-3">
              <div className="bg-white/20 backdrop-blur-sm rounded-full p-3">
                <AlertTriangle className="w-6 h-6" aria-hidden="true" />
              </div>
              <div>
                <h1 className="text-xl font-bold">
                  {t('Something went wrong', 'Bir şeyler ters gitti')}
                </h1>
                <p className="text-sm text-white/90">
                  {t(
                    'The application encountered an unexpected error',
                    'Uygulama beklenmedik bir hatayla karşılaştı',
                  )}
                </p>
              </div>
            </div>
          </div>

          {/* Body */}
          <div className="p-6 space-y-4">
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {t(
                "Don't worry — your data is safe. You can try the following:",
                'Merak etmeyin — verileriniz güvende. Şunları deneyebilirsiniz:',
              )}
            </p>

            {/* Development'ta stack trace göster. Production'da bu blok
                render edilmez — hassas iç işleyiş bilgisi leak olmasın. */}
            {IS_DEV && (
              <details className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                <summary className="cursor-pointer text-xs font-mono text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100">
                  {t('Error details (dev only)', 'Hata detayları (sadece dev)')}
                </summary>
                <div className="mt-3 space-y-2">
                  <div>
                    <div className="text-xs font-semibold text-red-600 dark:text-red-400 mb-1">
                      {this.state.error.name}:
                    </div>
                    <div className="text-xs font-mono text-slate-800 dark:text-slate-200 break-words">
                      {this.state.error.message}
                    </div>
                  </div>
                  {this.state.error.stack && (
                    <pre className="text-[10px] font-mono text-slate-500 dark:text-slate-400 overflow-x-auto max-h-48 whitespace-pre-wrap">
                      {this.state.error.stack}
                    </pre>
                  )}
                </div>
              </details>
            )}

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-2 pt-2">
              <Button
                onClick={this.handleReset}
                variant="default"
                className="flex-1"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                {t('Try Again', 'Tekrar Dene')}
              </Button>
              <Button
                onClick={this.handleReload}
                variant="outline"
                className="flex-1"
              >
                {t('Reload Page', 'Sayfayı Yenile')}
              </Button>
              <Button
                onClick={this.handleGoHome}
                variant="ghost"
                className="flex-1"
              >
                <Home className="w-4 h-4 mr-2" />
                {t('Home', 'Ana Sayfa')}
              </Button>
            </div>

            <p className="text-xs text-slate-400 dark:text-slate-500 text-center pt-2">
              {t(
                'If this problem persists, please contact support.',
                'Sorun devam ederse lütfen destek ekibiyle iletişime geçin.',
              )}
            </p>
          </div>
        </div>
      </div>
    );
  }
}