import * as Sentry from '@sentry/react';
import { Toaster } from '@/components/ui/toaster';
import { Toaster as Sonner } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { BackgroundFX } from '@/components/BackgroundFX';
import Index from './pages/Index';
import { PrivacyPage } from './pages/PrivacyPage';
import NotFound from './pages/NotFound';

const queryClient = new QueryClient();

const handleErrorBoundaryError = (error: Error, errorInfo: { componentStack: string }) => {
  Sentry.captureException(error, {
    contexts: {
      react: {
        componentStack: errorInfo.componentStack,
      },
    },
  });
};

/**
 * Z-INDEX YAPISI:
 * BackgroundFX (fixed) varsayılan olarak normal flow'daki kartlardan SONRA
 * paint ediliyor — yani contrail'ler kartların üstüne çıkıyordu. Bunu
 * önlemek için Routes'u `relative z-10` wrapper ile sarıyoruz: bu wrapper
 * yeni bir stacking context yaratır ve tüm route içeriği BackgroundFX'in
 * (z-auto) üstünde paint edilir. Sonuç: contrail'ler arka planda kalır,
 * kart bölgelerinde otomatik gizlenir, sadece body-bg boşluklarında görünür.
 */
const App = () => (
  <ErrorBoundary onError={handleErrorBoundaryError}>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter basename="/global_gate">
          <ErrorBoundary onError={handleErrorBoundaryError}>
            <BackgroundFX />
            <div className="relative z-10">
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/privacy" element={<PrivacyPage />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </div>
          </ErrorBoundary>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;