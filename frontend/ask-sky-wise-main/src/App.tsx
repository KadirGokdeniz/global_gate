import * as Sentry from '@sentry/react';
import { Toaster } from '@/components/ui/toaster';
import { Toaster as Sonner } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { BackgroundFX } from '@/components/BackgroundFX';   // ← YENİ SATIR
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

const App = () => (
  <ErrorBoundary onError={handleErrorBoundaryError}>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <ErrorBoundary onError={handleErrorBoundaryError}>
            <BackgroundFX />                                 {/* ← YENİ SATIR */}
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/privacy" element={<PrivacyPage />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </ErrorBoundary>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;