import { Toaster } from '@/components/ui/toaster';
import { Toaster as Sonner } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import Index from './pages/Index';
import NotFound from './pages/NotFound';

const queryClient = new QueryClient();

/**
 * Nested Error Boundary stratejisi:
 *
 * - Dış (root) boundary: Provider'lardan biri (QueryClient, Router vs.)
 *   veya toast sistemi çökerse yakalar. Burada hata olursa tüm app
 *   fallback UI göstermek zorunda — zaten başka çare yok.
 *
 * - İç (route) boundary: Sadece page component'i (Index/NotFound) çökerse
 *   yakalar. Provider'lar ayakta kalır, kullanıcı "Tekrar Dene" deyince
 *   app'i baştan yüklemeden page remount olur.
 *
 * Böylece bir sayfa crash'i diğerini etkilemez ve transient hatalardan
 * kurtulma yolu kolaylaşır.
 */
const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Index />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </ErrorBoundary>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;