import { createRoot } from 'react-dom/client';
import * as Sentry from '@sentry/react';
import App from './App.tsx';
import './index.css';

// ═══════════════════════════════════════════════════════════════════
// Sentry — Sadece production build'de ve DSN varsa aktif
// Dev modunda (npm run dev) init edilmez, console'u kirletmez
// ═══════════════════════════════════════════════════════════════════
if (import.meta.env.PROD && import.meta.env.VITE_SENTRY_DSN) {
  Sentry.init({
    dsn: import.meta.env.VITE_SENTRY_DSN,
    environment: import.meta.env.MODE,

    // Performance monitoring — free tier dostu oran
    tracesSampleRate: 0.1,

    // Session replay kapalı — kota korunsun
    replaysSessionSampleRate: 0,
    // Hata anında replay al — debugging için çok değerli
    replaysOnErrorSampleRate: 0.1,

    // PII (personal info) gönderme
    sendDefaultPii: false,

    // Hassas header'ları filtrele
    beforeSend(event) {
      if (event.request?.headers) {
        const headers = event.request.headers as Record<string, string>;
        for (const key of Object.keys(headers)) {
          if (
            key.toLowerCase().includes('authorization') ||
            key.toLowerCase().includes('api-key')
          ) {
            headers[key] = '[FILTERED]';
          }
        }
      }
      return event;
    },
  });
}

createRoot(document.getElementById('root')!).render(<App />);