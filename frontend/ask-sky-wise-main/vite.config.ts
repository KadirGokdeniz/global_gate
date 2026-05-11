import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { componentTagger } from 'lovable-tagger';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: '::',
    port: 8080,
  },
  base: '/global_gate/',
  plugins: [
    react(),
    mode === 'development' && componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // ✅ Production build optimizasyonları
  esbuild: {
    // Production'da 'debugger' statement'larını bundle'dan kaldır
    drop: mode === 'production' ? ['debugger'] : [],
    // console.log, console.debug, console.info çağrılarını "pure" olarak
    // işaretle — sonuçları kullanılmadığı için minifier bunları tree-shake
    // edebilir. console.error ve console.warn KALIR — production'da gerçek
    // hatalar görünmeli (Sentry vb. servisler yakalayabilsin).
    pure: mode === 'production'
      ? ['console.log', 'console.debug', 'console.info', 'console.trace']
      : [],
  },
  build: {
    // Source map'leri production'da kapat — bundle size ve debugging
    // trade-off'u. Sentry kullanacaksan 'hidden' yap (source map üretilir
    // ama bundle'a referans verilmez; sadece Sentry'e upload edilir).
    sourcemap: false,
    // Chunk boyutu uyarı eşiği — 500kb default, bizim app'imiz şu an
    // bundle size için Recharts/Embla gibi ağır dep'ler var (P1 #11'de
    // temizleyeceğiz). Uyarıyı şimdilik 800kb'a çekiyoruz.
    chunkSizeWarningLimit: 800,
  },
}));