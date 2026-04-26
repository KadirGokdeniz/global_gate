import { useId } from 'react';

/**
 * Body arka planı için atmosferik katman.
 *
 * Tasarım dili: favicon'daki altın kağıt uçak ile uyumlu — soldan sağa-yukarı
 * doğru süzülen ince altın "uçuş izleri" (contrails). Çok yavaş, çok soluk,
 * marka rengiyle (`--accent`).
 *
 * Performans:
 * - Sadece SVG path + CSS `transform` ve `opacity` animasyonu → GPU compositor.
 * - Layout/paint thrashing yok, her frame composite-only.
 * - `will-change` set edilmiş; tek seferlik mount, app boyunca sabit.
 *
 * Erişilebilirlik:
 * - `aria-hidden` → ekran okuyucu görmezden gelir.
 * - `pointer-events: none` → tıklamaları engellemez.
 * - `motion-reduce:hidden` → `prefers-reduced-motion: reduce` aktifken
 *   component hiç render edilmez (Tailwind utility'si).
 *
 * Mount noktası:
 * App.tsx içinde, BrowserRouter'ın iç ErrorBoundary'sinin altında, Routes'un
 * SİBLİNG'i olarak — böylece tüm route'larda görünür ve route değişince
 * remount olmaz (sürekliliği bozmaz).
 */
export const BackgroundFX = () => {
  // SSR/hydration güvenli, çoklu instance çakışmasını önleyen unique gradient ID
  const gradId = useId();

  return (
    <div
      aria-hidden="true"
      className="fixed inset-0 -z-10 pointer-events-none overflow-hidden motion-reduce:hidden"
    >
      {/* ─── Contrails — diyagonal süzülen altın izler ─────────────────── */}
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        <defs>
          {/* Stroke gradient: kenarlardan başlangıç/bitiş yumuşak, ortada
              belirgin. Çizginin başı ve sonu görünmez şekilde silinsin diye. */}
          <linearGradient
            id={`contrail-${gradId}`}
            x1="0%"
            y1="100%"
            x2="100%"
            y2="0%"
          >
            <stop offset="0%" stopColor="hsl(var(--accent))" stopOpacity="0" />
            <stop offset="50%" stopColor="hsl(var(--accent))" stopOpacity="1" />
            <stop offset="100%" stopColor="hsl(var(--accent))" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Path 1 — alçak, hafif kavis, en yavaş (95s) */}
        <g className="contrail contrail-slow">
          <path
            d="M -10 95 Q 50 70 110 30"
            fill="none"
            stroke={`url(#contrail-${gradId})`}
            strokeWidth="1.5"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </g>

        {/* Path 2 — orta yükseklik, daha düz, orta hız (75s, gecikmeli başlar) */}
        <g className="contrail contrail-medium">
          <path
            d="M -10 70 Q 40 55 110 15"
            fill="none"
            stroke={`url(#contrail-${gradId})`}
            strokeWidth="1.25"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </g>

        {/* Path 3 — yüksek, en kavisli, en hızlı (60s, daha fazla gecikmeli) */}
        <g className="contrail contrail-fast">
          <path
            d="M -10 50 Q 30 45 110 5"
            fill="none"
            stroke={`url(#contrail-${gradId})`}
            strokeWidth="1"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </g>
      </svg>

      {/* ─── Horizon glow — alt kısımda çok soluk altın aydınlanma ──────
          Contrail'lerin geldiği "ufuk" hissini güçlendirir. Statik. */}
      <div
        className="absolute inset-x-0 bottom-0 h-[45vh]"
        style={{
          background:
            'radial-gradient(ellipse 65% 100% at 50% 100%, hsl(var(--accent) / 0.07) 0%, transparent 70%)',
        }}
      />
    </div>
  );
};
