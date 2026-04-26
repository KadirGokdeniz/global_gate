import { useId } from 'react';

/**
 * Body arka planı için atmosferik katman.
 *
 * KAPSAM: Tüm viewport. Contrail'ler her yerde drift eder, ama App.tsx'te
 * Routes wrapper'ına verdiğimiz `relative z-10` sayesinde kartlar her
 * zaman contrail'lerin üstünde paint edilir. Sonuç: contrail'ler sadece
 * body-bg'sinin görünür olduğu BOŞLUK alanlarında görünür — kart
 * bölgelerinde otomatik gizlenir.
 *
 * İki katman:
 *   1) Contrails (üst-orta) — soldan-sağa-yukarı süzülen ince altın izler
 *   2) Horizon glow (alt) — sayfanın en altında çok soluk altın aydınlanma
 *
 * Performans: SVG path + CSS transform/opacity → GPU compositor.
 *
 * Erişilebilirlik:
 * - aria-hidden, pointer-events:none
 * - prefers-reduced-motion → animasyon durur, görsel kalır
 */
export const BackgroundFX = () => {
  const gradId = useId();

  return (
    <div
      aria-hidden="true"
      className="fixed inset-0 pointer-events-none overflow-hidden"
    >
      {/* ─── Contrails — diyagonal süzülen altın izler ─────────────────── */}
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient
            id={`contrail-${gradId}`}
            x1="0%"
            y1="100%"
            x2="100%"
            y2="0%"
          >
            <stop
              offset="0%"
              style={{ stopColor: 'hsl(var(--accent))', stopOpacity: 0 }}
            />
            <stop
              offset="50%"
              style={{ stopColor: 'hsl(var(--accent))', stopOpacity: 1 }}
            />
            <stop
              offset="100%"
              style={{ stopColor: 'hsl(var(--accent))', stopOpacity: 0 }}
            />
          </linearGradient>
        </defs>

        {/* Path 1 — alçak, hafif kavis, en yavaş (95s) */}
        <g className="contrail contrail-slow">
          <path
            d="M -10 95 Q 50 70 110 30"
            fill="none"
            stroke={`url(#contrail-${gradId})`}
            strokeWidth="2.5"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </g>

        {/* Path 2 — orta, gecikmeli (75s) */}
        <g className="contrail contrail-medium">
          <path
            d="M -10 70 Q 40 55 110 15"
            fill="none"
            stroke={`url(#contrail-${gradId})`}
            strokeWidth="2"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </g>

        {/* Path 3 — yüksek, en hızlı (60s) */}
        <g className="contrail contrail-fast">
          <path
            d="M -10 50 Q 30 45 110 5"
            fill="none"
            stroke={`url(#contrail-${gradId})`}
            strokeWidth="1.5"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </g>
      </svg>

      {/* ─── Alt horizon glow — sayfanın altında çok soluk altın aydınlanma */}
      <div
        className="absolute inset-x-0 bottom-0 h-[40vh]"
        style={{
          background:
            'radial-gradient(ellipse 65% 100% at 50% 100%, hsl(var(--accent) / 0.08) 0%, transparent 70%)',
        }}
      />
    </div>
  );
};