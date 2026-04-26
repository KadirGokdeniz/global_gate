import { useId } from 'react';

/**
 * Body arka planı için atmosferik katman.
 *
 * Tasarım dili: favicon'daki altın kağıt uçak ile uyumlu — soldan sağa-yukarı
 * doğru süzülen ince altın "uçuş izleri" (contrails). Çok yavaş, çok soluk,
 * marka rengiyle (`--accent`).
 *
 * Z-INDEX NOTU:
 * Negatif z-index KULLANMIYORUZ. Sebep: body'nin background-color'ı CSS
 * stacking context'inde negatif z-indexli fixed elementleri kapatabiliyor
 * (browser implementation farkı). Bunun yerine wrapper'ı z-index olmadan
 * fixed konumda bırakıp, document order'a güveniyoruz: BackgroundFX,
 * Routes'tan ÖNCE render edildiği için arkada kalır. `pointer-events:none`
 * sayesinde interaksiyonu engellemez.
 *
 * Performans: SVG path + CSS transform/opacity → GPU compositor.
 *
 * Erişilebilirlik:
 * - `aria-hidden` → ekran okuyucu görmezden gelir.
 * - `pointer-events: none` → tıklamaları engellemez.
 * - Reduced-motion: index.css'teki @media query animasyonu durdurur ama
 *   contrail'ler statik olarak görünmeye devam eder (görsel derinlik kalır).
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
          {/* Stroke gradient: kenarlardan başlangıç/bitiş yumuşak, ortada
              belirgin. style attribute kullanıyoruz (SVG attribute'ta CSS
              variable bazı tarayıcılarda resolve olmuyor — defansif tercih). */}
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

        {/* Path 2 — orta yükseklik, daha düz, orta hız (75s, gecikmeli başlar) */}
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

        {/* Path 3 — yüksek, en kavisli, en hızlı (60s, daha fazla gecikmeli) */}
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

      {/* ─── Horizon glow — alt kısımda çok soluk altın aydınlanma ────── */}
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