import { useId } from 'react';

/**
 * Body arka planı için atmosferik katman.
 *
 * KRİTİK TASARIM KARARI: Contrail'ler viewport'un sadece ÜST %50'sinde
 * dolaşır. Sebep: tam ekran versiyonunda kartların altından/üstünden
 * geçince görsel kirlilik yaratıyordu. Üst banda sıkıştırınca:
 * - Header altında "gökyüzü" hissi olur (ufuk metaforu doğru durur)
 * - Yoğun içerik bölgesi (alt %50) tertemiz kalır
 * - Mobil + masaüstünde tutarlı çalışır
 *
 * Performans: SVG path + CSS transform/opacity → GPU compositor.
 *
 * Erişilebilirlik:
 * - aria-hidden, pointer-events:none
 * - prefers-reduced-motion: index.css'te animasyon kapanır,
 *   contrail'ler statik durur (görsel derinlik korunur)
 *
 * Z-index: NEGATIF DEĞİL. Negatif z-index body bg'sinin altına itiyordu.
 * Bunun yerine document order'a güveniyoruz — BackgroundFX, Routes'tan
 * önce render edildiği için arkada kalır.
 */
export const BackgroundFX = () => {
  const gradId = useId();

  return (
    <div
      aria-hidden="true"
      className="fixed top-0 inset-x-0 h-[50vh] pointer-events-none overflow-hidden"
    >
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        <defs>
          {/* Stroke gradient — başlangıç ve bitiş noktaları görünmez,
              ortası belirgin. CSS değişkenleri SVG attribute'ta her
              tarayıcıda resolve olmuyor; style kullanıyoruz. */}
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

        {/* Path 2 — orta yükseklik, orta hız (75s, gecikmeli) */}
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

        {/* Path 3 — yüksek, en hızlı (60s, en gecikmeli) */}
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
    </div>
  );
};