import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * localStorage'a state persist eden generic hook.
 *
 * Özellikler:
 * - SSR-safe: window yoksa initial değeri kullanır
 * - Quota hatalarını sessizce yutup console.warn eder (app crash etmez)
 * - Schema migration için opsiyonel validator
 * - Cross-tab senkronizasyon: başka bir tab'da değişirse bu tab da güncellenir
 *
 * @param key localStorage anahtarı. Uygulamanız için eşsiz olmalı.
 * @param initialValue İlk render'da ve localStorage boşken/bozukken kullanılır.
 * @param validator Opsiyonel: parse edilen değeri doğrulamak için.
 *                  Dönerse `true` → değer kullanılır. `false` → initialValue'ya düşer.
 *                  Schema migration'da yararlı: `(v) => typeof v === 'object' && v.version === 2`
 */
export function useLocalStorage<T>(
  key: string,
  initialValue: T,
  validator?: (value: unknown) => boolean,
): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  // SSR guard: window yoksa initial değer
  const isClient = typeof window !== 'undefined';

  // İlk değeri localStorage'tan oku (useState initializer — senkron, flicker yok)
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (!isClient) return initialValue;

    try {
      const item = window.localStorage.getItem(key);
      if (item === null) return initialValue;

      const parsed = JSON.parse(item);

      // Validator varsa kontrol et — fail olursa initial'a dön (schema mismatch)
      if (validator && !validator(parsed)) {
        console.warn(
          `[useLocalStorage] Validation failed for key "${key}", using initial value`,
        );
        return initialValue;
      }

      return parsed as T;
    } catch (error) {
      console.warn(
        `[useLocalStorage] Failed to read key "${key}":`,
        error,
      );
      return initialValue;
    }
  });

  // Son yazılan değeri tutuyoruz ki storage event'inde kendi yazdığımız
  // değişikliği tekrar parse etmeyelim (gereksiz re-render önlemi)
  const lastWrittenRef = useRef<string | null>(null);

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      setStoredValue((prev) => {
        // Functional update desteği
        const resolved =
          typeof value === 'function'
            ? (value as (p: T) => T)(prev)
            : value;

        if (isClient) {
          try {
            const serialized = JSON.stringify(resolved);
            lastWrittenRef.current = serialized;
            window.localStorage.setItem(key, serialized);
          } catch (error) {
            // QuotaExceededError veya serialization hatası
            console.warn(
              `[useLocalStorage] Failed to write key "${key}":`,
              error,
            );
            // State yine de güncellenmiş olur; sadece persistence fail
          }
        }

        return resolved;
      });
    },
    [key, isClient],
  );

  const removeValue = useCallback(() => {
    if (!isClient) return;
    try {
      window.localStorage.removeItem(key);
      lastWrittenRef.current = null;
      setStoredValue(initialValue);
    } catch (error) {
      console.warn(
        `[useLocalStorage] Failed to remove key "${key}":`,
        error,
      );
    }
  }, [key, isClient, initialValue]);

  // Cross-tab sync: başka bir tab localStorage'ı değiştirdiyse bu tab'ı güncelle
  useEffect(() => {
    if (!isClient) return;

    const handleStorageChange = (e: StorageEvent) => {
      // Sadece bizim key değişmişse ve bizim yazmadığımız bir değişiklikse
      if (e.key !== key) return;
      if (e.newValue === lastWrittenRef.current) return;

      try {
        if (e.newValue === null) {
          setStoredValue(initialValue);
          return;
        }
        const parsed = JSON.parse(e.newValue);
        if (validator && !validator(parsed)) return;
        setStoredValue(parsed as T);
      } catch {
        // Parse hatası — görmezden gel
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, isClient]);

  return [storedValue, setValue, removeValue];
}
