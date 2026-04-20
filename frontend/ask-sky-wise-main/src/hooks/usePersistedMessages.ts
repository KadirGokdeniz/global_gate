import { useState, useCallback, useRef, useEffect } from 'react';
import { Message, AirlinePreference } from '@/types';

// Her havayolu için ayrı mesaj geçmişi
export type AirlineMessages = Record<AirlinePreference, Message[]>;

/**
 * Schema versiyonu. Message interface'ine breaking change yaparsanız
 * bu sayıyı artırın — kullanıcıdaki eski veri otomatik discard edilir.
 */
const SCHEMA_VERSION = 1;

/**
 * Her havayolu için tutulacak maksimum mesaj sayısı. Üstüne çıkılırsa
 * en eskiden kırpılır. localStorage quota (~5MB) dolmasın.
 */
const MAX_MESSAGES_PER_AIRLINE = 100;

const STORAGE_KEY = 'airline-assistant:messages';

interface PersistedPayload {
  version: number;
  messages: AirlineMessages;
}

const EMPTY_MESSAGES: AirlineMessages = { thy: [], pegasus: [] };

/**
 * localStorage'tan mesajları geri yüklerken Date objelerini revive eder.
 * JSON.parse Date'i otomatik Date objesine çevirmez — string bırakır.
 */
function reviveMessages(raw: unknown): AirlineMessages | null {
  if (!raw || typeof raw !== 'object') return null;

  const payload = raw as Partial<PersistedPayload>;

  if (payload.version !== SCHEMA_VERSION) {
    // Eski şema — discard et
    return null;
  }

  if (!payload.messages) return null;

  const revived: AirlineMessages = { thy: [], pegasus: [] };

  for (const airline of ['thy', 'pegasus'] as const) {
    const list = payload.messages[airline];
    if (!Array.isArray(list)) continue;

    revived[airline] = list
      .map((msg) => {
        if (!msg || typeof msg !== 'object') return null;

        // Date'i revive et
        const timestamp =
          msg.timestamp instanceof Date
            ? msg.timestamp
            : new Date(msg.timestamp);

        // Invalid date guard
        if (isNaN(timestamp.getTime())) return null;

        return { ...msg, timestamp } as Message;
      })
      .filter((msg): msg is Message => msg !== null);
  }

  return revived;
}

/**
 * Havayolu mesajlarını localStorage'da persist eder.
 * - Date objeleri düzgün serialize/revive edilir
 * - Schema versioning: breaking change durumunda eski veri discard edilir
 * - Mesaj sayısı MAX_MESSAGES_PER_AIRLINE ile sınırlıdır
 * - Quota dolarsa fail-silent: app çalışmaya devam eder, sadece persist etmez
 */
export function usePersistedMessages() {
  const isClient = typeof window !== 'undefined';

  const [messages, setMessages] = useState<AirlineMessages>(() => {
    if (!isClient) return EMPTY_MESSAGES;

    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return EMPTY_MESSAGES;

      const parsed = JSON.parse(raw);
      const revived = reviveMessages(parsed);
      return revived ?? EMPTY_MESSAGES;
    } catch (error) {
      console.warn('[persistedMessages] Failed to load:', error);
      return EMPTY_MESSAGES;
    }
  });

  // Değişiklikte localStorage'a yaz — debounce ile (her mesaj setter'ında
  // hemen yazmak yerine, hızlı ardışık değişiklikleri batch'le)
  const writeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!isClient) return;

    if (writeTimeoutRef.current) clearTimeout(writeTimeoutRef.current);

    writeTimeoutRef.current = setTimeout(() => {
      try {
        const payload: PersistedPayload = {
          version: SCHEMA_VERSION,
          messages,
        };
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      } catch (error) {
        console.warn('[persistedMessages] Failed to save:', error);
      }
    }, 300);

    return () => {
      if (writeTimeoutRef.current) clearTimeout(writeTimeoutRef.current);
    };
  }, [messages, isClient]);

  /**
   * Mesaj ekle. Havayolu listesi MAX_MESSAGES_PER_AIRLINE'ı aşarsa en
   * eskiler otomatik kırpılır.
   */
  const addMessage = useCallback(
    (airline: AirlinePreference, message: Message) => {
      setMessages((prev) => {
        const current = prev[airline];
        const next = [...current, message];

        // Limit: en eskiden kırp
        const trimmed =
          next.length > MAX_MESSAGES_PER_AIRLINE
            ? next.slice(-MAX_MESSAGES_PER_AIRLINE)
            : next;

        return { ...prev, [airline]: trimmed };
      });
    },
    [],
  );

  /**
   * Bir havayolunun geçmişini temizle.
   */
  const clearAirline = useCallback((airline: AirlinePreference) => {
    setMessages((prev) => ({ ...prev, [airline]: [] }));
  }, []);

  /**
   * Tüm havayollarının geçmişini temizle.
   */
  const clearAll = useCallback(() => {
    setMessages(EMPTY_MESSAGES);
  }, []);

  return {
    messages,
    addMessage,
    clearAirline,
    clearAll,
  };
}
