import { Button } from '@/components/ui/button';
import {
  ThumbsUp,
  ThumbsDown,
  Clock,
  AlertTriangle,
  Volume2,
  VolumeX,
  Play,
  Pause,
  Square,
  ExternalLink,
  ChevronDown,
  Copy,
  Check,
} from 'lucide-react';
import { Message, FeedbackType, Language } from '@/types';
import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';

interface ResponseCardProps {
  message: Message;
  language: Language;
  onFeedback: (messageId: string, feedback: FeedbackType) => void;
  onPlayAudio?: (text: string) => Promise<string | null>;
  feedbackGiven?: FeedbackType | null;
}

const log = import.meta.env.DEV ? console.log : () => {};
const logError = console.error;

export const ResponseCard = ({
  message,
  language,
  onFeedback,
  onPlayAudio,
  feedbackGiven,
}: ResponseCardProps) => {
  const [audioState, setAudioState] = useState<
    'idle' | 'loading' | 'playing' | 'paused' | 'error'
  >('idle');
  const [audioProgress, setAudioProgress] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [feedbackLoading, setFeedbackLoading] = useState<FeedbackType | null>(
    null,
  );
  const [expandedSources, setExpandedSources] = useState<number[]>([]);
  const [showAudioPanel, setShowAudioPanel] = useState(false);
  const [copied, setCopied] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // TTS memory leak fix (önceki çalışmamızdan korundu)
  const audioUrlRef = useRef<string | null>(null);

  const isEn = language === 'en';

  const revokeAudioUrl = useCallback(() => {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
  }, []);

  const teardownAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.removeAttribute('src');
      audioRef.current.load();
      audioRef.current = null;
    }
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  }, []);

  const toggleSource = (index: number) => {
    setExpandedSources((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index],
    );
  };

  useEffect(() => {
    return () => {
      teardownAudio();
      revokeAudioUrl();
    };
  }, [teardownAudio, revokeAudioUrl]);

  useEffect(() => {
    return () => {
      revokeAudioUrl();
    };
  }, [message.answer, revokeAudioUrl]);

  const startProgressTracking = () => {
    if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    progressIntervalRef.current = setInterval(() => {
      if (audioRef.current) {
        const currentTime = audioRef.current.currentTime;
        const duration = audioRef.current.duration;
        if (duration > 0 && !isNaN(duration)) {
          setAudioProgress((currentTime / duration) * 100);
        }
      }
    }, 100);
  };

  const stopProgressTracking = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  const attachAudioListeners = (audio: HTMLAudioElement) => {
    audio.addEventListener('loadedmetadata', () => {
      setAudioDuration(audio.duration);
      setAudioProgress(0);
    });
    audio.addEventListener('play', () => {
      setAudioState('playing');
      startProgressTracking();
    });
    audio.addEventListener('pause', () => {
      if (audioState !== 'idle') setAudioState('paused');
      stopProgressTracking();
    });
    audio.addEventListener('ended', () => {
      setAudioState('idle');
      setAudioProgress(0);
      stopProgressTracking();
    });
    audio.addEventListener('error', () => {
      logError('TTS audio error:', audio.error);
      setAudioState('error');
      stopProgressTracking();
    });
  };

  const handlePlayAudio = async () => {
    if (!onPlayAudio) return;
    setShowAudioPanel(true);

    try {
      if (audioRef.current && audioState === 'paused') {
        await audioRef.current.play();
        return;
      }

      if (audioUrlRef.current && audioState === 'idle') {
        teardownAudio();
        const audio = new Audio(audioUrlRef.current);
        audioRef.current = audio;
        attachAudioListeners(audio);
        await audio.play();
        return;
      }

      setAudioState('loading');
      if (audioUrlRef.current) revokeAudioUrl();

      const audioUrl = await onPlayAudio(message.answer);
      if (!audioUrl) {
        setAudioState('error');
        return;
      }

      teardownAudio();
      audioUrlRef.current = audioUrl;
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      attachAudioListeners(audio);

      try {
        await audio.play();
      } catch (playErr) {
        logError('play() rejected:', playErr);
        setAudioState('error');
      }
    } catch (error) {
      logError('handlePlayAudio error:', error);
      setAudioState('error');
    }
  };

  const handlePauseAudio = () => {
    if (audioRef.current && audioState === 'playing') audioRef.current.pause();
  };

  const handleStopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setAudioState('idle');
      setAudioProgress(0);
      stopProgressTracking();
    }
  };

  const handleSeekAudio = (percentage: number) => {
    if (audioRef.current && audioDuration > 0) {
      const newTime = (percentage / 100) * audioDuration;
      audioRef.current.currentTime = newTime;
      setAudioProgress(percentage);
    }
  };

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API desteklenmiyorsa (eski tarayıcı / HTTP) sessizce geç
    }
  }, [message.answer]);

  const formatTime = (seconds: number) => {
    if (!isFinite(seconds) || seconds < 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleFeedbackClick = async (type: FeedbackType) => {
    try {
      setFeedbackLoading(type);
      await onFeedback(message.id, type);
    } catch (error) {
      logError('Feedback error:', error);
    } finally {
      setFeedbackLoading(null);
    }
  };

  const formatTimestamp = (date: Date) => {
    return new Intl.DateTimeFormat(isEn ? 'en-US' : 'tr-TR', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  // Feedback buttons — sakin, tek sıra
  const feedbackButtons: {
    type: FeedbackType;
    icon: typeof ThumbsUp;
    label: { en: string; tr: string };
  }[] = [
    { type: 'helpful', icon: ThumbsUp, label: { en: 'Helpful', tr: 'Yardımcı' } },
    {
      type: 'not_helpful',
      icon: ThumbsDown,
      label: { en: 'Not helpful', tr: 'Değil' },
    },
    {
      type: 'too_slow',
      icon: Clock,
      label: { en: 'Slow', tr: 'Yavaş' },
    },
    {
      type: 'incorrect',
      icon: AlertTriangle,
      label: { en: 'Incorrect', tr: 'Yanlış' },
    },
  ];

  return (
    <article className="w-full max-w-3xl mx-auto">
      {/* ─── Question ──────────────────────────────────────────────
          Diyalog akışı için sorunun kendisi bir kart değil, sadece
          üstte yer alan bir "label". */}
      <div className="mb-3 px-4">
        <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
          {isEn ? 'Question' : 'Soru'} · {formatTimestamp(message.timestamp)}
        </div>
        <div className="text-sm text-slate-700 dark:text-slate-300">
          {message.question}
        </div>
      </div>

      {/* ─── Answer Card ───────────────────────────────────────────
          Ana içerik tek bir temiz kart içinde. Renkli badge'ler,
          gradient'ler yok. */}
      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-layer-sm overflow-hidden">
        {/* Answer body */}
        <div className="p-5 sm:p-6">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mt-0.5">
              {isEn ? 'Answer' : 'Cevap'}
            </div>

            {/* Action buttons — copy + audio, sağ üst köşede */}
            <div className="flex items-center gap-1 shrink-0">
              {/* Copy button */}
              <button
                type="button"
                onClick={handleCopy}
                aria-label={
                  copied
                    ? isEn ? 'Copied!' : 'Kopyalandı!'
                    : isEn ? 'Copy answer' : 'Yanıtı kopyala'
                }
                title={
                  copied
                    ? isEn ? 'Copied!' : 'Kopyalandı!'
                    : isEn ? 'Copy answer' : 'Yanıtı kopyala'
                }
                className={`h-7 w-7 flex items-center justify-center rounded-md transition-all duration-150 ${
                  copied
                    ? 'text-emerald-600 dark:text-emerald-500 bg-emerald-50 dark:bg-emerald-950/40'
                    : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>

              {/* Audio button */}
              {onPlayAudio && (
                <button
                  type="button"
                  onClick={
                    audioState === 'playing' ? handlePauseAudio : handlePlayAudio
                  }
                  disabled={audioState === 'loading'}
                  aria-label={
                    audioState === 'playing'
                      ? isEn ? 'Pause audio' : 'Duraklat'
                      : isEn ? 'Listen to answer' : 'Cevabı dinle'
                  }
                  title={
                    audioState === 'playing'
                      ? isEn ? 'Pause' : 'Duraklat'
                      : isEn ? 'Listen' : 'Dinle'
                  }
                  className="h-7 w-7 flex items-center justify-center rounded-md text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 transition-colors"
                >
                  {audioState === 'loading' ? (
                    <div className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  ) : audioState === 'playing' ? (
                    <Pause className="w-3.5 h-3.5" />
                  ) : audioState === 'error' ? (
                    <VolumeX className="w-3.5 h-3.5 text-red-500" />
                  ) : (
                    <Volume2 className="w-3.5 h-3.5" />
                  )}
                </button>
              )}
            </div>
          </div>

          {/* ─── Markdown renderer ─────────────────────────────
              LLM yanıtlarındaki **bold**, *italic*, listeler,
              kod blokları, başlıklar vs. düzgün render edilsin. */}
          <div className="markdown-body text-[15px] leading-relaxed text-slate-900 dark:text-slate-100">
                <ReactMarkdown
                  components={{
                    // ─── Paragraf
                    p: ({ children }) => (
                      <p className="mb-3 last:mb-0 leading-relaxed">
                        {children}
                      </p>
                    ),
                    // ─── Başlıklar — hiyerarşi sıkı tracking'le
                    h1: ({ children }) => (
                      <h2 className="text-lg font-semibold mb-3 mt-5 first:mt-0 tracking-tight text-slate-900 dark:text-slate-100">
                        {children}
                      </h2>
                    ),
                    h2: ({ children }) => (
                      <h3 className="text-base font-semibold mb-2 mt-4 first:mt-0 tracking-tight text-slate-900 dark:text-slate-100">
                        {children}
                      </h3>
                    ),
                    h3: ({ children }) => (
                      <h4 className="text-[15px] font-semibold mb-2 mt-3 first:mt-0 text-slate-900 dark:text-slate-100">
                        {children}
                      </h4>
                    ),
                    // ─── Bullet list
                    ul: ({ children }) => (
                      <ul className="mb-3 last:mb-0 space-y-1 pl-0">
                        {children}
                      </ul>
                    ),
                    // ─── Numbered list
                    ol: ({ children }) => (
                      <ol className="mb-3 last:mb-0 space-y-1 pl-0 list-none counter-reset-[item]">
                        {children}
                      </ol>
                    ),
                    // ─── List item — custom bullet ile
                    li: ({ children, ...props }) => {
                      const isOrdered = (props as any).ordered;
                      return (
                        <li className="flex items-start gap-2 text-[15px]">
                          <span
                            className="mt-[0.35em] shrink-0 text-slate-400 dark:text-slate-500 select-none"
                            aria-hidden="true"
                          >
                            {isOrdered ? '›' : '·'}
                          </span>
                          <span className="flex-1">{children}</span>
                        </li>
                      );
                    },
                    // ─── Inline code
                    code: ({ inline, children, ...props }: any) =>
                      inline ? (
                        <code className="px-1.5 py-0.5 rounded text-[13px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700">
                          {children}
                        </code>
                      ) : (
                        <code
                          className="block px-4 py-3 rounded-lg text-[13px] font-mono bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-800 overflow-x-auto leading-relaxed"
                          {...props}
                        >
                          {children}
                        </code>
                      ),
                    // ─── Code block wrapper
                    pre: ({ children }) => (
                      <pre className="mb-3 last:mb-0 rounded-lg overflow-hidden">
                        {children}
                      </pre>
                    ),
                    // ─── Blockquote — önemli notlar için
                    blockquote: ({ children }) => (
                      <blockquote className="pl-4 border-l-2 border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-400 italic mb-3 last:mb-0">
                        {children}
                      </blockquote>
                    ),
                    // ─── Bold
                    strong: ({ children }) => (
                      <strong className="font-semibold text-slate-900 dark:text-slate-100">
                        {children}
                      </strong>
                    ),
                    // ─── Italic
                    em: ({ children }) => (
                      <em className="italic text-slate-700 dark:text-slate-300">
                        {children}
                      </em>
                    ),
                    // ─── Link — dışarı açılan
                    a: ({ href, children }) => (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-slate-900 dark:text-slate-100 underline underline-offset-2 decoration-slate-300 dark:decoration-slate-600 hover:decoration-slate-600 dark:hover:decoration-slate-300 transition-colors"
                      >
                        {children}
                      </a>
                    ),
                    // ─── Yatay çizgi — bölüm ayırıcı
                    hr: () => (
                      <hr className="my-4 border-slate-200 dark:border-slate-800" />
                    ),
                    // ─── Tablo — havayolu politikalarında sık görünür
                    table: ({ children }) => (
                      <div className="mb-3 last:mb-0 overflow-x-auto">
                        <table className="w-full text-sm border-collapse">
                          {children}
                        </table>
                      </div>
                    ),
                    thead: ({ children }) => (
                      <thead className="border-b border-slate-200 dark:border-slate-800">
                        {children}
                      </thead>
                    ),
                    th: ({ children }) => (
                      <th className="px-3 py-2 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                        {children}
                      </th>
                    ),
                    td: ({ children }) => (
                      <td className="px-3 py-2 text-[14px] border-b border-slate-100 dark:border-slate-800/50">
                        {children}
                      </td>
                    ),
                    tr: ({ children }) => (
                      <tr className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
                        {children}
                      </tr>
                    ),
                  }}
                >
                  {message.answer}
                </ReactMarkdown>
          </div>

          {/* Audio player expanded panel — sadece aktifken */}
          {showAudioPanel && audioState !== 'idle' && audioState !== 'error' && (
            <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-3">
                <span className="text-xs text-slate-500 dark:text-slate-400 tabular-nums min-w-[36px]">
                  {formatTime((audioProgress / 100) * audioDuration)}
                </span>

                <div
                  className="flex-1 h-1 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden cursor-pointer"
                  role="slider"
                  tabIndex={0}
                  aria-label={isEn ? 'Audio progress' : 'Ses ilerlemesi'}
                  aria-valuenow={Math.round(audioProgress)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const pct =
                      ((e.clientX - rect.left) / rect.width) * 100;
                    handleSeekAudio(Math.max(0, Math.min(100, pct)));
                  }}
                >
                  <div
                    className="h-full bg-slate-900 dark:bg-slate-100 transition-all duration-100 ease-out"
                    style={{ width: `${audioProgress}%` }}
                  />
                </div>

                <span className="text-xs text-slate-500 dark:text-slate-400 tabular-nums min-w-[36px]">
                  {formatTime(audioDuration)}
                </span>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleStopAudio}
                  className="h-7 w-7 p-0 text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
                  aria-label={isEn ? 'Stop' : 'Durdur'}
                >
                  <Square className="w-3 h-3" />
                </Button>
              </div>
            </div>
          )}

          {audioState === 'error' && (
            <div className="mt-4 text-xs text-red-600 dark:text-red-500 flex items-center gap-1.5">
              <VolumeX className="w-3 h-3" />
              <span>
                {isEn ? 'Audio playback failed' : 'Ses oynatılamadı'}
              </span>
              <button
                onClick={handlePlayAudio}
                className="underline hover:no-underline ml-1"
              >
                {isEn ? 'Retry' : 'Tekrar dene'}
              </button>
            </div>
          )}
        </div>

        {/* Metadata strip — ince bir çizgi, küçük rakamlar */}
        {(message.stats || message.provider) && (
          <div className="px-5 sm:px-6 py-2.5 bg-slate-100/70 dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-800 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
            <span className="font-mono">
              {message.provider} · {message.model}
            </span>
            {message.stats && (
              <>
                <span className="text-slate-300 dark:text-slate-700">·</span>
                <span>
                  {message.stats.total_retrieved}{' '}
                  {isEn ? 'sources' : 'kaynak'}
                </span>
                <span className="text-slate-300 dark:text-slate-700">·</span>
                <span>
                  {(message.stats.avg_similarity * 100).toFixed(0)}%{' '}
                  {isEn ? 'match' : 'eşleşme'}
                </span>
                <span className="text-slate-300 dark:text-slate-700">·</span>
                <span className="capitalize">
                  {isEn
                    ? message.stats.context_quality
                    : message.stats.context_quality === 'high'
                      ? 'yüksek'
                      : message.stats.context_quality === 'medium'
                        ? 'orta'
                        : 'düşük'}{' '}
                  {isEn ? 'quality' : 'kalite'}
                </span>
              </>
            )}
          </div>
        )}

        {/* Feedback strip — en altta, tek sıra */}
        <div className="px-5 sm:px-6 py-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between flex-wrap gap-2">
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {isEn ? 'Was this helpful?' : 'Yardımcı oldu mu?'}
          </span>
          <div className="flex items-center gap-1">
            {feedbackButtons.map(({ type, icon: Icon, label }) => {
              const selected = feedbackGiven === type;
              const disabled =
                feedbackGiven !== null &&
                feedbackGiven !== undefined &&
                feedbackGiven !== type;
              const loading = feedbackLoading === type;

              return (
                <button
                  key={type}
                  type="button"
                  onClick={() => handleFeedbackClick(type)}
                  disabled={disabled || loading}
                  aria-pressed={selected}
                  aria-label={label[isEn ? 'en' : 'tr']}
                  title={label[isEn ? 'en' : 'tr']}
                  className={`h-7 px-2 flex items-center gap-1 rounded-md text-xs transition-colors ${
                    selected
                      ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                      : disabled
                        ? 'text-slate-300 dark:text-slate-700 cursor-not-allowed'
                        : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  {loading ? (
                    <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Icon className="w-3 h-3" />
                  )}
                  <span className="hidden sm:inline">
                    {label[isEn ? 'en' : 'tr']}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* ─── Sources ───────────────────────────────────────────────
          Ayrı bir bölüm, collapse edilebilir liste. */}
      {message.sources && message.sources.length > 0 && (
        <details className="mt-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-layer-sm group">
          <summary className="px-4 py-3 flex items-center justify-between cursor-pointer list-none text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100">
            <div className="flex items-center gap-2">
              <ExternalLink className="w-3.5 h-3.5" aria-hidden="true" />
              <span>
                {message.sources.length}{' '}
                {isEn
                  ? message.sources.length === 1
                    ? 'source'
                    : 'sources'
                  : 'kaynak'}
              </span>
            </div>
            <ChevronDown
              className="w-4 h-4 transition-transform group-open:rotate-180"
              aria-hidden="true"
            />
          </summary>

          <div className="border-t border-slate-100 dark:border-slate-800 divide-y divide-slate-100 dark:divide-slate-800">
            {message.sources.map((source, index) => {
              const isExpanded = expandedSources.includes(index);
              const content =
                source.content_full || source.content_preview || '';
              const isLong = content.length > 200;
              const displayContent =
                isLong && !isExpanded ? content.slice(0, 200) + '…' : content;

              return (
                <div key={index} className="px-4 py-3">
                  <div className="flex items-start justify-between gap-3 mb-1.5">
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <span className="text-xs font-mono text-slate-400 dark:text-slate-500 shrink-0">
                        {source.airline === 'turkish_airlines'
                          ? 'TK'
                          : source.airline === 'pegasus'
                            ? 'PC'
                            : source.airline}
                      </span>
                      <span className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                        {source.source}
                      </span>
                    </div>
                    <span className="text-xs font-mono text-slate-500 dark:text-slate-400 shrink-0 tabular-nums">
                      {(source.similarity_score * 100).toFixed(0)}%
                    </span>
                  </div>

                  {content && (
                    <>
                      <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed whitespace-pre-wrap">
                        {displayContent}
                      </p>
                      {isLong && (
                        <button
                          type="button"
                          onClick={() => toggleSource(index)}
                          className="mt-1.5 text-xs text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 underline"
                        >
                          {isExpanded
                            ? isEn
                              ? 'Show less'
                              : 'Daha az göster'
                            : isEn
                              ? 'Show more'
                              : 'Devamını göster'}
                        </button>
                      )}
                    </>
                  )}

                  {(source.updated_date || source.url) && (
                    <div className="flex items-center gap-3 mt-2 text-xs text-slate-400 dark:text-slate-500">
                      {source.updated_date && (
                        <span>
                          {new Intl.DateTimeFormat(
                            isEn ? 'en-US' : 'tr-TR',
                          ).format(new Date(source.updated_date))}
                        </span>
                      )}
                      {source.url && (
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 hover:text-slate-900 dark:hover:text-slate-100 underline"
                        >
                          <ExternalLink className="w-3 h-3" aria-hidden="true" />
                          {isEn ? 'View source' : 'Kaynağa git'}
                        </a>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </details>
      )}
    </article>
  );
};