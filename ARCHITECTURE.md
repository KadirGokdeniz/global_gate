# Architecture & Design Decisions

This document explains *why* the system is built the way it is, not *what* it does. For setup and usage see [README.md](README.md).

The project has moved past MVP: it runs on production infrastructure (Railway), processes real scraped data, and is actively deployed. It is not yet a commercial product, but the architectural decisions reflect production constraints rather than prototype shortcuts.

---

## High-level philosophy

Three principles drove most decisions:

1. **Graceful degradation over hard failures.** Every external dependency (reranker, TTS, LLM provider) has a fallback path. If one service is down, the system produces a lower-quality answer instead of an error.
2. **Cost awareness.** Open-source models are preferred where quality is comparable; paid APIs are used only where they clearly outperform alternatives for Turkish.
3. **Swappable components.** Provider-specific code is isolated behind thin wrappers so individual pieces (LLM, TTS, embedding model) can be replaced without touching the pipeline.

---

## Retrieval stack

### Vector store — pgvector over Pinecone / Weaviate / Qdrant

Policy-scale datasets (a few thousand documents, ~768-dim embeddings) don't justify a dedicated vector database. Running pgvector inside the existing Postgres instance means one service to monitor, one backup story, one connection pool. The same row can carry its embedding, full-text index, scalar filters, and metadata — which matters because retrieval uses all four.

The main trade-off is ceiling: pgvector with `ivfflat` struggles past roughly 1M rows without careful tuning. At current scale this isn't close to being a concern.

### Index type — ivfflat, post-load creation

`ivfflat` is training-based: the index builds K-means clusters from existing rows at creation time. Creating it on an empty table produces meaningless clusters, and new inserts never re-cluster themselves. The index is therefore created by the scraper **after** all embeddings are written, with `lists` auto-tuned from row count (`sqrt(n)` for small tables, `n/1000` for large ones, minimum 10). See `scraper_only.py::create_vector_index`.

`hnsw` was considered and is the more modern choice (training-free, generally better recall). Kept `ivfflat` because the post-load creation pattern is already documented and working, and the memory overhead of `hnsw` isn't worth the migration at current scale. This is a candidate for a later change.

### Embedding model — `Alibaba-NLP/gte-multilingual-base`

768 dimensions, open-source, strong multilingual (including Turkish) support. Three things mattered:

- **Cost.** Running locally means no per-query embedding cost. At ~2000 policies embedded once and cached, and ~N queries per day each requiring one embedding, an API-based embedding model (OpenAI `text-embedding-3-small` etc.) would add unnecessary ongoing cost with no quality advantage for this corpus.
- **Turkish quality.** Alibaba GTE is explicitly trained on multilingual data; empirically it handles Turkish airline policy text well. English-only models (`bge-large-en`, `all-mpnet-base-v2`) were non-starters.
- **Dimension pragmatism.** 768 is small enough for fast pgvector queries, large enough to preserve semantic nuance.

`bge-m3` and `multilingual-e5-large` are plausible alternatives; they offer stronger benchmarks in some settings. Haven't done a rigorous head-to-head evaluation yet, which is a known gap.

### Reranking — Cohere `rerank-v3.5`

Two-stage retrieval (bi-encoder → cross-encoder rerank) is now standard practice in RAG. The bi-encoder gets ~20 candidates cheaply; the reranker picks the top 5 with much better precision because it encodes query and document *jointly*.

Chose Cohere over alternatives because:

- **Multilingual quality.** `rerank-v3.5` supports 100+ languages natively, including Turkish. Monolingual rerankers (`bge-reranker-base`) would have required running both EN and TR variants.
- **Stability.** v3.5 has been the production-default multilingual model for ~a year. `rerank-v4.0-pro` is newer and potentially higher quality but less battle-tested; `rerank-v4.0-fast` trades too much quality for speed given our 5-document top-k.
- **Latency.** Typical call time is ~300-500ms for 20 documents — acceptable within the overall <3s answer budget.

**Graceful degradation.** If Cohere is unreachable or returns an error, `CohereRerankerService.rerank()` falls back to the original bi-encoder ordering truncated to `top_n`. This is intentional: reranker failures should cause quality degradation, not outages.

---

## LLM layer

### Dual-provider setup — Claude + OpenAI

Both providers are supported behind the same `BaseLLMService` interface (`llm_base.py`). Two reasons:

- **Vendor lock-in avoidance.** Provider outages and pricing changes are frequent enough that being able to flip providers with a config change has real operational value.
- **Quality comparison.** For the same query and context, two providers can give noticeably different answers. Having both makes prompt iteration measurable (is the new prompt better on *both* providers, or just one?).

The cost is maintenance: both providers follow the same prompt versioning scheme and CoT format. The `BaseLLMService` abstraction keeps provider-specific code to ~80 lines per wrapper; shared logic (prompt assembly, CoT parsing, response shaping, error envelope) lives once in `llm_base.py`.

### Default models

- **Claude**: `claude-haiku-4-5-20251001`. Fast, cheap, sufficient for structured policy Q&A. Upgrade to Sonnet/Opus is one config change.
- **OpenAI**: `gpt-4o-mini`. Similar rationale. GPT-3.5 was used initially; swapped because 4o-mini is both cheaper and meaningfully better.

Temperature is `0.2` across both — low enough for consistent citation behavior, not fully deterministic so format variance doesn't compound.

### Prompt engineering

Prompts are versioned (`PROMPT_VERSION` constant in `llm_base.py`, currently `v4`). Each version is tracked in a comment:

- **v1**: Baseline.
- **v2**: Few-shot examples added (good vs bad outputs shown to the model).
- **v3**: Citation format spec tightened — `(Source: category_name)` strict.
- **v4**: Tone rule added — forbids bureaucratic openings like "In the provided documents...", which users found off-putting.

Each prompt enforces four rules explicitly in the system instruction:

1. No fabrication — answer only from the provided documents.
2. Cite every specific claim with `(Source: category_name)`.
3. No marketing language or brand fluff.
4. Surface missing information honestly rather than hiding gaps.

The prompt version propagates through the response envelope (`prompt_version` field), which makes it possible to run A/B-style comparisons if/when an evaluation harness is added.

### Chain of Thought — toggleable, structured markers

CoT is exposed as a per-request `use_cot` flag. When on, the model writes:

```
[REASONING]
- Question:
- Relevant documents:
- Key facts with sources:
- Missing information:

[ANSWER]
(user-facing text)
```

The parser (`parse_cot_response` in `llm_base.py`) splits these two sections. It has a chain of regex fallbacks for cases where the model deviates from the format. This is fragile and a known weak spot — Anthropic's XML tags or OpenAI's `response_format=json_object` would be more robust, and migrating is on the radar.

---

## Voice pipeline

### STT — AssemblyAI

Picked for Turkish accuracy. Whisper was the obvious alternative but Turkish performance is noticeably worse on informal/accented speech, which is what users actually produce when dictating. AssemblyAI also handles punctuation and formatting out of the box, reducing post-processing.

### TTS — ElevenLabs `eleven_turbo_v2_5`

Two things mattered: latency (turbo family is sub-second on short text) and multilingual prosody (better than open alternatives for Turkish). Voice settings are currently a starting baseline (`stability: 0.5, similarity_boost: 0.75`), not tuned.

Voice IDs are the same across TR/EN, which is a compromise — a TR-native voice would improve perceived quality but adds management overhead. Worth revisiting if user feedback highlights it.

### Decoupled from the main pipeline

STT and TTS run as separate streams. If TTS fails (rate limit, API issue), the query response is still delivered; the user just doesn't get audio. STT failures fall back to keyboard input. This keeps the voice layer an enhancement rather than a dependency.

---

## Frontend architecture

### Stack — React + Vite + TailwindCSS

Vite dev loop is fast; Tailwind keeps the design system tight without a CSS framework runtime. TypeScript is used throughout for shape safety on API boundaries (see `src/types/index.ts`).

### State management

No Redux/Zustand — `useState` + custom hooks (`useLanguage`, `usePersistedMessages`, `useVoiceRecording`) cover the entire app. Two reasons:

- **Scope.** The chat-style UI has limited cross-cutting state; hooks are enough.
- **Persistence is scoped by airline.** `usePersistedMessages` keeps separate message lists per airline in `localStorage`, with schema versioning (`SCHEMA_VERSION` constant) so breaking changes can invalidate old data cleanly.

### Error handling — nested error boundaries

Two `ErrorBoundary` instances in `App.tsx`: one wraps the entire app (providers + router), one wraps the route content. A crash inside a page doesn't take down the toast system or the router; the inner boundary offers "Try Again" which remounts just the page.

Both boundaries funnel caught errors into Sentry via `onError`, so errors visible to users are also visible to the dev.

### Voice recording — refs to work around stale closures

`useVoiceRecording` uses `isRecordingRef` and `stopRecordingRef` rather than relying on `useState` values inside the `requestAnimationFrame` loop. React state updates don't propagate into the RAF closure, so a ref is the only reliable way to let the loop check "should I still be running?" on every frame. This is explicitly documented in the hook's comments; it's a subtle but necessary pattern for real-time media loops in React.

---

## Infrastructure

### Dev vs. Prod compose files

Two `docker-compose` files with deliberately different shapes:

- **`docker-compose.yml` (dev)**: full stack — db, scraper, api, frontend, Prometheus, Grafana. Secrets are **file-based** (`./secrets/*`), mounted via Docker Secrets.
- **`docker-compose-prod.yml` (prod)**: just `api` and `frontend`. Scraper and monitoring are omitted (scraper runs on-demand; monitoring is handled by the hosting platform). Secrets are **environment-variable-based** because Railway / Render / Fly.io don't support Docker Secrets.

The two compose files share Dockerfiles but diverge on operational concerns — this is intentional, not duplication.

### Secrets — `SecretsLoader` abstraction

The `SecretsLoader` utility (`src/api/core/secrets_loader.py`) looks for credentials in three places, in order:

1. `/run/secrets/<name>` — Docker Secrets mount (dev)
2. Environment variable pointed to by `<NAME>_FILE` — alternative file-based pattern
3. Environment variable `<NAME>` — direct env var (prod platforms)

This keeps application code identical across environments. Adding a new secret means adding a line in `docker-compose.yml` and calling `loader.get_secret('my_key', 'MY_KEY_ENV')` in code.

### Observability

- **Prometheus + Grafana** (dev): metrics scraped from `/metrics` on the FastAPI container via `prometheus-fastapi-instrumentator`. Dashboards provisioned from `monitoring/grafana/dashboards/`.
- **Sentry** (both): backend errors via the FastAPI integration, frontend errors via `@sentry/react`. Configured with `sendDefaultPii: false` and a header filter that redacts `Authorization` and `api-key` headers before events are sent.
- **Health checks**: every docker-compose service has a `healthcheck` block; the `api` service's `start_period: 1200s` is generous because the embedding model download on first boot can legitimately take that long.
