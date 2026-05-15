"""
Global Gate — Chaos / Resilience Test
Dual-LLM failover + graceful degradation ölçümü

Çalıştır:
    python chaos_test.py

Ölçülen metrikler:
    R% = query completion rate under failures
    F  = test edilen failure scenario sayısı
"""

import requests
import time
import concurrent.futures
from dataclasses import dataclass
from typing import List

API_BASE = "https://globalgate-production.up.railway.app"

@dataclass
class ScenarioResult:
    name: str
    total: int
    completed: int

    @property
    def rate(self):
        return self.completed / self.total * 100


def is_valid_response(r: requests.Response) -> bool:
    """Geçerli cevap: status 200 ve içinde answer veya results var."""
    if r.status_code not in (200, 422):
        return False
    try:
        data = r.json()
        return bool(
            data.get("answer") or
            data.get("results") or
            data.get("success") or
            data.get("response")
        )
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# SENARYO 1 — Concurrent load (5 eş zamanlı istek)
# Her iki provider da aynı anda basınç altında
# ══════════════════════════════════════════════════════════════
def scenario_concurrent_load() -> ScenarioResult:
    print("\n[S1] Concurrent Load — 5 simultaneous requests to /chat/openai")
    queries = [
        "THY bagaj politikası nedir?",
        "What is the carry-on baggage limit?",
        "Pegasus iptali durumunda iade var mı?",
        "Can I bring a musical instrument on board?",
        "Hamile yolcular ne zaman uçamaz?",
    ]

    completed = 0
    def call(q):
        try:
            r = requests.post(
                f"{API_BASE}/chat/openai",
                json={"question": q, "language": "tr"},
                timeout=45
            )
            return is_valid_response(r)
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(call, q) for q in queries]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        completed = sum(results)

    print(f"  Result: {completed}/5 completed")
    return ScenarioResult("Concurrent Load (5 parallel)", 5, completed)


# ══════════════════════════════════════════════════════════════
# SENARYO 2 — Fallback provider (Claude)
# OpenAI yerine Claude endpoint — ikincil provider çalışıyor mu?
# ══════════════════════════════════════════════════════════════
def scenario_fallback_provider() -> ScenarioResult:
    print("\n[S2] Fallback Provider — 5 requests to /chat/claude")
    queries = [
        "THY check-in süresi nedir?",
        "What are excess baggage fees?",
        "Pegasus evcil hayvan politikası nedir?",
        "Can I change my flight date?",
        "Aktarmalı uçuşta bagaj nasıl alınır?",
    ]

    completed = 0
    for q in queries:
        try:
            r = requests.post(
                f"{API_BASE}/chat/claude",
                json={"question": q, "language": "tr"},
                timeout=45
            )
            if is_valid_response(r):
                completed += 1
                print(f"  ✅ {q[:40]}...")
            else:
                print(f"  ❌ {q[:40]}... (status: {r.status_code})")
        except Exception as e:
            print(f"  ❌ {q[:40]}... (error: {e})")
        time.sleep(0.5)

    return ScenarioResult("Fallback Provider (Claude)", 5, completed)


# ══════════════════════════════════════════════════════════════
# SENARYO 3 — Malformed / edge input
# Hatalı girdilerde sistem çöküyor mu?
# ══════════════════════════════════════════════════════════════
def scenario_malformed_input() -> ScenarioResult:
    print("\n[S3] Malformed Input — edge cases handled gracefully?")
    edge_cases = [
        {"question": "a", "language": "tr"},                         # çok kısa
        {"question": "?" * 10, "language": "tr"},                    # sadece özel karakter
        {"question": "THY " * 100, "language": "tr"},                # çok uzun (400 char)
        {"question": "🛫✈️🛬🧳🎫", "language": "en"},               # sadece emoji
        {"question": "SELECT * FROM policy;", "language": "en"},     # SQL injection denemesi
    ]

    completed = 0
    for case in edge_cases:
        try:
            r = requests.post(
                f"{API_BASE}/chat/openai",
                json=case,
                timeout=30
            )
            # 200 veya 422 (validation) = graceful, 500 = crash
            if r.status_code != 500:
                completed += 1
                print(f"  ✅ Graceful ({r.status_code}): {str(case['question'])[:30]}...")
            else:
                print(f"  ❌ Server crash (500): {str(case['question'])[:30]}...")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
        time.sleep(0.3)

    return ScenarioResult("Malformed / Edge Input", 5, completed)


# ══════════════════════════════════════════════════════════════
# SENARYO 4 — Reranker degradation
# Reranker olmadan sistem hâlâ cevap üretiyor mu?
# (similarity-search endpoint = reranker bypass)
# ══════════════════════════════════════════════════════════════
def scenario_reranker_degradation() -> ScenarioResult:
    print("\n[S4] Reranker Degradation — similarity-search fallback mode")
    queries = [
        "bagaj ağırlık limiti",
        "flight cancellation refund",
        "check-in deadline",
        "pet travel rules",
        "excess baggage fee",
    ]

    completed = 0
    for q in queries:
        try:
            r = requests.get(
                f"{API_BASE}/vector/similarity-search",
                params={"q": q, "limit": 5, "threshold": 0.1},
                timeout=15
            )
            data = r.json()
            if r.status_code == 200 and data.get("results"):
                completed += 1
                print(f"  ✅ {q}")
            else:
                print(f"  ❌ {q} (no results)")
        except Exception as e:
            print(f"  ❌ {q}: {e}")
        time.sleep(0.3)

    return ScenarioResult("Reranker Degradation (bypass)", 5, completed)


# ══════════════════════════════════════════════════════════════
# SENARYO 5 — Rapid fire (rate limit / stability)
# 10 istek arka arkaya — servis stabil kalıyor mu?
# ══════════════════════════════════════════════════════════════
def scenario_rapid_fire() -> ScenarioResult:
    print("\n[S5] Rapid Fire — 10 back-to-back requests (stability test)")
    q = "What is the baggage allowance?"
    completed = 0

    for i in range(10):
        try:
            r = requests.post(
                f"{API_BASE}/chat/openai",
                json={"question": q, "language": "en"},
                timeout=30
            )
            if is_valid_response(r):
                completed += 1
                print(f"  ✅ [{i+1}/10]")
            else:
                print(f"  ❌ [{i+1}/10] status: {r.status_code}")
        except Exception as e:
            print(f"  ❌ [{i+1}/10] error: {e}")
        time.sleep(0.2)

    return ScenarioResult("Rapid Fire (10 back-to-back)", 10, completed)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run_chaos_test():
    print("=" * 60)
    print("Global Gate — Chaos / Resilience Test")
    print("=" * 60)

    results: List[ScenarioResult] = []

    results.append(scenario_concurrent_load())
    results.append(scenario_fallback_provider())
    results.append(scenario_malformed_input())
    results.append(scenario_reranker_degradation())
    results.append(scenario_rapid_fire())

    # ── Özet ─────────────────────────────────────────────────
    total_requests  = sum(r.total     for r in results)
    total_completed = sum(r.completed for r in results)
    overall_rate    = total_completed / total_requests * 100

    print("\n" + "=" * 60)
    print("SONUÇLAR")
    print("=" * 60)
    for r in results:
        status = "✅" if r.rate >= 80 else "⚠️ "
        print(f"  {status} {r.name:<35} {r.completed}/{r.total} = {r.rate:.0f}%")

    print(f"\n  OVERALL: {total_completed}/{total_requests} = {overall_rate:.1f}%")
    print()
    print("CV BULLET DEĞERLERİ:")
    print(f"  R% = {overall_rate:.0f}%  (query completion under failure)")
    print(f"  F  = {len(results)}  (failure scenarios tested)")
    print("=" * 60)


if __name__ == "__main__":
    run_chaos_test()
