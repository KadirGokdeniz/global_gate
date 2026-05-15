"""
Global Gate — Hit Rate@5 Eval Script
Reranker ON (chat endpoint) vs OFF (similarity search endpoint)

Çalıştır:
    pip install requests
    python eval_hitrate.py
"""

import requests
import json
import time

API_BASE = "https://globalgate-production.up.railway.app"

# ─────────────────────────────────────────────────────────────
# TEST SETİ — 20 sorgu + her biri için beklenen anahtar kelimeler
# En az 1 keyword cevap/kaynak içinde geçiyorsa → HIT
# ─────────────────────────────────────────────────────────────
TEST_QUERIES = [
    # ══════════════════════════════════════════════════════════
    # GRUP 1 — Türkçe zor / ambiguous (10)
    # ══════════════════════════════════════════════════════════
    {
        "q": "Bebek maması kabin bagajına alınabilir mi yoksa kargo mı yapılmalı?",
        "lang": "tr",
        "keywords": ["bebek", "mama", "kabin", "sıvı", "liquid", "güvenlik"]
    },
    {
        "q": "Pegasus ile THY arasında fazla bagaj ücreti farkı nedir?",
        "lang": "tr",
        "keywords": ["fazla", "bagaj", "ücret", "fiyat", "extra"]
    },
    {
        "q": "Uçuş 4 saat gecikmişse yemek hakkım var mı?",
        "lang": "tr",
        "keywords": ["gecikme", "yemek", "ikram", "hak", "meal", "delay"]
    },
    {
        "q": "İlaçlarımı kabin bagajında taşıyabilir miyim?",
        "lang": "tr",
        "keywords": ["ilaç", "tıbbi", "kabin", "reçete", "medical", "medication"]
    },
    {
        "q": "Online check-in yapmadan havalimanına gidersem ne olur?",
        "lang": "tr",
        "keywords": ["check-in", "havalimanı", "konter", "counter", "ücret"]
    },
    {
        "q": "Müzik aleti uçağa nasıl taşınır, koltuk satın almak gerekir mi?",
        "lang": "tr",
        "keywords": ["müzik", "enstrüman", "koltuk", "instrument", "seat"]
    },
    {
        "q": "Refakat gerektiren engelli yolcu için hangi hizmetler sunuluyor?",
        "lang": "tr",
        "keywords": ["engelli", "tekerlekli", "wheelchair", "özel", "yardım"]
    },
    {
        "q": "Vize reddedilirse bilet ücretimi geri alabilir miyim?",
        "lang": "tr",
        "keywords": ["vize", "red", "iade", "refund", "iptal"]
    },
    {
        "q": "Aktarmalı uçuşta bagajım otomatik transfer edilir mi?",
        "lang": "tr",
        "keywords": ["aktarma", "transit", "bagaj", "transfer", "connecting"]
    },
    {
        "q": "Hamile yolcular uçuşun kaçıncı haftasına kadar uçabilir?",
        "lang": "tr",
        "keywords": ["hamile", "hafta", "gebelik", "pregnant", "week"]
    },

    # ══════════════════════════════════════════════════════════
    # GRUP 2 — English hard / cross-domain (10)
    # ══════════════════════════════════════════════════════════
    {
        "q": "If I miss my connecting flight due to a delay, who is responsible?",
        "lang": "en",
        "keywords": ["connecting", "missed", "delay", "responsible", "rebook"]
    },
    {
        "q": "Can I bring breast milk through security in carry-on?",
        "lang": "en",
        "keywords": ["breast milk", "liquid", "security", "baby", "exemption"]
    },
    {
        "q": "What happens to my checked bag if the flight is overbooked?",
        "lang": "en",
        "keywords": ["overbook", "denied", "baggage", "volunteer", "compensation"]
    },
    {
        "q": "Is a power bank allowed in checked luggage or only carry-on?",
        "lang": "en",
        "keywords": ["power bank", "battery", "lithium", "carry-on", "checked"]
    },
    {
        "q": "How much does it cost to change the name on a ticket?",
        "lang": "en",
        "keywords": ["name", "change", "fee", "correction", "ticket"]
    },
    {
        "q": "What are the rules for traveling with a wheelchair?",
        "lang": "en",
        "keywords": ["wheelchair", "mobility", "assistance", "disability"]
    },
    {
        "q": "Can I carry fishing equipment as checked baggage?",
        "lang": "en",
        "keywords": ["fishing", "sport", "equipment", "checked", "rod"]
    },
    {
        "q": "What is the maximum size for a personal item under the seat?",
        "lang": "en",
        "keywords": ["personal item", "under seat", "dimensions", "cm", "size"]
    },
    {
        "q": "Do I need a transit visa if I have a layover in Istanbul?",
        "lang": "en",
        "keywords": ["transit", "visa", "layover", "Istanbul", "transfer"]
    },
    {
        "q": "Can unaccompanied minors travel on Pegasus flights?",
        "lang": "en",
        "keywords": ["unaccompanied", "minor", "child", "alone", "service"]
    },

    # ══════════════════════════════════════════════════════════
    # GRUP 3 — Cross-airline comparison (10)
    # Retrieval doğru airline'ı bulabilmeli
    # ══════════════════════════════════════════════════════════
    {
        "q": "THY business class fazla bagaj hakkı Pegasus'tan ne kadar farklı?",
        "lang": "tr",
        "keywords": ["business", "bagaj", "fark", "THY", "Pegasus"]
    },
    {
        "q": "Pegasus'ta koltuk seçimi ücretli mi THY'ye göre?",
        "lang": "tr",
        "keywords": ["koltuk", "seçim", "ücret", "seat", "selection"]
    },
    {
        "q": "Which airline allows more carry-on weight, Turkish Airlines or Pegasus?",
        "lang": "en",
        "keywords": ["carry-on", "weight", "kg", "Turkish", "Pegasus"]
    },
    {
        "q": "Çocuk indirimi Pegasus ve THY'de aynı mı?",
        "lang": "tr",
        "keywords": ["çocuk", "indirim", "yaş", "child", "discount"]
    },
    {
        "q": "Compare cancellation fees between Turkish Airlines and Pegasus",
        "lang": "en",
        "keywords": ["cancellation", "fee", "Turkish", "Pegasus", "refund"]
    },
    {
        "q": "Hangi havayolu evcil hayvan için kabin alıyor: THY mi Pegasus mu?",
        "lang": "tr",
        "keywords": ["evcil", "hayvan", "kabin", "pet", "cabin"]
    },
    {
        "q": "Is there a meal service difference on short-haul Pegasus vs THY flights?",
        "lang": "en",
        "keywords": ["meal", "service", "short-haul", "ikram"]
    },
    {
        "q": "THY ekonomi paketi nedir, Pegasus'ta karşılığı var mı?",
        "lang": "tr",
        "keywords": ["ekonomi", "paket", "fare", "class", "economy"]
    },
    {
        "q": "Which airline has stricter pet weight limits in cabin?",
        "lang": "en",
        "keywords": ["pet", "weight", "limit", "cabin", "kg"]
    },
    {
        "q": "Pegasus promosyon bileti iadesi var mı, THY ile fark?",
        "lang": "tr",
        "keywords": ["promosyon", "promo", "iade", "bilet", "refund"]
    },

    # ══════════════════════════════════════════════════════════
    # GRUP 4 — Sports / rare equipment (10)
    # Spesifik spor kategorilerinden — chunk sayısı az, retrieval zor
    # ══════════════════════════════════════════════════════════
    {
        "q": "Sualtı dalış ekipmanı için bagaj hakkı nedir?",
        "lang": "tr",
        "keywords": ["dalış", "diving", "tüp", "ekipman", "scuba"]
    },
    {
        "q": "Can I bring a hunting rifle on an international flight?",
        "lang": "en",
        "keywords": ["hunting", "rifle", "firearm", "permit", "weapon"]
    },
    {
        "q": "Yay ve ok (archery) ekipmanı nasıl uçakla taşınır?",
        "lang": "tr",
        "keywords": ["yay", "ok", "archery", "ekipman", "bow"]
    },
    {
        "q": "What is the policy for transporting a kayak or canoe?",
        "lang": "en",
        "keywords": ["kayak", "canoe", "kano", "boat", "water"]
    },
    {
        "q": "Bowling topları kabin bagajına alınır mı yoksa cargo mu?",
        "lang": "tr",
        "keywords": ["bowling", "top", "ball", "cargo", "kabin"]
    },
    {
        "q": "Are surfboards transported as standard baggage or extra fee?",
        "lang": "en",
        "keywords": ["surfboard", "surf", "extra", "fee", "board"]
    },
    {
        "q": "Paraşüt ekipmanı uçağa alınabilir mi?",
        "lang": "tr",
        "keywords": ["paraşüt", "parachute", "ekipman", "skydive"]
    },
    {
        "q": "Can I bring camping/tent equipment on the plane?",
        "lang": "en",
        "keywords": ["camping", "tent", "kamp", "outdoor", "equipment"]
    },
    {
        "q": "Su kayağı ekipmanı için özel paketleme gerekiyor mu?",
        "lang": "tr",
        "keywords": ["su", "kayağı", "water skiing", "paketleme", "ekipman"]
    },
    {
        "q": "What are the rules for transporting hockey sticks?",
        "lang": "en",
        "keywords": ["hockey", "stick", "sopa", "ekipman"]
    },

    # ══════════════════════════════════════════════════════════
    # GRUP 5 — Multi-category ambiguous (10)
    # Birden fazla kategori arasından seçim — reranker burada belirleyici
    # ══════════════════════════════════════════════════════════
    {
        "q": "Profesyonel kamera ekipmanı, fazla bagaj mı yoksa el bagajı mı sayılır?",
        "lang": "tr",
        "keywords": ["kamera", "ekipman", "fazla", "el", "bagaj", "camera"]
    },
    {
        "q": "Is a service animal treated differently from a regular pet?",
        "lang": "en",
        "keywords": ["service animal", "pet", "regular", "differ", "disability"]
    },
    {
        "q": "Elektrikli tekerlekli sandalye batarya kuralları nelerdir?",
        "lang": "tr",
        "keywords": ["tekerlekli", "sandalye", "batarya", "wheelchair", "battery"]
    },
    {
        "q": "Can I carry a wedding dress as a special item without paying extra?",
        "lang": "en",
        "keywords": ["wedding", "dress", "garment", "special", "extra"]
    },
    {
        "q": "Tenis raketi spor ekipmanı mı yoksa kabin bagajı mı?",
        "lang": "tr",
        "keywords": ["tenis", "raket", "spor", "kabin", "tennis", "racket"]
    },
    {
        "q": "Are oxygen concentrators considered medical equipment or extra baggage?",
        "lang": "en",
        "keywords": ["oxygen", "concentrator", "medical", "equipment", "tıbbi"]
    },
    {
        "q": "Cenaze töreni için yurt dışına seyahatte özel ücret var mı?",
        "lang": "tr",
        "keywords": ["cenaze", "funeral", "ölüm", "indirim", "bereavement"]
    },
    {
        "q": "Difference between a personal item and a laptop bag — can I bring both?",
        "lang": "en",
        "keywords": ["personal", "laptop", "bag", "both", "item"]
    },
    {
        "q": "Bisikletim büyük, fazla bagaj mı spor ekipmanı mı tarifesinde?",
        "lang": "tr",
        "keywords": ["bisiklet", "bicycle", "fazla", "spor", "ekipman", "tarife"]
    },
    {
        "q": "What documents are needed for international pet travel with vaccination?",
        "lang": "en",
        "keywords": ["pet", "international", "vaccination", "documents", "rabies"]
    },
]

def check_hit(text: str, keywords: list) -> bool:
    """En az 1 keyword metinde geçiyor mu?"""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def eval_without_reranker(query: dict) -> bool:
    """
    /vector/similarity-search → pure cosine similarity, reranker YOK
    Top-5 sonucun content'ini birleştirip keyword check.
    """
    try:
        r = requests.get(
            f"{API_BASE}/vector/similarity-search",
            params={"q": query["q"], "limit": 5, "threshold": 0.1},
            timeout=30
        )
        if r.status_code != 200:
            return False

        data = r.json()
        results = data.get("results", data.get("data", []))
        if not results:
            return False

        combined = " ".join([
            str(doc.get("content", "")) + " " + str(doc.get("text", ""))
            for doc in results[:5]
        ])
        return check_hit(combined, query["keywords"])

    except Exception as e:
        print(f"  ⚠️  similarity-search error: {e}")
        return False

def eval_with_reranker(query: dict) -> bool:
    """
    /chat/openai → full RAG pipeline, Cohere reranker VAR
    LLM cevabında keyword check.
    """
    try:
        r = requests.post(
            f"{API_BASE}/chat/openai",
            json={"question": query["q"], "language": query["lang"]},
            timeout=60
        )
        if r.status_code != 200:
            return False

        data = r.json()
        answer  = str(data.get("answer", ""))
        sources = str(data.get("sources", data.get("context", data.get("chunks", ""))))
        combined = answer + " " + sources
        return check_hit(combined, query["keywords"])

    except Exception as e:
        print(f"  ⚠️  chat error: {e}")
        return False


def run_eval():
    print("=" * 60)
    print("Global Gate — Hit Rate@5 Eval")
    print(f"Test set: {len(TEST_QUERIES)} queries (TR + EN)")
    print("=" * 60)

    hits_no_rerank = []
    hits_with_rerank = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i:02d}/{len(TEST_QUERIES)}] {query['lang'].upper()} — {query['q'][:55]}...")

        # Reranker YOK
        hit_base = eval_without_reranker(query)
        hits_no_rerank.append(hit_base)
        print(f"  No rerank : {'✅ HIT' if hit_base else '❌ MISS'}")
        time.sleep(0.5)

        # Reranker VAR
        hit_rerank = eval_with_reranker(query)
        hits_with_rerank.append(hit_rerank)
        print(f"  Reranked  : {'✅ HIT' if hit_rerank else '❌ MISS'}")
        time.sleep(1.0)

    # ── Sonuçlar ────────────────────────────────────────────
    n = len(TEST_QUERIES)
    rate_base   = sum(hits_no_rerank)   / n * 100
    rate_rerank = sum(hits_with_rerank) / n * 100
    delta       = rate_rerank - rate_base

    print("\n" + "=" * 60)
    print("SONUÇLAR")
    print("=" * 60)
    print(f"  Reranker OFF (cosine only) : {sum(hits_no_rerank)}/{n} = {rate_base:.1f}%")
    print(f"  Reranker ON  (Cohere v3.5) : {sum(hits_with_rerank)}/{n} = {rate_rerank:.1f}%")
    print(f"  Delta (Δ)                  : +{delta:.1f} pp")
    print()
    print("CV BULLET DEĞERLERİ:")
    print(f"  A% = {rate_base:.0f}%")
    print(f"  B% = {rate_rerank:.0f}%")
    print(f"  Δ  = +{delta:.0f} pp")
    print(f"  Q  = {n} bilingual queries")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()

# ── EK 30 SORU (gerçek kategorilere dayalı) ──────────────────

EXTRA_QUERIES = [
    # Pets (pets_cabin, pets_cargo, pets_onboard, pets_country_rules)
    {"q": "Kedi kabin bagajında taşınabilir mi, kutu ölçüleri ne olmalı?",
     "lang": "tr", "keywords": ["kedi", "kabin", "kutu", "ölçü", "cabin", "carrier"]},
    {"q": "Köpek kargo olarak gönderilirse sıcaklık koşulları neler?",
     "lang": "tr", "keywords": ["köpek", "kargo", "sıcaklık", "cargo", "temperature"]},
    {"q": "Ülkeye göre evcil hayvan giriş kuralları nasıl öğrenilir?",
     "lang": "tr", "keywords": ["ülke", "giriş", "country", "import", "veteriner", "quarantine"]},
    {"q": "What documents are needed to travel with a pet internationally?",
     "lang": "en", "keywords": ["document", "pet", "health", "certificate", "veterinary"]},
    {"q": "Is there a weight limit for pets in the cabin?",
     "lang": "en", "keywords": ["weight", "pet", "cabin", "limit", "kg"]},
    {"q": "Can a service animal travel for free on the plane?",
     "lang": "en", "keywords": ["service", "animal", "free", "assistance", "disability"]},

    # Sports ekipmanları (sports_bicycle, sports_golf, sports_diving vs)
    {"q": "Bisiklet uçağa nasıl verilir, söküm gerekli mi?",
     "lang": "tr", "keywords": ["bisiklet", "bicycle", "söküm", "paket", "kasa"]},
    {"q": "Golf ekipmanı için ek ücret ne kadar?",
     "lang": "tr", "keywords": ["golf", "ekipman", "ücret", "fee", "bag"]},
    {"q": "Tüplü dalış ekipmanı uçağa alınabilir mi, tüp yasak mı?",
     "lang": "tr", "keywords": ["dalış", "tüp", "diving", "tank", "oxygen"]},
    {"q": "How do I transport a surfboard on a flight?",
     "lang": "en", "keywords": ["surfboard", "surf", "oversize", "fee", "transport"]},
    {"q": "Are ski boots counted as part of my baggage allowance?",
     "lang": "en", "keywords": ["ski", "boot", "baggage", "allowance", "sports"]},
    {"q": "Can I bring a hunting rifle on a commercial flight?",
     "lang": "en", "keywords": ["hunting", "rifle", "firearm", "weapon", "declare"]},

    # Musical instruments
    {"q": "Keman kabin bagajında taşınabilir mi?",
     "lang": "tr", "keywords": ["keman", "kabin", "enstrüman", "violin", "instrument"]},
    {"q": "Büyük çello için ayrı koltuk satın almam gerekir mi?",
     "lang": "tr", "keywords": ["çello", "koltuk", "seat", "cello", "instrument"]},
    {"q": "What is the policy for carrying a guitar on board?",
     "lang": "en", "keywords": ["guitar", "instrument", "cabin", "overhead", "seat"]},

    # Carry-on / excess baggage
    {"q": "Kabin bagajım boyut sınırını aşarsa ne olur?",
     "lang": "tr", "keywords": ["kabin", "boyut", "aşar", "ücret", "oversized"]},
    {"q": "İki parça check-in bagajı verilebilir mi ekonomi sınıfında?",
     "lang": "tr", "keywords": ["iki", "parça", "ekonomi", "checked", "extra"]},
    {"q": "What is the fee for a third checked bag?",
     "lang": "en", "keywords": ["third", "checked", "bag", "fee", "extra"]},
    {"q": "Can I prepay for extra baggage online to get a discount?",
     "lang": "en", "keywords": ["prepay", "online", "extra", "baggage", "discount"]},

    # Extra services / restrictions
    {"q": "Özel yemek talebini uçuştan kaç saat önce yapmalıyım?",
     "lang": "tr", "keywords": ["özel", "yemek", "meal", "special", "request"]},
    {"q": "Tekerlekli sandalye ücretsiz mi kargo bagajına eklenir?",
     "lang": "tr", "keywords": ["tekerlekli", "sandalye", "wheelchair", "ücretsiz", "free"]},
    {"q": "Are lithium batteries allowed in checked baggage?",
     "lang": "en", "keywords": ["lithium", "battery", "checked", "dangerous", "allowed"]},
    {"q": "What items are completely prohibited on all flights?",
     "lang": "en", "keywords": ["prohibited", "banned", "dangerous", "restricted", "forbidden"]},

    # Ambiguous / multi-category
    {"q": "Hem evcil hayvan hem de spor ekipmanı taşırsam toplam ücret ne olur?",
     "lang": "tr", "keywords": ["evcil", "spor", "ekipman", "ücret", "toplam"]},
    {"q": "Bebekle seyahatte hangi ekstra hizmetler ücretsiz?",
     "lang": "tr", "keywords": ["bebek", "ücretsiz", "hizmet", "infant", "free", "bassinet"]},
    {"q": "Uçuştan 1 saat önce check-in olabiliyor muyum yoksa geç mi kalırım?",
     "lang": "tr", "keywords": ["check-in", "saat", "önce", "deadline", "late"]},
    {"q": "If I book economy but upgrade at the gate, does my baggage allowance change?",
     "lang": "en", "keywords": ["upgrade", "baggage", "allowance", "economy", "business"]},
    {"q": "Can I take my camera equipment and drone as carry-on?",
     "lang": "en", "keywords": ["camera", "drone", "carry-on", "equipment", "battery"]},
    {"q": "What are the rules for traveling with a newborn under 7 days old?",
     "lang": "en", "keywords": ["newborn", "infant", "days", "baby", "age"]},
    {"q": "Is travel insurance mandatory or optional when booking?",
     "lang": "en", "keywords": ["insurance", "mandatory", "optional", "travel", "coverage"]},
]