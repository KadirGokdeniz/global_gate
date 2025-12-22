import psycopg2
import logging
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------
# DB CONFIG
# -------------------------------------------------
DB_CONFIG = {
    "host": "db",
    "port": 5432,
    "dbname": "global_gate",
    "user": "postgres",
    "password": "postgres"
}

# -------------------------------------------------
# NOISE FILTERS (FINAL)
# -------------------------------------------------

BASE_STOPWORDS = {
    'the', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with',
    'is', 'are', 'be', 'as', 'by', 'from', 'that', 'this'
}

PRONOUN_NOISE = {
    'we', 'our', 'your', 'their', 'them', 'us', 'you'
}

LOGICAL_WORDS = {
    'not', 'if', 'than', 'more', 'can', 'cannot', 'only'
}

UI_NOISE = {
    'clicking', 'file', 'included', 'consists',
    'charged', 'charge', 'applied', 'accepted'
}

LOCATION_NOISE = {
    'istanbul', 'manila', 'cyprus', 'north',
    'turkiye', 'türkiye', 'world', 'international',
    'states', 'english', 'new'
}

UNITS = {
    'kg', 'cm', 'mm', 'wh', 'inch', 'inches'
}

CONDITIONAL_PHRASES = [
    'not exceed', 'cannot exceed',
    'more than', 'than one',
    'accepted if', 'if deemed',
    'not having', 'having appropriate',
    'if applicable', 'put ski'
]

# -------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_bad_phrase(phrase: str) -> bool:
    if any(p in phrase for p in CONDITIONAL_PHRASES):
        return True

    tokens = phrase.split()

    if any(t in BASE_STOPWORDS for t in tokens):
        return True

    if any(t in PRONOUN_NOISE for t in tokens):
        return True

    if any(t in LOGICAL_WORDS for t in tokens):
        return True

    if any(t in UI_NOISE for t in tokens):
        return True

    if any(loc in phrase for loc in LOCATION_NOISE):
        return True

    if any(u == phrase or phrase.endswith(u) for u in UNITS):
        return True

    if 'thedangerous' in phrase:
        return True

    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return True

    return False


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    logging.info("Veritabanina baglandi: %s:%s", DB_CONFIG["host"], DB_CONFIG["port"])

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    logging.info("KATEGORI OPTIMIZASYONU BASLIYOR")

    cur.execute("""
        SELECT category, content
        FROM policies
        WHERE content IS NOT NULL
    """)

    rows = cur.fetchall()

    category_docs = defaultdict(list)
    for category, content in rows:
        category_docs[category].append(normalize_text(content))

    logging.info("%d kategori bulundu, toplam %d policy",
                 len(category_docs), len(rows))

    for k, v in category_docs.items():
        logging.info("  - %s: %d belge", k, len(v))

    categories = list(category_docs.keys())
    documents = [" ".join(category_docs[c]) for c in categories]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9
    )

    tfidf = vectorizer.fit_transform(documents)
    feature_names = np.array(vectorizer.get_feature_names_out())

    logging.info("TF-IDF matrisi: %s", tfidf.shape)

    category_keywords = {}

    for idx, category in enumerate(categories):
        row = tfidf[idx].toarray().flatten()

        # global mean penalty
        row = row - tfidf.mean(axis=0).A1
        row[row < 0] = 0

        top_idx = row.argsort()[::-1]

        keywords = []
        for i in top_idx:
            term = feature_names[i]
            if is_bad_phrase(term):
                continue
            keywords.append(term)
            if len(keywords) == 9:
                break

        category_keywords[category] = keywords
        logging.info("%s: %s", category, keywords)

    # -------------------------------------------------
    # SAVE TO DB
    # -------------------------------------------------
    cur.execute("TRUNCATE category_representatives")

    for category, keywords in category_keywords.items():
        cur.execute("""
            INSERT INTO category_representatives (category, keywords)
            VALUES (%s, %s)
        """, (category, " ".join(keywords)))

    conn.commit()

    logging.info("Kategori temsilcileri DB'ye kaydedildi")
    logging.info("OPTIMIZASYON TAMAMLANDI")

    cur.close()
    conn.close()
    logging.info("Veritabani baglantisi kapatildi")

    # -------------------------------------------------
    # PRINT RESULT
    # -------------------------------------------------
    print("\nSONUC:\n")
    for category, keywords in category_keywords.items():
        print(f"{category}:")
        print(" ", " ".join(keywords))
        print()


# -------------------------------------------------
if __name__ == "__main__":
    main()
