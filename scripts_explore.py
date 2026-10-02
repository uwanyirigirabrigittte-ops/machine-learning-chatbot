"""Exploration script: understand the Amazon Tools & Home Improvement 5-core files."""
import gzip, json, re, collections, itertools, os

CACHE = "/home/student/machine_learning_RAG/data/_source_cache"
REV = os.path.join(CACHE, "reviews_Tools_and_Home_Improvement_5.json.gz")
META = os.path.join(CACHE, "meta_Tools_and_Home_Improvement.json.gz")

print("=== first 2 review records ===")
with gzip.open(REV, "rt", encoding="utf-8") as f:
    for line in itertools.islice(f, 2):
        r = json.loads(line)
        print({k: (str(v)[:200]) for k, v in r.items()})
        print("keys:", list(r.keys()))

print("\n=== review count per asin (top 15) ===")
counts = collections.Counter()
n_reviews = 0
with gzip.open(REV, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        counts[r["asin"]] += 1
        n_reviews += 1
print("total reviews:", n_reviews, "unique asins:", len(counts))
print(counts.most_common(15))

top_asins = {a for a, _ in counts.most_common(3000)}
print("\n=== metadata scan for titles of top-3000 asins ===")
found = {}
import ast
def load(line):
    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

with gzip.open(META, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = load(line)
        if m and m.get("asin") in top_asins:
            found[m["asin"]] = m
print("metadata rows matched:", len(found))
m = next(iter(found.values()))
print("meta keys:", list(m.keys()))
print("sample:", {k: str(v)[:150] for k, v in m.items()})

KEYWORDS = ["drill", "saw", "grinder", "sander", "jigsaw", "router", "nail gun",
            "impact wrench", "pressure washer", "lawn mower", "hedge trimmer",
            "chainsaw", "leaf blower", "weed", "shop vac", "planer", "staple",
            "battery", "charger", "power tool", "miter", "lathe", "compressor"]
hits = []
for asin, m in found.items():
    t = (m.get("title") or "").lower()
    if any(k in t for k in KEYWORDS):
        hits.append((counts[asin], asin, m.get("title", "")[:90], m.get("brand", "")[:25],
                     m.get("price"), (m.get("categories") or [[]])[0]))
hits.sort(reverse=True)
print("\n=== keyword-matching products (top 50 of %d) ===" % len(hits))
for h in hits[:50]:
    print(h)
