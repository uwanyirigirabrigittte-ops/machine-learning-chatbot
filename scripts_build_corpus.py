"""Pilot corpus builder for the Amazon power-tools RAG dataset."""
import ast, collections, csv, gzip, json, os, re, sys, time

CACHE = "/home/student/machine_learning_RAG/data/_source_cache"
RAW = "/home/student/machine_learning_RAG/data/raw"
REV = os.path.join(CACHE, "reviews_Tools_and_Home_Improvement_5.json.gz")
META = os.path.join(CACHE, "meta_Tools_and_Home_Improvement.json.gz")

N_DOCS = 45
REVIEWS_PER_DOC = 20
MIN_REVIEW_CHARS = 250
TOP_ASIN_POOL = 1500

POWER_TOOL_LEAVES = (
    "power tools", "drills", "saws", "table saws", "circular saws", "reciprocating saws",
    "jig saws", "miter saws", "routers", "sanders", "random-orbit sanders", "orbital sanders",
    "grinders", "air compressors", "planers", "nailers & staplers", "staplers",
    "impact wrenches",
)
# Bare consumer batteries/chargers also sit under "Battery Packs & Chargers", so we keep
# only the tool-brand ones (a DEWALT 18V pack is a power-tool accessory; an Energizer
# D-cell is not part of this topic).
TOOL_BRANDS = ("dewalt", "makita", "bosch", "milwaukee", "ryobi", "craftsman", "porter-cable",
               "hitachi", "rockwell", "metabo", "festool", "ridgid", "snap-on", "dremel",
               "black+decker", "black & decker", "kreg", "victor", "forrest", "fein", "jet")


def is_power_tool(m, path):
    if any(leaf in path for leaf in POWER_TOOL_LEAVES):
        return True
    if "battery packs & chargers" in path or "power tool accessories" in path:
        title = (m.get("title") or "").lower()
        return any(b in title for b in TOOL_BRANDS)
    return False


def load(line):
    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None


def cat_path(m):
    cats = m.get("categories") or []
    try:
        if isinstance(cats, str):
            cats = ast.literal_eval(cats)
        return " > ".join(cats[0])
    except Exception:
        return ""


t0 = time.time()
counts = collections.Counter()
with gzip.open(REV, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        r = load(line)
        if r:
            counts[r["asin"]] += 1
print("pass1 %.1fs" % (time.time() - t0), len(counts), counts.most_common(3))

pool = {a for a, _ in counts.most_common(TOP_ASIN_POOL)}
meta = {}
with gzip.open(META, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = load(line)
        if m and m.get("asin") in pool:
            path = cat_path(m).lower()
            if is_power_tool(m, path):
                meta[m["asin"]] = m
print("pass2 %.1fs" % (time.time() - t0), "matched", len(meta))

sel = sorted(meta, key=lambda a: -counts[a])[:N_DOCS]
print("selected", len(sel))

by_asin = collections.defaultdict(list)
with gzip.open(REV, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        r = load(line)
        if r and r["asin"] in set(sel):
            t = (r.get("reviewText") or "").strip()
            if len(t) >= MIN_REVIEW_CHARS:
                by_asin[r["asin"]].append(r)
print("pass3 %.1fs" % (time.time() - t0))

os.makedirs(RAW, exist_ok=True)
rows, tot_words, tot_chars = [], 0, 0
for asin in sel:
    m = meta[asin]
    rs = sorted(by_asin[asin], key=lambda r: (-r["helpful"][1], -r["unixReviewTime"]))[:REVIEWS_PER_DOC]
    if len(rs) < 5:
        continue
    title = re.sub(r"\s+", " ", m.get("title") or "").strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:60].strip("_")
    fn = f"{slug}_{asin}.txt"
    avg = sum(float(r["overall"]) for r in rs) / len(rs)
    parts = [
        "AMAZON CUSTOMER REVIEW DIGEST",
        "=" * 70,
        f"Product      : {title}",
        f"Brand        : {m.get('brand') or 'Unknown'}",
        f"ASIN         : {asin}",
        f"Price (USD)  : {m.get('price')}",
        f"Category     : {cat_path(m)}",
        f"Reviews here : {len(rs)} (of {counts[asin]} in the source category file)",
        f"Avg rating   : {avg:.2f} / 5.0",
        "Source       : Julian McAuley, 'Amazon Review Data' (2014), Tools & Home Improvement 5-core",
        f"Product URL  : https://www.amazon.com/dp/{asin}",
        "Licence      : non-commercial research/educational use only (McAuley Lab terms)",
        "=" * 70,
        "",
    ]
    for i, r in enumerate(rs, 1):
        parts += [
            f"--- REVIEW {i} ---",
            f"Rating: {r['overall']}/5.0 | Date: {r.get('reviewTime','')} | Helpful votes: {r['helpful'][1]}",
            f"Heading: {re.sub(chr(92)+'s+', ' ', (r.get('summary') or '').strip())}",
            re.sub(r"\s+", " ", r["reviewText"]).strip(),
            "",
        ]
    text = "\n".join(parts).strip() + "\n"
    with open(os.path.join(RAW, fn), "w", encoding="utf-8") as out:
        out.write(text)
    words = len(text.split())
    tot_words += words
    tot_chars += len(text)
    rows.append((fn, asin, len(rs), counts[asin], round(avg, 2), words, m.get("price"), cat_path(m)))

print("\nwrote %d docs | %d words | %d chars" % (len(rows), tot_words, tot_chars))
for r in sorted(rows, key=lambda x: -x[5])[:12]:
    print(r[0][:70], "| revs", r[2], "| words", r[5], "| avg", r[4])
print("\nsmallest:")
for r in sorted(rows, key=lambda x: x[5])[:5]:
    print(r[0][:70], "| words", r[5])
