CELLS__ = [
("md", r"""# Retrieval-Augmented Generation (RAG) — Amazon Power-Tool Customer Reviews

**Topic:** What Amazon customers actually report about the performance, durability and
failure modes of corded and cordless **power tools and outdoor power equipment**
(drills/drivers, saws, sanders, routers, grinders, air compressors, nailers, jigs, blades
and tool batteries).

| Item | Value |
|---|---|
| **Topic** | Customer-reported performance & failure patterns of Amazon power tools (Tools & Home Improvement corpus) |
| **Source** | Julian McAuley (UCSD), *Amazon Review Data* (2014) — `Tools & Home Improvement` 5-core, SNAP mirror (`snap.stanford.edu`), review text + product metadata |
| **Licence / terms** | Non-commercial research & educational use only (McAuley Lab / SNAP terms of use). No redistribution of the raw archives; only a derived 45-document subset is used here. |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, 256 word-piece context |
| **LLM** | Google Gemini (`gemini-3.5-flash`, free tier) via `google-genai`, with automatic model-id failover; retrieval and evaluation still run without a key |
| **Vector database** | **ChromaDB** (open-source, local, persistent on disk) with cosine distance |
| **Collection size** | 45 TXT documents / ~184,000 words (requirement: 30–100 documents) |
| **Secrets** | No API key in code — read from `GOOGLE_API_KEY` env var or `google.colab.userdata` |

### How to run this notebook

1. **Runtime:** Google Colab or Jupyter. `Kernel → Restart & Run All`, top to bottom.
2. **Secrets (LLM only):** in Colab use the 🔑 *Secrets* panel to add `GOOGLE_API_KEY`;
   locally run `export GOOGLE_API_KEY=...` before starting Jupyter. Embeddings and the
   vector database need **no** key at all.
3. **First run** downloads two source archives (~126 MB) into `data/_source_cache/`,
   builds `data/raw/` + `data/clean/`, chunks, embeds and writes the persistent Chroma
   database to `chroma_db/`. Expect ~4–6 minutes.
4. **Later runs** reuse `data/raw`, `data/clean` and `chroma_db/`: the corpus builder,
   chunker and embedder are skipped automatically, and retrieval runs in well under a
   second. Deleting `chroma_db/` forces a rebuild.

### What each stage does

| Stage | Mechanism |
|---|---|
| Collect | Streamed gunzip + line-delimited JSON parse of the McAuley release; three passes over the archives (asin histogram → metadata join → review harvest) so only ~1 MB of the 126 MB is ever held in memory. |
| Clean | Unicode NFKC, HTML entity unescape, de-hyphenation, whitespace collapse, per-review sentence dedup, and corpus-level near-duplicate review detection with 5-gram Jaccard similarity. |
| Chunk | Review-aware sliding window: the document is split into `--- REVIEW n ---` blocks, each block into sentences, then sentences are packed into windows with character overlap. Every chunk keeps `filename`, `chunk_id`, `page_number`, `char_start`, `char_end`, `review_index`, `rating`, `brand`. |
| Embed | `all-MiniLM-L6-v2` is a 6-layer BERT (384 hidden units) with **mean pooling** over word-piece tokens and L2 normalisation — semantically similar sentences land at a small cosine angle. |
| Store | Chroma persists an HNSW index (cosine space) + payload metadata to `chroma_db/`, so `PersistentClient(path=...)` reopens the same vectors without re-embedding. |
| Generate | Top-k chunks are embedded with the *same* model, so query and document live in one vector space; cosine ranking returns the passages handed to the LLM. |"""),
("md", r"""## Setup — dependencies, configuration and secrets

Everything below is standard open-source Python. The only credential is the LLM key, and
it is read from the environment / Colab secret store — never hard-coded."""),
("code", r'''import importlib, subprocess, sys

REQUIRED = {
    "sentence_transformers": "sentence-transformers",
    "chromadb":             "chromadb",
    "pandas":               "pandas",
    "ipywidgets":           "ipywidgets",
    "google.genai":         "google-genai",
}
for module, package in REQUIRED.items():
    try:
        importlib.import_module(module)
        print(f"[ok]   {package}")
    except ImportError:
        print(f"[install] {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)

import os, json, time, gzip, re, ast, html, hashlib, shutil, collections, unicodedata, logging
from pathlib import Path
import pandas as pd
from IPython.display import display, HTML

def find_project_dir():
    """Jupyter starts the kernel in the server folder, not in the notebook folder, so the
    project directory is located from its own artefacts rather than from the working dir."""
    candidates = []
    if os.environ.get("RAG_PROJECT_DIR"):
        candidates.append(Path(os.environ["RAG_PROJECT_DIR"]))
    cwd = Path.cwd()
    candidates += [cwd, cwd/"machine_learning_RAG", *sorted(p for p in cwd.glob("*") if p.is_dir())]
    for c in candidates:
        if (c/"data"/"raw").is_dir() or (c/"chroma_db").is_dir() or c.name == "machine_learning_RAG":
            return c
    return cwd

PROJECT = find_project_dir()
DATA, RAW, CLEAN, CACHE = PROJECT/"data", PROJECT/"data/raw", PROJECT/"data/clean", PROJECT/"data/_source_cache"
CHROMA_DIR, ARTIFACTS = PROJECT/"chroma_db", PROJECT/"artifacts"
for d in (RAW, CLEAN, CACHE, ARTIFACTS):
    d.mkdir(parents=True, exist_ok=True)
print(f"project directory  : {PROJECT}")
print(f"kernel working dir : {Path.cwd()}")

def resolve_secret(name):
    """Read a secret from the environment, from a local key file, or from Colab's secret store.

    The key file keeps the key out of the notebook and out of shell history, which matters
    because this notebook is submitted. Create it with:
        echo -n 'your-key' > ~/.rag_api_key
    """
    if os.environ.get(name):
        return os.environ[name].strip()
    for candidate in (Path.home()/".rag_api_key", PROJECT/".rag_api_key"):
        try:
            if candidate.is_file():
                value = candidate.read_text().strip()
                if value:
                    return value
        except OSError:
            pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None

GOOGLE_API_KEY = resolve_secret("GOOGLE_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = os.environ.get("LLM_MODEL", "gemini-3.5-flash")

LLM_FALLBACKS   = ["gemini-flash-latest", "gemini-3.8-flash", "gemini-3.1-flash-lite-preview",
                   "gemini-3.5-flash"]
GEN_TEMPERATURE = 0.1
GEN_MAX_TOKENS  = 1500
TOP_K           = 6
RELEVANCE_FLOOR = 0.30
DISCLAIMER = ("Research/educational prototype built on public Amazon review data. "
              "It summarises customer opinion, not verified facts or engineering "
              "specifications — do not use it to make a purchase or safety decision.")

print("\nGOOGLE_API_KEY found:", bool(GOOGLE_API_KEY),
      "| LLM:", f"gemini:{LLM_MODEL}" if GOOGLE_API_KEY
      else "none yet - retrieval works, generation will report a config error")
print(f"embedding={EMBEDDING_MODEL}")
print(f"llm={LLM_MODEL} | temp={GEN_TEMPERATURE} | top_k={TOP_K} | floor={RELEVANCE_FLOOR}")
'''),
("md", r"""# Task 0 — Topic & Source Selection

**One-sentence topic.** *This system retrieves and answers questions about the performance,
durability and failure modes that Amazon customers report for corded and cordless power
tools (drills, saws, sanders, routers, grinders, air compressors, nailers, jigs, blades) and
tool batteries, using the McAuley Lab *Amazon Review Data (2014)* "Tools & Home
Improvement" corpus.*

**Source.** Julian McAuley (UC San Diego) released the *Amazon Review Data (2014)*
collection under a non-commercial research/educational licence. We use one main source and
two files from it: `reviews_Tools_and_Home_Improvement_5.json.gz` (134,476 reviews with
full text, 10,217 products) and `meta_Tools_and_Home_Improvement.json.gz` (titles, brands,
prices, category paths). The 5-core subset means every remaining user *and* product has
at least 5 reviews, so short-lived spam listings are already filtered out by the authors.

**Why a narrow topic?** A whole 134k-review dump makes retrieval meaningless — every query
is "relevant" to something. We first restrict to products whose Amazon category path sits
under *Power & Hand Tools → Power Tools* (plus power-tool accessories from tool brands),
then keep the **45 most-reviewed** of them. That yields a coherent corpus in which a
question like *"how long does the battery last on a DEWALT drill?"* has a real answer.

**Intended users — who uses this and why.** Consumer-product researchers, marketplace
sellers and product managers who need fast, citable answers about *how customers actually
experience* these tools (runtime, breakage, dust collection, blade quality) without
down-reading 45 review threads; and QA engineers who want a small, transparent RAG
reference system to debug retrieval and prompt design against."""),
("md", r"""### Questions this system **should** answer (in scope)

1. *How long does a battery charge take, and how much runtime do customers report on the
   DEWALT 18V drill/driver kit?*
2. *What is the most common complaint about cordless tool batteries across the reviews?*
3. *Do customers report that the Bosch 12-amp plunge router overheats or stalls under
   load?*

### Question this system **should not** be able to answer (out of scope → must abstain)

*What is the current price of a DeWalt DC970K-2 drill on Amazon in Kenya, and what is
Kenya's import duty on power tools?*

The corpus stops in 2014, contains no pricing feeds (only 2014 list prices), no
non-US geography and no customs data — so the system must decline rather than guess."""),
("md", r"""# Task 1 — Document Collection & Logging

Collection strategy (three streaming passes, so we never load 126 MB into memory):

1. **Pass 1** streams the reviews archive and builds a histogram of reviews per ASIN.
2. **Pass 2** streams the metadata archive, keeping only rows for the 1,500
   most-reviewed ASINs *and* only those whose category path is a power tool
   (or a power-tool accessory from a tool brand).
3. **Pass 3** re-streams the reviews and harvests the 20 most-helpful reviews that are
   ≥ 250 characters long for each of the top 45 products.

One TXT document = **one product's review digest** (header block + 20 individual reviews).
A document is the unit a human would want: "everything Amazon customers said about *this*
tool"."""),
("code", r'''import urllib.request, urllib.error

SNAP = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
SOURCES = {
    "reviews_Tools_and_Home_Improvement_5.json.gz": f"{SNAP}/reviews_Tools_and_Home_Improvement_5.json.gz",
    "meta_Tools_and_Home_Improvement.json.gz":      f"{SNAP}/meta_Tools_and_Home_Improvement.json.gz",
}
COLLECTION_DATE = "2026-09-30"

def download(url, dest, chunk=1 << 20):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cached] {dest.name}  ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"[get]    {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, t0 = 0, time.time()
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf); got += len(buf)
            pct = f"{100*got/total:5.1f}%" if total else "  ?  "
            print(f"\r         {pct}  {got/1e6:7.1f} MB  {time.time()-t0:5.1f}s", end="")
    print()
    tmp.replace(dest)
    return dest

SOURCE_FILES = {name: download(url, CACHE/name) for name, url in SOURCES.items()}
'''),
("code", r'''REV_FILE, META_FILE = (SOURCE_FILES["reviews_Tools_and_Home_Improvement_5.json.gz"],
                       SOURCE_FILES["meta_Tools_and_Home_Improvement.json.gz"])

N_DOCS, REVIEWS_PER_DOC, MIN_REVIEW_CHARS, TOP_ASIN_POOL = 45, 20, 250, 1500

POWER_TOOL_LEAVES = ("power tools", "drills", "saws", "table saws", "circular saws",
    "reciprocating saws", "jig saws", "miter saws", "routers", "sanders",
    "random-orbit sanders", "orbital sanders", "grinders", "air compressors", "planers",
    "nailers & staplers", "staplers", "impact wrenches")

TOOL_BRANDS = ("dewalt", "makita", "bosch", "milwaukee", "ryobi", "craftsman",
    "porter-cable", "hitachi", "rockwell", "metabo", "festool", "ridgid", "dremel",
    "black+decker", "black & decker", "kreg", "victor", "forrest", "fein", "jet")

def load(line):
    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

def category_path(meta):
    cats = meta.get("categories") or []
    if isinstance(cats, str):
        try: cats = ast.literal_eval(cats)
        except Exception: return ""
    return " > ".join(cats[0]) if cats else ""

def is_power_tool(meta, path):
    if any(leaf in path for leaf in POWER_TOOL_LEAVES):
        return True
    if "battery packs & chargers" in path or "power tool accessories" in path:
        return any(b in (meta.get("title") or "").lower() for b in TOOL_BRANDS)
    return False

t_build = time.time()
print("pass 1/3 — counting reviews per product ...")
review_counts = collections.Counter()
with gzip.open(REV_FILE, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        r = load(line)
        if r:
            review_counts[r["asin"]] += 1
print(f"   {sum(review_counts.values()):,} reviews across {len(review_counts):,} products")

pool = {a for a, _ in review_counts.most_common(TOP_ASIN_POOL)}
print("pass 2/3 — joining product metadata for the topic categories ...")
meta_by_asin = {}
with gzip.open(META_FILE, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        m = load(line)
        if m and m.get("asin") in pool:
            path = category_path(m).lower()
            if is_power_tool(m, path):
                meta_by_asin[m["asin"]] = m
print(f"   {len(meta_by_asin)} of {len(pool):,} in-scope products found")

selected = sorted(meta_by_asin, key=lambda a: -review_counts[a])[:N_DOCS]
selected_set = set(selected)
print("pass 3/3 — harvesting reviews ...")
reviews_by_asin = collections.defaultdict(list)
with gzip.open(REV_FILE, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        r = load(line)
        if r and r["asin"] in selected_set and len((r.get("reviewText") or "").strip()) >= MIN_REVIEW_CHARS:
            reviews_by_asin[r["asin"]].append(r)

for f in RAW.glob("*.txt"):
    f.unlink()
SOURCES_KEY = SOURCES["reviews_Tools_and_Home_Improvement_5.json.gz"]
collection_log, skipped = [], 0
for asin in selected:
    meta = meta_by_asin[asin]
    revs = sorted(reviews_by_asin[asin],
                  key=lambda r: (-r["helpful"][1], -r["unixReviewTime"]))[:REVIEWS_PER_DOC]
    if len(revs) < 5:
        skipped += 1
        continue
    title = re.sub(r"\s+", " ", meta.get("title") or "").strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:58].strip("_")
    avg = sum(float(r["overall"]) for r in revs) / len(revs)
    lines = [
        "AMAZON CUSTOMER REVIEW DIGEST", "=" * 70,
        f"Product      : {title}", f"Brand        : {meta.get('brand') or 'Unknown'}",
        f"ASIN         : {asin}", f"Price (USD)  : {meta.get('price')}",
        f"Category     : {category_path(meta)}",
        f"Reviews here : {len(revs)} (of {review_counts[asin]} in the source category file)",
        f"Avg rating   : {avg:.2f} / 5.0",
        "Source       : Julian McAuley, 'Amazon Review Data' (2014), Tools & Home Improvement 5-core",
        f"Product URL  : https://www.amazon.com/dp/{asin}",
        "Licence      : non-commercial research/educational use only (McAuley Lab / SNAP terms)",
        "=" * 70, "",
    ]
    for i, r in enumerate(revs, 1):
        lines += [f"--- REVIEW {i} ---",
                  f"Rating: {r['overall']}/5.0 | Date: {r.get('reviewTime','')} | Helpful votes: {r['helpful'][1]}",
                  "Heading: " + re.sub(r"\s+", " ", (r.get("summary") or "").strip()),
                  re.sub(r"\s+", " ", r["reviewText"]).strip(), ""]
    text = "\n".join(lines).strip() + "\n"
    fname = f"{slug}_{asin}.txt"
    (RAW/fname).write_text(text, encoding="utf-8")
    collection_log.append({
        "filename": fname, "source_url": SOURCES_KEY, "date_accessed": COLLECTION_DATE,
        "type": "TXT", "size": f"{(RAW/fname).stat().st_size/1024:.1f} KB",
        "licence_or_terms": "McAuley Lab Amazon Review Data (2014) - non-commercial research use",
        "asin": asin, "product": title, "brand": meta.get("brand") or "Unknown",
        "n_reviews": len(revs), "avg_rating": round(avg, 2),
    })

pd.DataFrame(collection_log).to_csv(ARTIFACTS/"collection_log.csv", index=False)
print(f"\nbuilt {len(collection_log)} documents in {time.time()-t_build:.1f}s "
      f"({skipped} skipped for insufficient long reviews)")
'''),
("code", r'''files = sorted(RAW.glob("*.txt"))
stats = {
    "Total documents": len(files),
    "Total words": sum(len(f.read_text(encoding="utf-8").split()) for f in files),
    "Total characters": sum(f.stat().st_size for f in files),
    "PDF files": len(list(RAW.glob("*.pdf"))),
    "TXT files": len(list(RAW.glob("*.txt"))),
    "Reviews per document (mean)": round(sum(r["n_reviews"] for r in collection_log)/len(collection_log), 1),
    "Mean product rating in corpus": round(sum(r["avg_rating"] for r in collection_log)/len(collection_log), 2),
    "Distinct brands": len({r["brand"] for r in collection_log}),
}
print("=" * 46); print("DATASET STATISTICS"); print("=" * 46)
for k, v in stats.items():
    print(f"{k:.<34}{v:>12,}" if isinstance(v, int) else f"{k:.<34}{v:>12}")

PAGE_CHARS = 2000
logical_pages = sum(max(1, -(-f.stat().st_size // PAGE_CHARS)) for f in files)
print(f"{'Logical pages (2000 chars/page)':.<34}{logical_pages:>12,}")
print(f"{'File type breakdown':.<34}{'TXT %d (100%%), PDF %d' % (stats['TXT files'], stats['PDF files']):>12}")
'''),
("code", r'''REQUIRED_COLUMNS = ["filename", "source_url", "date_accessed", "type", "size", "licence_or_terms"]
log_df = pd.DataFrame(collection_log)
print(f"Collection log — {len(log_df)} documents.\n")
print("--- the six required columns (filename truncated for width; full values in the CSV) ---")
view = log_df[REQUIRED_COLUMNS].copy()
view["filename"] = view["filename"].str.slice(0, 44) + "…"
view["source_url"] = view["source_url"].str.replace(
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/", "…/categoryFiles/")
view["licence_or_terms"] = "McAuley Lab 2014 review data — non-commercial research use"
print(view.head(12).to_string(index=False))
print(f"  ... {len(view) - 12} more rows\n")
print("--- one full row, unabridged ---")
for k, v in log_df[REQUIRED_COLUMNS].iloc[0].items():
    print(f"  {k:<18}: {v}")
print(f"\n  extra provenance columns kept: brand, product, asin, n_reviews, avg_rating")
print(f"  written to: {ARTIFACTS/'collection_log.csv'}")
'''),
("md", r"""### Licence, terms of use and compliance

* **Source.** Julian McAuley, *Amazon Review Data (2014)*, distributed via the SNAP
  repository (`snap.stanford.edu/data/amazon/productGraph/categoryFiles/`) and described
  on `cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html`.
* **Terms.** Free for **non-commercial research and educational use**; the authors ask that
  citations reference the dataset paper and that the raw archives are not redistributed.
  The 5-core subsets and the category files are the versions the authors publish for
  coursework-sized projects.
* **Compliance in this notebook.** (1) We download the archives at run time from the
  authors' own host rather than redistributing them — the two `.json.gz` files live in
  `data/_source_cache/` and are *not* part of the derived corpus. (2) We keep only a
  45-document derived subset (≈1 MB) in `data/raw/`, and every document carries its
  source URL, ASIN, access date and the licence line in its header. (3) The corpus is used
  for a coursework/educational RAG system only — no commercial use. (4) Every document
  contains review *text* only; reviewer IDs, reviewer names and profile links present in
  the raw archive are dropped during ingestion (see Task 1b), so no reviewer identity is
  stored."""),
("code", r'''for f in [files[0], files[len(files)//2], files[-1]]:
    text = f.read_text(encoding="utf-8")
    print("=" * 78); print(f"{f.name}  ({len(text.split()):,} words)"); print("=" * 78)
    print(text[:500].rstrip()); print()
'''),
]
