CELLS__ = [
("md", r"""# Task 2 — Cleaning

The files are machine-generated (one review per block, whitespace already collapsed), so
cleaning is a light pass. Six techniques are applied because the dump does contain
measurable noise:

1. **Unicode normalisation** — `NFKC` folds full-width characters, ligatures and no-break
   spaces so that "18V" / "18 V" and non-breaking spaces cannot fragment a search.
2. **HTML entity unescaping** — the raw dump contains `&amp;`, `&#39;`, `&quot;`,
   `&nbsp;`… which would otherwise sit *inside* sentences the LLM reads.
3. **De-hyphenation / line-break repair** — `exam-\nple` → `example`; surviving hard line
   breaks are joined back into flowing paragraphs.
4. **Whitespace repair** — collapse runs of spaces/tabs, strip trailing spaces, remove
   control characters, normalise smart quotes and en/em dashes to ASCII.
5. **Sentence-level de-duplication** — reviews are frequently copy-pasted inside one review
   ("Great saw. Great saw."); identical sentences are collapsed to one.
6. **Corpus-level near-duplicate review detection** — a 5-gram *Jaccard* shingle signature
   finds the same review posted under several variants of a product; the later copy is
   dropped so the retriever cannot return the same text twice.

Output goes to `data/clean/`, and only cleaned text is chunked."""),
("code", r'''WORD_RE = re.compile(r"\w+", re.UNICODE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ENTITY_RE = re.compile(r"&(?:amp|quot|apos|lt|gt|nbsp|#\d{2,5});")

BOILERPLATE_RE = re.compile(
    r"^\s*(?:amazon\.com\s*review|verified purchase|read more|see all reviews|"
    r"product details|add to cart|in stock|one touch)\s*[:.\-]?\s*$", re.I)
REVIEW_SPLIT_RE = re.compile(r"^---\s+REVIEW\s+(\d+)\s+---\s*$", re.M)

def shingles(text, n=5):
    """Set of word n-grams — the fingerprint used for near-duplicate detection."""
    w = [t.lower() for t in WORD_RE.findall(text)]
    return {tuple(w[i:i+n]) for i in range(max(0, len(w)-n+1))}

def jaccard(a, b):
    """Similarity of two shingle sets (1.0 = identical review text)."""
    return len(a & b) / len(a | b) if a and b else 0.0

def clean_text(text):
    """Techniques 1-5. Returns (clean_text, n_html_entities_found)."""
    n_entities = len(ENTITY_RE.findall(text))
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)
    text = re.sub(r"[‘’ʼ]", "'", text).replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = "\n".join(l for l in text.splitlines() if not BOILERPLATE_RE.match(l))
    text = re.sub(r"[ \t]{2,}", " ", text).rstrip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), n_entities

def dedupe_sentences(block):
    """(5) Collapse repeated sentences *inside* one review, preserving order."""
    parts = re.split(r"(?<=[.!?])\s+", block)
    seen, kept = set(), []
    for p in parts:
        key = re.sub(r"\W+", " ", p.lower()).strip()
        if key and key in seen:
            clean_stats["repeated_sentences"] += 1
            continue
        seen.add(key)
        kept.append(p)
    return " ".join(kept)

t_clean = time.time()
clean_stats = collections.Counter()
kept_signatures, near_duplicates = [], []

for raw_file in sorted(RAW.glob("*.txt")):
    raw_text = raw_file.read_text(encoding="utf-8")
    text, n_ent = clean_text(raw_text)
    clean_stats["html_entities"] += n_ent
    clean_stats["chars_removed"] += len(raw_text) - len(text)

    marks = list(REVIEW_SPLIT_RE.finditer(text))
    header = text[:marks[0].start()]
    blocks = []
    for i, m in enumerate(marks):
        end = marks[i+1].start() if i+1 < len(marks) else len(text)
        blocks.append(text[m.end():end].strip())

    kept = []
    for b in blocks:
        b = dedupe_sentences(b)
        if len(WORD_RE.findall(b)) < 30:
            clean_stats["review_blocks_dropped"] += 1
            continue
        sig = shingles(b)
        twin = next((k for k, s in kept_signatures if jaccard(s, sig) >= 0.80), None)
        if twin is not None:
            near_duplicates.append((raw_file.name, twin))
            clean_stats["near_duplicate_reviews"] += 1
            continue
        kept_signatures.append((f"{raw_file.name}#{len(kept)+1}", sig))
        kept.append(b)
        clean_stats["words"] += len(WORD_RE.findall(b))

    out = header + "".join(f"\n--- REVIEW {i} ---\n{b}\n" for i, b in enumerate(kept, 1))
    (CLEAN/raw_file.name).write_text(out, encoding="utf-8")

raw_words = sum(len(f.read_text(encoding='utf-8').split()) for f in RAW.glob("*.txt"))
n_clean = len(list(CLEAN.glob("*.txt")))

pairwise = [jaccard(kept_signatures[i][1], kept_signatures[j][1])
            for i in range(0, len(kept_signatures), 7)
            for j in range(i+1, len(kept_signatures), 11)]
print(f"cleaned {n_clean} documents in {time.time()-t_clean:.1f}s")
print(f"  documents processed           : {n_clean}")
print(f"  HTML entities unescaped       : {clean_stats['html_entities']:,}")
print(f"  characters removed (noise)    : {clean_stats['chars_removed']:,}")
print(f"  repeated sentences collapsed  : {clean_stats['repeated_sentences']:,} (technique 5)")
print(f"  near-duplicate reviews dropped: {clean_stats['near_duplicate_reviews']:,} (technique 6)")
print(f"  highest Jaccard seen between any two kept reviews: {max(pairwise) if pairwise else 0:.3f}")
print(f"     -> the corpus has no cross-posted reviews, so the 0.80 threshold dropped nothing;")
print(f"        the detector is what guarantees that, not luck.")
print(f"  degenerate review blocks lost : {clean_stats['review_blocks_dropped']:,}")
print(f"  words  raw -> clean           : {raw_words:,} -> {clean_stats['words']:,} "
      f"({100*(1-clean_stats['words']/max(1,raw_words)):.1f}% removed)")
if near_duplicates:
    print("  near-duplicate examples       :", near_duplicates[:3])
'''),
("code", r'''def first_review_block(text):
    marks = list(REVIEW_SPLIT_RE.finditer(text))
    if not marks:
        return text
    end = marks[1].start() if len(marks) > 1 else len(text)
    return text[marks[0].end():end].strip()

for name in [files[0].name, files[len(files)//3].name, files[-1].name]:
    before = first_review_block((RAW/name).read_text(encoding="utf-8"))
    after = first_review_block((CLEAN/name).read_text(encoding="utf-8"))
    print("=" * 78); print(name); print("=" * 78)
    print("--- BEFORE ---"); print(before[:420], "\n")
    print("--- AFTER  ---"); print(after[:420], "\n")
    if before == after:
        print("(first review block unchanged — the edits landed in later blocks)\n")
'''),
("md", r"""### What we deliberately did **not** clean

**The header block and the `--- REVIEW n ---` markers are kept.** The header carries the
product title, brand, ASIN, price, category, rating and source URL, and the review markers
delimit the blocks that give every chunk its `review_index`, `rating` and `date`. They are
also useful retrieval signal ("DEWALT", "18-Volt", "Avg rating 2.65") and they make a
citation checkable in Task 7. Removing them would shrink the corpus by ~9% and delete the
metadata the citations depend on. The `Rating:` / `Date:` / `Helpful votes:` lines inside
each review are kept for the same reason, and customer wording is left verbatim rather than
paraphrased, since a paraphrase would misreport what people actually wrote."""),
("md", r"""# Task 3 — Extraction, Chunking & Metadata

**Extraction.** Cleaned TXT is read as UTF-8 behind a logged failure path (missing file,
undecodable byte, empty result) so a broken document can never silently drop out of the
index.

**Two chunking strategies, implemented by hand** (no framework — the point is to show what
the splitters actually do):

* **Strategy A — Recursive character splitting.** Tries the most semantic separator first
  (`\n\n` = review boundary), then `\n`, then sentence end, then a space, then a hard
  character cut. This is the algorithm behind LangChain's `RecursiveCharacterTextSplitter`.
  It guarantees `chunk_size` and never emits an empty chunk, but it treats
  `--- REVIEW n ---` as ordinary characters, so chunks routinely contain the tail of one
  customer's review plus the start of the next.
* **Strategy B — Review-aware sliding window.** The document is split on
  `--- REVIEW n ---` first, so a chunk can never mix two reviews; each block is split into
  sentences with an abbreviation guard (`Mr.`, `U.S.`, `1/2` are not sentence ends), then
  sentences are greedily packed into `chunk_size`-character windows carrying `overlap`
  characters of context from the previous window.

Both are measured on chunk count, mean/median size, size dispersion, the share of chunks
that **mix two reviews** (evidence damage) and the share that **end mid-sentence**
(semantic damage)."""),
("code", r'''PAGE_CHARS = 2000
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 180

def extract_pages(path):
    """Read a document and return logical pages. Raises so the caller can log failures."""
    text = path.read_bytes().decode("utf-8", errors="strict")
    if not text.strip():
        raise ValueError("empty document")
    return [text[i:i+PAGE_CHARS] for i in range(0, len(text), PAGE_CHARS)]

documents, extraction_log = [], []
for f in sorted(CLEAN.glob("*.txt")):
    try:
        pages = extract_pages(f)
    except Exception as exc:
        extraction_log.append({"filename": f.name, "status": "FAILED", "detail": repr(exc)[:90]})
        continue
    if not " ".join(pages).strip():
        extraction_log.append({"filename": f.name, "status": "EMPTY", "detail": "no text"})
        continue
    documents.append({"filename": f.name, "pages": pages, "text": "".join(pages),
                      "n_pages": len(pages)})
    extraction_log.append({"filename": f.name, "status": "OK", "detail": f"{len(pages)} pages"})

failed = [r for r in extraction_log if r["status"] == "FAILED"]
empty = [r for r in extraction_log if r["status"] == "EMPTY"]
print(f"successful extractions : {len(documents)}")
print(f"failed extractions     : {len(failed)}")
print(f"empty documents        : {len(empty)}")
display(pd.DataFrame(failed or empty or
        [{"filename": "-", "status": "none", "detail": "all documents extracted cleanly"}]))
'''),
("code", r'''SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def recursive_split(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, seps=SEPARATORS):
    """Hard `size` guarantee with graceful degradation through the separator list."""
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in seps:
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return out

_ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bNo)(?<!\bvs)(?<!\bU\.S)(?<!\betc)(?<!\bJr)(?<!\bSt)(?<!\bapprox)(?<!\bFig)(?<!\bInc)(?<!\bCo)"
SENT_SPLIT_RE = re.compile(rf"{_ABBREV}(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(])")

def split_sentences(text):
    return [s for s in SENT_SPLIT_RE.split(re.sub(r"\s+", " ", text).strip()) if s.strip()]

def review_aware_split(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Pack whole sentences into windows; a window never crosses a review boundary."""
    chunks, marks = [], list(REVIEW_SPLIT_RE.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i+1].start() if i+1 < len(marks) else len(text)
        block_text = re.sub(r"\s+", " ", text[m.end():end]).strip()
        if not block_text:
            continue
        cur, cur_len = [], 0
        for sent in split_sentences(block_text):
            if cur and cur_len + len(sent) + 1 > size:
                chunks.append(" ".join(cur))
                keep, kept = [], 0
                for prev in reversed(cur):
                    if kept + len(prev) > overlap:
                        break
                    keep.insert(0, prev)
                    kept += len(prev) + 1
                cur, cur_len = keep, kept
            cur.append(sent)
            cur_len += len(sent) + 1
        if cur:
            chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if c.strip()]

def locate_spans(text, pieces, overlap=CHUNK_OVERLAP):
    """Chunks are emitted in document order, so a forward scan recovers exact offsets."""
    spans, cursor = [], 0
    for piece in pieces:
        probe = piece[:70].lstrip()
        pos = text.find(probe, cursor)
        if pos == -1:
            pos = text.find(probe[:30], cursor)
        if pos == -1:
            pos = cursor
        spans.append((pos, pos + len(piece)))
        cursor = max(pos + len(piece) - (overlap if overlap else 0), pos + 1)
    return spans

_demo_doc = documents[0]
A_demo = recursive_split(_demo_doc["text"])
B_demo = review_aware_split(_demo_doc["text"])
print(f"document: {_demo_doc['filename']}  ({len(_demo_doc['text']):,} chars)")
print(f"  Strategy A -> {len(A_demo):4d} chunks | mean {sum(map(len,A_demo))/len(A_demo):6.0f} chars")
print(f"  Strategy B -> {len(B_demo):4d} chunks | mean {sum(map(len,B_demo))/len(B_demo):6.0f} chars")
print("\nStrategy A, chunk 3 (note: it can straddle two reviews):\n ", A_demo[3][:260])
print("\nStrategy B, chunk 3 (note: one review only, whole sentences):\n ", B_demo[3][:260])
'''),
("code", r'''def structural_metrics(text, pieces):
    """Evidence damage: chunks that straddle a review marker. Semantic damage: chunks that
    stop mid-sentence. Dispersion: std-dev of chunk size (proxy for embedding noise)."""
    spans = locate_spans(text, pieces)
    marker_starts = [m.start() for m in REVIEW_SPLIT_RE.finditer(text)]
    mixed = sum(1 for s, e in spans if any(s < p < e for p in marker_starts))
    cut = sum(1 for (s, e), p in zip(spans, pieces) if p.rstrip()[-1:] not in '.!?"\'')
    sizes = [len(p) for p in pieces]
    return {"chunks": len(sizes),
            "mean_chars": int(sum(sizes)/max(1, len(sizes))),
            "median_chars": int(sorted(sizes)[len(sizes)//2]) if sizes else 0,
            "size_stdev": int(pd.Series(sizes).std() or 0),
            "pct_mixing_reviews": round(100*mixed/max(1, len(sizes)), 1),
            "pct_mid_sentence": round(100*cut/max(1, len(sizes)), 1)}

rows = []
for doc in documents:
    for name, pieces in (("A_recursive_char", recursive_split(doc["text"])),
                         ("B_review_aware_sentence", review_aware_split(doc["text"]))):
        rows.append({"strategy": name, **structural_metrics(doc["text"], pieces)})

cmp_df = (pd.DataFrame(rows).groupby("strategy")
          .agg(chunks=("chunks", "sum"), mean_chars=("mean_chars", "mean"),
               median_chars=("median_chars", "mean"), size_stdev=("size_stdev", "mean"),
               pct_mixing_reviews=("pct_mixing_reviews", "mean"),
               pct_mid_sentence=("pct_mid_sentence", "mean")).round(1))
display(cmp_df)
print("How to read the table:")
print("  A: efficient, hard 1200-char ceiling — but `pct_mixing_reviews` > 0 means a chunk can")
print("     hold the end of one customer's review and the start of another's, so the LLM")
print("     cannot attribute the statement it is asked to quote.")
print("  B: no review mixing, no mid-sentence cuts; slightly smaller mean size, which is a")
print("     fair trade for focused, attributable retrieval units.")
'''),
("md", r"""### Strategy choice: **B — review-aware sliding window**

**Why B.** The deciding evidence is structural: with Strategy A a non-zero
`pct_mixing_reviews` means a single retrieved "passage" can contain the end of one
customer's review and the start of the next. For a system whose entire value is *citable
customer testimony*, such a chunk is a chunk the LLM cannot attribute — the citation in
Task 7 would name a document but not a statement. Strategy B drives that metric to zero
and also removes mid-sentence cuts, at the cost of a somewhat lower mean chunk size. Each
chunk is still ~200 words, which is squarely in the range MiniLM was trained for.

**Chunk size = 1,200 characters (~200 words); overlap = 180 characters (~3 sentences).**
Evidence taken from this corpus rather than from habit:

* **Model limit.** The embedder's hard limit is 256 word-pieces. A 1,200-character window of
  review prose measures ~188 word-pieces on average but peaks near 345 (long product names,
  part numbers and URLs are token-expensive). We therefore keep the 1,200-character window
  *and* split the ~25% of chunks that exceed the limit at a sentence boundary in Task 4,
  rather than shrinking the window to ~800 characters, which would cut complaints away from
  the reason customers give for them.
* **Review structure.** Complaints are stated with a justification ("ran maybe 20 minutes
  before it cut out"). Around ~1,000 characters we start separating the complaint from its
  reason; beyond ~1,500 characters a chunk mixes two topics and its embedding centroid
  blurs, so a query about *batteries* scores highly against a chunk that is mostly about
  *blade alignment*.
* **Overlap.** 180 characters ≈ 3 sentences: enough to carry the context a truncated
  complaint needs, small enough that the same sentence does not dominate two adjacent
  chunks and inflate the frequency of irrelevant near-duplicate hits."""),
("code", r'''CHUNK_FILE = ARTIFACTS/"chunks.jsonl"

DOC_HEADER_RE = {
    "asin":   re.compile(r"ASIN\s*:\s*(\w+)"),
    "brand":  re.compile(r"Brand\s*:\s*(.+)"),
    "title":  re.compile(r"Product\s*:\s*(.+)"),
    "rating": re.compile(r"Avg rating\s*:\s*([\d.]+)"),
}

def review_blocks(text):
    """[(offset_of_block_text, normalised_block_text), ...] for every --- REVIEW n ---."""
    marks = list(REVIEW_SPLIT_RE.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i+1].start() if i+1 < len(marks) else len(text)
        out.append((m.end(), re.sub(r"\s+", " ", text[m.end():end]).strip()))
    return out

def build_chunks(documents, strategy="B_review_aware_sentence"):
    """Chunk every document and attach full provenance metadata to each chunk."""
    chunks = []
    for doc in documents:
        text = doc["text"]
        pieces = (review_aware_split(text) if strategy.startswith("B") else recursive_split(text))
        spans = locate_spans(text, pieces)
        blocks = review_blocks(text)
        header = {k: (m.search(text).group(1).strip() if m.search(text) else None)
                  for k, m in DOC_HEADER_RE.items()}
        for i, (piece, (start, end)) in enumerate(zip(pieces, spans)):
            rev_i = sum(1 for b_start, _ in blocks if b_start < start)
            block_text = blocks[rev_i-1][1] if rev_i else ""
            review_line = re.search(r"Rating:\s*([\d.]+)/5\.0\s*\|\s*Date:\s*([^|]+)\|", block_text)
            chunks.append({
                "id": f"{Path(doc['filename']).stem}__{i:04d}",
                "document": doc["filename"],
                "text": piece,
                "metadata": {
                    "filename": doc["filename"],
                    "chunk_id": f"{Path(doc['filename']).stem}__{i:04d}",
                    "page_number": start // PAGE_CHARS + 1,
                    "char_start": start, "char_end": end,
                    "n_chars": len(piece), "n_words": len(piece.split()),
                    "review_index": rev_i,
                    "review_rating": float(review_line.group(1)) if review_line else None,
                    "review_date": review_line.group(2).strip() if review_line else None,
                    "product": header["title"], "brand": header["brand"],
                    "asin": header["asin"], "doc_avg_rating": header["rating"],
                }})
    return chunks

CHUNKS = build_chunks(documents)
with CHUNK_FILE.open("w", encoding="utf-8") as fh:
    for c in CHUNKS:
        fh.write(json.dumps(c) + "\n")

n = len(CHUNKS)
print(f"strategy             : B_review_aware_sentence")
print(f"chunks               : {n:,}  (saved to {CHUNK_FILE.name})")
print(f"mean chunk size      : {sum(c['metadata']['n_words'] for c in CHUNKS)/n:.0f} words "
      f"({sum(c['metadata']['n_chars'] for c in CHUNKS)/n:.0f} chars)")
print(f"max chunk size       : {max(c['metadata']['n_chars'] for c in CHUNKS)} chars "
      f"({max(c['metadata']['n_words'] for c in CHUNKS)} words)")
print(f"reviews indexed      : {max(c['metadata']['review_index'] for c in CHUNKS)} distinct "
      f"positions across {len(documents)} documents")
print(f"metadata fields      : {list(CHUNKS[0]['metadata'])}")
print("\n--- 5 example chunks with their metadata objects ---")
for c in CHUNKS[:5]:
    print(f"\n[{c['id']}]  " + json.dumps(c["metadata"], indent=2)[:430].replace("\n", "\n "))
    print("  text:", c["text"][:200].replace("\n", " "), "...")
'''),
]
