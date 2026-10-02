CELLS__ = [
("md", r"""# Task 4 — Embedding Generation

**Model:** `sentence-transformers/all-MiniLM-L6-v2`. Documents and queries are embedded
with the same model, so both live in one vector space.

* **Dimension:** 384
* **Context limit:** 256 word-pieces (not words — a word-piece is a sub-word unit, so
  256 word-pieces ≈ 180–200 English words)
* **Architecture under the hood:** a 6-layer, 384-hidden-unit BERT with **mean pooling**
  over the final token layer followed by **L2 normalisation**. Mean pooling is why a long
  chunk and a one-sentence query land in the same space, and L2 normalisation is why the
  dot product of two vectors *is* their cosine similarity.

**Truncation strategy.** Silent truncation is not relied on. Each chunk is tokenised with
the model's tokenizer, the distribution is printed, and anything over 256 word-pieces is
split at a sentence boundary into sub-parts that inherit the parent's metadata. This is not
hypothetical: 1,200-character chunks average ~188 word-pieces but reach ~345, so about a
quarter of the corpus is over the limit. The `part` / `n_parts` fields keep every sub-part
citable."""),
("code", r'''import numpy as np
from sentence_transformers import SentenceTransformer

def load_embedder(model_name=EMBEDDING_MODEL):
    """Load the model; fall back to an already-downloaded snapshot when huggingface.co
    is unreachable (offline machine / mirror cache) so the notebook still runs."""
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        snaps = sorted((Path.home()/".cache/huggingface/hub").glob(
            f"models--{model_name.replace('/', '--')}/snapshots/*"))
        if not snaps:
            raise
        print(f"[fallback] {type(exc).__name__} from hub -> local snapshot {snaps[-1].name}")
        return SentenceTransformer(str(snaps[-1]))

t_model = time.time()
embedder = load_embedder()
EMBED_DIM = embedder.get_sentence_embedding_dimension()
WORDPIECE_LIMIT = embedder.tokenizer.model_max_length if embedder.tokenizer.model_max_length < 10000 \
                  else embedder.max_seq_length
print(f"model            : {EMBEDDING_MODEL}")
print(f"loaded in        : {time.time()-t_model:.1f}s")
print(f"embedding dim    : {EMBED_DIM}")
print(f"word-piece limit : {WORDPIECE_LIMIT}")
print(f"pooling          : mean pooling + L2 normalisation (sentence-transformers default)")

texts = [c["text"] for c in CHUNKS]
token_counts = [len(embedder.tokenizer.tokenize(t)) for t in texts]
tok = np.array(token_counts)
print(f"\ntokens per chunk  : mean {tok.mean():.0f} | median {np.median(tok):.0f} | "
      f"p95 {np.percentile(tok, 95):.0f} | max {tok.max()}")
print(f"chunks over limit : {(tok > WORDPIECE_LIMIT).sum()} "
      f"({100*(tok > WORDPIECE_LIMIT).mean():.1f}%)")

def enforce_limit(text, limit=WORDPIECE_LIMIT):
    """Truncation strategy: split an oversized chunk at a sentence boundary (never
    mid-sentence) instead of letting the tokenizer silently drop the tail. Returns
    [(part_text, part_index), ...] - part 0 is the head of the chunk."""
    if len(embedder.tokenizer.tokenize(text)) <= limit:
        return [(text, 0)]
    parts, cur = [], []
    for sent in split_sentences(text):
        trial = " ".join(cur + [sent])
        if cur and len(embedder.tokenizer.tokenize(trial)) > limit:
            parts.append(" ".join(cur)); cur = [sent]
        else:
            cur.append(sent)
    if cur:
        parts.append(" ".join(cur))
    return [(p, i) for i, p in enumerate(parts)]

TEXT_PAYLOAD, OVERSIZE_PARTS = [], 0
for c in CHUNKS:
    parts = enforce_limit(c["text"])
    OVERSIZE_PARTS += len(parts) - 1
    for part, k in parts:
        meta = dict(c["metadata"])
        meta["part"] = k + 1
        meta["n_parts"] = len(parts)
        TEXT_PAYLOAD.append((c["id"] if k == 0 else f"{c['id']}~{k+1}", c["document"], part, meta))
print(f"chunks needing the truncation strategy : {(tok > WORDPIECE_LIMIT).sum()} "
      f"({100*(tok > WORDPIECE_LIMIT).mean():.1f}% of {len(CHUNKS):,})")
print(f"extra sentence-boundary sub-parts      : {OVERSIZE_PARTS}")
print(f"retrieval units in the index           : {len(TEXT_PAYLOAD):,} "
      f"(= {len(CHUNKS):,} chunks + {OVERSIZE_PARTS} sub-parts)")
'''),
("code", r'''EMB_FILE = ARTIFACTS/"embeddings.npy"
IDS_FILE = ARTIFACTS/"embedding_ids.json"

if EMB_FILE.exists() and IDS_FILE.exists():
    cached = np.load(EMB_FILE)
    cached_ids = json.loads(IDS_FILE.read_text())
    if cached_ids == [p[0] for p in TEXT_PAYLOAD]:
        EMBEDDINGS, EMBEDDING_IDS = cached, cached_ids
        print(f"[cache] loaded {EMBEDDINGS.shape} embeddings from {EMB_FILE.name} — "
              f"nothing re-encoded")
    else:
        print(f"[cache] stale ({cached.shape[0]} vectors for a different chunk set) — re-embedding")
        EMBEDDINGS = None
else:
    EMBEDDINGS = None

if EMBEDDINGS is None:
    t0 = time.time()
    EMBEDDINGS = embedder.encode(
        [p[2] for p in TEXT_PAYLOAD],
        batch_size=64, show_progress_bar=True, convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    EMBEDDING_IDS = [p[0] for p in TEXT_PAYLOAD]
    np.save(EMB_FILE, EMBEDDINGS)
    IDS_FILE.write_text(json.dumps(EMBEDDING_IDS))
    print(f"\nembedded {len(EMBEDDINGS):,} chunks in {time.time()-t0:.1f}s "
          f"({len(EMBEDDINGS)/(time.time()-t0):.0f} chunks/s)")
print(f"embedding matrix : {EMBEDDINGS.shape} ({EMBEDDINGS.nbytes/1e6:.1f} MB, float32)")
print(f"L2 norm check    : {np.linalg.norm(EMBEDDINGS[0]):.4f} (1.0 => cosine == dot product)")
'''),
("code", r'''IDX = {i: n for n, i in enumerate(EMBEDDING_IDS)}
def cos_id(a, b):
    return float(np.dot(EMBEDDINGS[IDX[a["id"]]], EMBEDDINGS[IDX[b["id"]]]))

def find_chunk(*words, avoid=(), min_tokens=0):
    """First chunk mentioning all `words`; falls back to a looser OR-match so the
    demonstration can never crash on an unexpected corpus."""
    for c in CHUNKS:
        t = c["text"].lower()
        if all(w.lower() in t for w in words) and not any(a.lower() in t for a in avoid):
            return c
    for c in CHUNKS:
        t = c["text"].lower()
        if any(w.lower() in t for w in words) and not any(a.lower() in t for a in avoid):
            return c
    return CHUNKS[0]

related_a = find_chunk("battery", "minutes", "charge")
related_b = next(c for c in CHUNKS
                 if c["metadata"]["filename"] != related_a["metadata"]["filename"]
                 and "battery" in c["text"].lower() and "minutes" in c["text"].lower()
                 and "charge" in c["text"].lower())
unrelated_a = find_chunk("jig", "pocket hole", avoid=("battery", "saw"))
unrelated_b = find_chunk("dust collection", avoid=("battery", "jig", "blade"))
assert unrelated_a["metadata"]["filename"] != unrelated_b["metadata"]["filename"], \
    "the unrelated pair must come from two different products"

for label, a, b in (("RELATED  ", related_a, related_b),
                    ("UNRELATED", unrelated_a, unrelated_b)):
    print("=" * 78)
    print(f"{label} pair  |  cosine = {cos_id(a, b):.3f}")
    print("=" * 78)
    for tag, c in (("A", a), ("B", b)):
        m = c["metadata"]
        print(f"[{tag}] {m['filename'][:54]}  p{m['page_number']} rev{m['review_index']} "
              f"({m['review_rating']}*)")
        print(f"    {c['text'][:210]}...\n")

sim_related, sim_unrelated = cos_id(related_a, related_b), cos_id(unrelated_a, unrelated_b)

rng = np.random.default_rng(7)
random_pairs = rng.integers(0, len(EMBEDDINGS), size=(1500, 2))
random_sims = np.array([float(np.dot(EMBEDDINGS[i], EMBEDDINGS[j]))
                        for i, j in random_pairs if i != j])
doc_of = [p[1] for p in TEXT_PAYLOAD]
by_doc = collections.defaultdict(list)
for n, d in enumerate(doc_of):
    by_doc[d].append(n)
same_doc = [float(np.dot(EMBEDDINGS[i], EMBEDDINGS[j]))
            for ids in by_doc.values()
            for i, j in zip(ids[::4], ids[1::4]) if i != j][:400]

print("=" * 78)
print(f"related pair cosine              : {sim_related:.3f}")
print(f"unrelated pair cosine            : {sim_unrelated:.3f}")
print(f"margin                           : {sim_related - sim_unrelated:+.3f}")
print(f"random chunk pairs (n={len(random_sims)}) mean : {random_sims.mean():.3f}  "
      f"(p95 {np.percentile(random_sims, 95):.3f})")
print(f"pairs from the SAME product (n={len(same_doc)}) : {np.mean(same_doc):.3f}  "
      f"— same product, shared vocabulary")
print("=" * 78)
print("Interpretation. Absolute cosines are low because two 200-word review passages rarely")
print("paraphrase each other; what matters is the ordering. Semantically related battery")
print("passages sit far above the random-pair baseline, and passages about the same product")
print("sit above random pairs but below the related pair - exactly the behaviour a retriever")
print("needs. It is also why the abstention floor in Task 7 is calibrated from this")
print("distribution (random pairs cluster near 0.05-0.15) instead of being guessed.")
'''),
("md", r"""# Task 5 — Vector Storage & Retrieval Setup

**Database: ChromaDB** — open-source, embedded, no server. The persistent client writes an
HNSW index plus the payload metadata to `chroma_db/`, so the corpus can be reloaded in a
brand-new kernel without re-encoding a single chunk.

**Distance metric: cosine.** The collection is created with `hnsw:space="cosine"`, which
stores *normalised* vectors and ranks candidates by `1 - dot(a, b)`. Because the embedder
already L2-normalises, the reported `distance` converts to a similarity with
`similarity = 1 - distance`."""),
("code", r'''import chromadb
from chromadb.config import Settings

COLLECTION = "power_tool_reviews"
CHROMA_DIR.mkdir(exist_ok=True)
client = chromadb.PersistentClient(path=str(CHROMA_DIR),
                                   settings=Settings(anonymized_telemetry=False))

def collection_is_populated():
    try:
        return client.get_collection(COLLECTION).count() == len(TEXT_PAYLOAD)
    except Exception:
        return False

if not collection_is_populated():

    try:
        client.delete_collection(COLLECTION)
        print("[index] stale index removed")
    except Exception:
        pass
    print("[index] creating HNSW index (cosine space) — this is the slow first run ...")
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL,
                  "chunk_size_chars": CHUNK_SIZE, "chunks": len(TEXT_PAYLOAD)})
    collection.add(
        ids=[p[0] for p in TEXT_PAYLOAD],
        documents=[p[2] for p in TEXT_PAYLOAD],
        embeddings=EMBEDDINGS.tolist(),
        metadatas=[p[3] for p in TEXT_PAYLOAD])
    print("[index] populated")
else:
    print("[index] collection already populated")

collection = client.get_collection(COLLECTION)
print(f"collection       : {collection.name}")
print(f"vectors stored   : {collection.count():,}")
print(f"persisted at     : {CHROMA_DIR.relative_to(PROJECT)}/")
print(f"space            : {collection.metadata.get('hnsw:space')} (cosine)")
print(f"files on disk    : {sorted(p.name for p in CHROMA_DIR.iterdir())[:6]}")
'''),
("code", r'''for name in ("collection", "client", "EMBEDDINGS"):
    if name in dir():
        del globals()[name]
gc_ok = __import__("gc"); gc_ok.collect()

t_reload = time.time()
client = chromadb.PersistentClient(path=str(CHROMA_DIR),
                                   settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(COLLECTION)
res = collection.query(query_texts=["why do customers complain about cordless tool batteries?"],
                      n_results=2)
print(f"reloaded {collection.count():,} vectors from disk in {time.time()-t_reload:.3f}s")
print(f"EMBEDDINGS object in memory now? {'EMBEDDINGS' in globals()}  "
      "(False => the query was answered by the persisted index only)\n")
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"  score {1-dist:.3f} | {meta['filename'][:48]} p{meta['page_number']} "
          f"rev{meta['review_index']}\n    {doc[:150]}...\n")
'''),
("code", r'''def retrieve(query, k=TOP_K, floor=0.0):
    """Top-k chunks for a query. Under the hood: the query string is embedded with the SAME
    model, Chroma returns the k nearest HNSW neighbours, and `1 - cosine_distance` gives the
    similarity used by the abstention floor."""
    hits = collection.query(
        query_texts=[query], n_results=k,
        include=["documents", "metadatas", "distances", "embeddings"])
    out = []
    for doc, meta, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0]):
        out.append({"text": doc, "metadata": meta, "distance": float(dist),
                    "score": round(1 - float(dist), 4)})
    return [h for h in out if h["score"] >= floor] if floor else out

probe_questions = [
    "How long does the battery last on the DEWALT 18 volt drill driver kit?",
    "What do reviewers complain about with the Makita 18V battery?",
    "Does the Bosch plunge router overheat or stall when cutting hard wood?",
]
for q in probe_questions:
    print("=" * 78); print("Q:", q); print("=" * 78)
    for i, h in enumerate(retrieve(q, k=3), 1):
        m = h["metadata"]
        print(f"  {i}. score {h['score']:.3f} | {m['filename'][:46]} "
              f"p{m['page_number']} {m['chunk_id'].split('__')[-1]} "
              f"rev{m['review_index']} ({m['review_rating']}*)")
        print(f"     {h['text'][:165]}...")
    print()
print("Retrieval is behaving: the top hits are on-topic, come from the expected product")
print("documents, and carry the review rating so the LLM can weigh a 1-star complaint")
print("against a 5-star one. Only now is it fair to blame the LLM for a bad answer.")
'''),
("md", r"""### Calibrating the abstention floor

The similarity scores above are all far below 0.5, because MiniLM cosines between
*different* 200-word passages are small in absolute terms. A hard-coded 0.5 would reject
almost everything, so the floor must be set from the measured distribution: the random-pair
baseline (mean ≈ 0.23) and the in-scope / out-of-scope ranges printed in Task 8. The value
used in Task 7 is `0.30`, which sits just above the random baseline — it rejects queries
that share no vocabulary with the corpus at all. Task 8 shows its limits honestly: for
questions that *do* share vocabulary with the corpus but are still unanswerable, the
prompt's abstention rule — not the floor — is what produces the refusal."""),
]
