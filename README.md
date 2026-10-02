# RAG over Amazon Power-Tool Customer Reviews

A retrieval-augmented generation (RAG) system that answers questions about what Amazon
customers actually report about power tools — battery runtime, durability, failure modes,
noise, blade quality — and refuses to answer when the reviews do not support an answer.

Everything lives in one notebook: **`Brigitte_Uwanyirigira_RAG_Assignment.ipynb`**
(48 cells: 18 Markdown, 30 code). Every cell has been run, so all outputs, tables and the
chat panel are visible without re-running anything.

---

## Topic and data source

**What customers say about power tools and outdoor power equipment** — drills and drivers,
saws, sanders, routers, grinders, air compressors, nailers, jigs, blades and tool
batteries.

| Item | Value |
|---|---|
| Source | Julian McAuley (UCSD), *Amazon Review Data* (2014) — `Tools & Home Improvement` |
| Access | `https://jmcauley.ucsd.edu/data/amazon/` |
| Licence | For research use; please cite the original dataset |
| Corpus | 45 TXT documents, ~184,000 words (assignment requires 30–100) |
| Reviews kept | 1,267 review chunks → 1,577 retrieval units |

The pipeline keeps only reviews whose product title names a tool brand, so ordinary
batteries sold by the same sellers do not leak into a tools corpus.

---

## How it works

```
download → clean → chunk → embed → index → retrieve → guardrails → generate → evaluate
```

| Stage | What happens |
|---|---|
| **Collect** | Two `.json.gz` archives are streamed to disk, then filtered to the `Power & Hand Tools` leaf categories and written as one TXT file per product. |
| **Clean** | HTML and markup stripped, whitespace collapsed, a provenance row recorded per file. |
| **Chunk** | Recursive character splitting to ~1,200 characters, respecting sentence boundaries. Chunks longer than the model's 256-token limit are split again at a sentence boundary. |
| **Embed** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), mean pooling and L2 normalisation, so cosine similarity is a plain dot product. Vectors are cached in `artifacts/embeddings.npy`. |
| **Index** | ChromaDB, persistent on disk, cosine space. The collection is reused across runs unless the chunking changed. |
| **Retrieve** | Top-6 nearest chunks. The query is embedded with the same model, so scores are comparable. |
| **Guardrails** | A relevance floor refuses weak matches, and the prompt separates instructions from retrieved data. |
| **Generate** | Google Gemini via `google-genai`, with model-id failover. Answers must cite `[S#]` markers. |
| **Evaluate** | A 10-question test set — 6 single-product lookups, 2 cross-product synthesis questions and 2 out-of-scope questions that must be refused — scored automatically for retrieval hit rate, answer correctness, correct abstention, citation validity and faithfulness. |

### Why the numbers are what they are

Absolute cosine scores are low (~0.2–0.7) because two 200-word review passages rarely
paraphrase each other. What matters is the **ordering**: passages about the same subject
score far above random pairs, and same-product passages sit above random pairs but below
the related pair. The relevance floor is calibrated from that measured distribution rather
than guessed.

---

## Task map

| Task | Content |
|---|---|
| 0 | Topic and source selection, including questions the system must refuse |
| 1 | Corpus collection, cleaning, provenance |
| 2 | Loading and parsing the documents |
| 3 | Chunking strategy and its structural justification |
| 4 | Embedding model, limits, and a semantic-similarity proof |
| 5 | ChromaDB build, persistence proof, retrieval test |
| 6 | LLM integration, context rendering, prompt V1 vs V2, injection defence |
| 7 | Interactive chat panel, scripted transcript, console REPL |
| 8 | Evaluation, metrics, reflection |

---

## Running it

**Requirements:** Python 3.10, and the packages the setup cell installs
(`sentence-transformers`, `chromadb`, `ipywidgets`, `google-genai`).

### The API key

Retrieval, evaluation and the interface all work **without** a key. Only text generation
needs one, and it is never stored in the notebook.

The key is read from the first of these that is present:

1. the `GOOGLE_API_KEY` environment variable
2. `~/.rag_api_key` or `.rag_api_key` in the project folder
3. Colab's secret store

To use the second option, which keeps the key out of your shell history:

```bash
echo -n 'your-key-here' > ~/.rag_api_key
```

### Run the whole notebook

```bash
export GOOGLE_API_KEY=...      # optional
jupyter lab
```

Then `Kernel → Restart & Run All`. A first run downloads ~126 MB of source archives and
builds the index, so budget a few minutes; later runs reuse the cache and the persisted
Chroma collection.

### Try the chat without the notebook

The console REPL accepts questions on stdin and ends on `exit`:

```python
chat_console()
```

---

## Two things worth knowing

**The free Gemini tier is quota-limited per model.** If generation fails you will see
`[error] the LLM free-tier quota is exhausted` — retrieval still worked, and the panel
shows the chunks it found. The notebook automatically fails over to another model id
(`gemini-flash-latest`, then the 3.x ids). Answers that are unavailable for this reason
are reported as such rather than being scored as if the system had abstained.

**Open the notebook from its own folder.** Jupyter starts the kernel in the *server*
folder, which is not necessarily where the notebook lives. The notebook locates its
project directory from its own artefacts, so it works either way, but the resolved path is
printed in the setup cell.

---

## Project layout

| Path | Purpose |
|---|---|
| `Brigitte_Uwanyirigira_RAG_Assignment.ipynb` | The deliverable, fully executed |
| `data/raw/` | 45 cleaned TXT documents |
| `data/clean/` | Normalised copies |
| `artifacts/` | Cached chunks, embeddings, ids, collection log |
| `chroma_db/` | Persisted ChromaDB collection `power_tool_reviews` |
| `nb_part_a.py` … `nb_part_e.py` | Notebook cells as source (Markdown and code) |
| `build_notebook.py` | Assembles and optionally executes the notebook |

`build_notebook.py` regenerates the notebook from the part files:

```bash
python build_notebook.py          # rebuild only
python build_notebook.py --run    # rebuild and execute every cell
```

Regenerating **discards the saved outputs**, so take a copy first if you need them.

---

## Limitations

- Answers summarise customer opinion, not verified specifications. They are not a basis
  for a purchase or a safety decision.
- Retrieval quality is bounded by the embedding model; the 384-dimension MiniLM encoder
  is fast but weaker than a larger encoder on long, technical passages.
- The evaluation set is 10 hand-written questions. It demonstrates the scoring machinery;
  it is not a statistically meaningful benchmark.
- The corpus is English-language Amazon reviews from one category, so nothing here
  generalises to other product categories or languages.
