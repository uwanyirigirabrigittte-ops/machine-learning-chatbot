"""Probe retrieval + read the evidence needed to author the 10 evaluation questions."""
import json, numpy as np, chromadb
from pathlib import Path
from chromadb.config import Settings

ROOT = Path("/home/student/machine_learning_RAG")
chunks = [json.loads(l) for l in open(ROOT/"artifacts/chunks.jsonl")]
ids = json.loads((ROOT/"artifacts/embedding_ids.json").read_text())
emb = np.load(ROOT/"artifacts/embeddings.npy")
IDX = {i: n for n, i in enumerate(ids)}
client = chromadb.PersistentClient(path=str(ROOT/"chroma_db"), settings=Settings(anonymized_telemetry=False))
col = client.get_collection("power_tool_reviews")

def show(q, n=5, width=200):
    print("=" * 90); print("Q:", q)
    r = col.query(query_texts=[q], n_results=n, include=["documents", "metadatas", "distances"])
    for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        print(f"  {1-dist:.3f} | {m['filename'][:44]:44} p{m['page_number']} rev{m['review_index']} "
              f"({m['review_rating']}*) {m['chunk_id'].split('__')[-1]}")
        print(f"        {d[:width]}...")

for q in [
    "How long does the DEWALT DC970K-2 drill battery last and how fast does it charge?",
    "Why do Makita BL1830 batteries fail so early according to reviewers?",
    "What problems do customers report with the Bosch 4100-09 table saw blade alignment?",
    "Which air compressor do reviewers complain about most for oil leaks or pressure loss?",
    "What do customers say about the Kreg pocket hole jig ease of use?",
    "What is the common complaint about the Dremel rotary tool flex shaft?",
    "Do reviewers say the DEWALT 18V NiCad batteries die quickly?",
    "What do customers praise about the Hitachi compound miter saw blade quality?",
    "Do any reviews mention using these tools in Kenya or in dollars?",
    "What is the average lifespan of a DEWALT DC9096 NiCad battery pack?",
]:
    show(q)

print("\n\n########## EVIDENCE SNIPPETS ##########")
def dump(pred, label, k=2, width=700):
    print("-" * 90); print(label)
    n = 0
    for c in chunks:
        if pred(c["text"].lower()):
            print(f"\n[{c['metadata']['filename'][:46]} p{c['metadata']['page_number']} "
                  f"rev{c['metadata']['review_index']} ({c['metadata']['review_rating']}*)]")
            print(c["text"][:width])
            n += 1
            if n >= k: break

dump(lambda t: "batter" in t and "30 minutes" in t, "charge time 30 min")
dump(lambda t: "20 minutes" in t and "batter" in t, "runtime 20 minutes")
dump(lambda t: "oil leak" in t, "oil leak compressor")
dump(lambda t: "blade" in t and "align" in t, "blade alignment")
