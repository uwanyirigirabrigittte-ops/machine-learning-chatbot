import sys, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

m = SentenceTransformer(sys.argv[1].rstrip("/"))
c = chromadb.PersistentClient(path="chroma_db", settings=Settings(anonymized_telemetry=False))
col = c.get_collection("power_tool_reviews")

qs = [
    "What do customers complain about on the Makita BL1830 battery?",
    "How long do the DEWALT DC9096 NiCd batteries last?",
    "What do customers dislike about the Dremel flex shaft?",
    "Is the Bosch plunge router worth buying?",
    "Tell me about battery life and charging speed",
    "What are common problems with these tools?",
    "How noisy is it?",
    "What does a DEWALT drill cost in Kenya?",
    "What do people say about the saw blades?",
    "Is it worth buying?",
    "hello there",
    "battery",
    "DEWALT",
    "What is the runtime on a full charge?",
    "Do the batteries die quickly?",
    "Recommend a good drill for home use.",
    "What did customers like about the product?",
]

emb = m.encode(qs, normalize_embeddings=True)
r = col.query(query_embeddings=emb.tolist(), n_results=1, include=["distances"])
lines = [f"{'score':>6}  {'verdict':9} question"]
for q, d in zip(qs, r["distances"][0]):
    s = 1 - d
    lines.append(f"{s:6.3f}  {'REFUSED' if s < 0.30 else 'answers':9} {q}")
with open("/tmp/floor_result.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", len(lines) - 1, "rows")
