import sys, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

m = SentenceTransformer(sys.argv[1].rstrip("/"))
c = chromadb.PersistentClient(path="chroma_db", settings=Settings(anonymized_telemetry=False))
col = c.get_collection("power_tool_reviews")
print("collection count:", col.count())

qs = ["battery runtime complaints", "how noisy is the router", "price in Kenya",
      "saw blade quality", "is it worth buying"]
emb = m.encode(qs, normalize_embeddings=True)
print("embeddings shape:", emb.shape)

r = col.query(query_embeddings=emb.tolist(), n_results=1, include=["distances"])
print("type distances:", type(r["distances"]), "outer len:", len(r["distances"]))
print("inner len:", [len(x) for x in r["distories"]] if False else [len(x) for x in r["distances"]])
print("raw:", r["distances"])
