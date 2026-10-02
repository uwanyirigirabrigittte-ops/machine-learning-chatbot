"""Re-execute only the LLM-dependent cells of the built notebook, in place.

The data pipeline (corpus, chunks, embeddings, Chroma) is cached, so the cheap cells
are re-run in a scratch kernel to rebuild state, but only the outputs of the cells
listed in --cells are written back into the .ipynb. Everything else keeps its saved
output, which keeps the repair fast and avoids touching Task 1-5 results.
"""
import argparse, json, time
from jupyter_client.manager import start_new_kernel

NB = "Brigitte_Uwanyirigira_RAG_Assignment.ipynb"

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="21,22,25,28", help="code-cell indices to refresh")
ap.add_argument("--prefix", type=int, default=20, help="run code cells 0..prefix first (state only)")
ap.add_argument("--retries", type=int, default=40, help="max quota retries per cell")
args = ap.parse_args()

targets = {int(x) for x in args.cells.split(",") if x.strip() != ""}
nb = json.load(open(NB))
# Indices are positions within the CODE cells, which is what --prefix counts too.
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
if max(targets) >= len(code):
    raise SystemExit(f"only {len(code)} code cells; got index {max(targets)}")

km, kc = start_new_kernel(kernel_name="python3", cwd=".")


def run(src, label, quota_aware):
    """Execute one cell. Returns (text, ok). When quota_aware, a quota error means 'not
    yet' and is retried with a long backoff. ok=False means the cell did not complete, in
    which case the caller must keep the previously saved output rather than overwrite it."""
    attempt = 0
    while True:
        attempt += 1
        kc.execute(src)
        stream, failed, quota = [], False, False
        while True:
            m = kc.get_iopub_msg(timeout=3600)
            t, c = m["msg_type"], m["content"]
            if t == "stream":
                stream.append(c["text"])
            elif t == "error":
                failed = True
                tb = "".join(c["traceback"])
                quota = "429" in tb or "RESOURCE_EXHAUSTED" in tb or "quota" in tb.lower()
                if not quota:
                    print(f"\n!!! {label} FAILED\n{tb[-1500:]}")
            elif t == "status" and c["execution_state"] == "idle":
                break
        text = "".join(stream)
        if not failed:
            return text, True
        if quota_aware and quota and attempt < args.retries:
            wait = min(600, 60 * attempt)
            print(f"  [quota] {label}: waiting {wait}s (attempt {attempt}/{args.retries})", flush=True)
            time.sleep(wait)
            continue
        return text, False


try:
    print(f"state prefix: code cells 0..{args.prefix}", flush=True)
    for i, c in enumerate(code[: args.prefix + 1]):
        if i in targets:
            continue
        _, ok = run("".join(c["source"]), f"state cell {i}", quota_aware=False)
        if not ok:
            raise SystemExit(f"state cell {i} failed; aborting before any output is written")
        print(f"  state cell {i} ok", flush=True)

    refreshed = []
    for i in sorted(targets):
        c = code[i]
        print(f"refreshing code cell {i} ...", flush=True)
        text, ok = run("".join(c["source"]), f"target cell {i}", quota_aware=True)
        if not ok:
            print(f"  [skip] cell {i} did not complete; keeping its previously saved output", flush=True)
            continue
        c["outputs"] = [{"output_type": "stream", "name": "stdout", "text": text}] if text else []
        c["execution_count"] = None
        refreshed.append(i)
        print(f"  cell {i} refreshed ({len(text)} chars)", flush=True)

    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
    print("[write]", NB, "refreshed cells:", refreshed)
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
