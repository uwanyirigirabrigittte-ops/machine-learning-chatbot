"""Apply source-level fixes to the built notebook without re-executing it.

The notebook on disk carries saved outputs from a completed run. When the generator
sources change (e.g. a faster model-failover path), the .ipynb would otherwise only pick
up the new code on the next full rebuild, which costs a full LLM pass. This copies the
edited source into the matching cells and leaves every output untouched.
"""
import json

NB = "Brigitte_Uwanyirigira_RAG_Assignment.ipynb"
nb = json.load(open(NB))
before = {i: len(c.get("outputs", [])) for i, c in enumerate(nb["cells"])}

REPLACEMENTS = [
    # 1) Prefer alias model ids, so a per-model quota block on gemini-3.5-flash does not
    #    stop the notebook from finding a working provider.
    (
        '# Tried in order at run time, since Google retires model ids and throttles free tiers.\n'
        'LLM_FALLBACKS   = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-3.8-flash",\n'
        '                   "gemini-3.1-flash-lite-preview"]',
        '# Tried in order at run time, since Google retires model ids and throttles free tiers\n'
        '# per model. The alias ids are listed first because a quota block on one id does not\n'
        '# necessarily apply to the others.\n'
        'LLM_FALLBACKS   = ["gemini-flash-latest", "gemini-3.8-flash", "gemini-3.1-flash-lite-preview",\n'
        '                   "gemini-3.5-flash"]',
    ),
    # 2) A quota block is not fixed by retrying the same model seconds later; move on.
    (
        '                    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in type(exc).__name__ \\\n'
        '                            or "quota" in str(exc).lower():\n'
        '                        print(f"[llm] quota hit while probing {model} - waiting 25s")\n'
        '                        time.sleep(25)',
        '                    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in type(exc).__name__ \\\n'
        '                            or "quota" in str(exc).lower():\n'
        '                        # A per-model quota block is not fixed by retrying the same model\n'
        '                        # seconds later: move to the next candidate straight away so the\n'
        '                        # notebook still gets a working provider.\n'
        '                        print(f"[llm] {model} is quota-blocked - trying the next model")\n'
        '                        break',
    ),
    (
        '        if working is None:\n'
        '            raise RuntimeError("no Gemini model responded - check the key and the model list")',
        '        if working is None:\n'
        '            raise RuntimeError("no Gemini model responded - check the key and the model list")\n'
        '        if working != LLM_MODEL:\n'
        '            print(f"[llm] {LLM_MODEL} was unavailable; using {working} instead")',
    ),
    # 3) Say what went wrong instead of a bare "retry in a minute".
    (
        '                return "[error] the LLM free-tier quota is exhausted - retry in a minute."',
        '                return ("[error] the LLM free-tier quota is exhausted for this key, so no "\n'
        '                        "answer could be generated. Retrieval still worked - the question was "\n'
        '                        "found in the review chunks. Wait a minute and ask again, or use a "\n'
        '                        "different key, or set RAG_LOCAL_FALLBACK=1 to answer offline.")',
    ),
    # 4) When generation fails but retrieval succeeded, still show the evidence.
    (
        '    raw = call_llm(prompt_fn(question, hits))\n'
        '    used = cited_ids(raw)\n'
        '    return {"answer": raw,',
        '    raw = call_llm(prompt_fn(question, hits))\n'
        '    used = cited_ids(raw)\n'
        '    if raw.startswith("[error]"):\n'
        '        # Generation failed but retrieval did not. Show the evidence that was found so the\n'
        '        # turn is still informative instead of looking like "no answer at all".\n'
        '        return {"answer": raw, "sources": hits[:3], "all_hits": hits, "abstained": False,\n'
        '                "best_score": best, "raw": raw, "error": "llm_unavailable"}\n'
        '    return {"answer": raw,',
    ),
]

applied = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    for old, new in REPLACEMENTS:
        if old in src:
            src = src.replace(old, new)
            applied += 1
    cell["source"] = src.splitlines(keepends=True)

after = {i: len(c.get("outputs", [])) for i, c in enumerate(nb["cells"])}
assert before == after, "outputs changed - aborting"
json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print(f"applied {applied} source edits; outputs untouched on all {len(nb['cells'])} cells")
