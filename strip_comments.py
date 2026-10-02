"""Strip `#` comments from the notebook's code cells and the generator sources.

Comments are located with `tokenize`, which understands Python string literals, so a `#`
inside a prompt string or an HTML colour is never touched. Only cells whose type is
"code" are modified: the Markdown cells hold the task descriptions and the write-up and
are left exactly as they are, as are the saved outputs of a completed run.

Each comment is deleted as a character span and the rest of its line is kept, so line
numbering and the surrounding indentation survive.
"""
import io
import json
import tokenize
from pathlib import Path

PARTS = [f"nb_part_{c}.py" for c in "abcde"]
NB = Path("Brigitte_Uwanyirigira_RAG_Assignment.ipynb")


def comment_spans(src):
    """Character ranges of every COMMENT token, as (start, end) offsets."""
    lines = src.splitlines(keepends=True)
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line)

    def offset(row, col):
        return starts[row - 1] + col

    return [(offset(t.start[0], t.start[1]), offset(t.end[0], t.end[1]))
            for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type == tokenize.COMMENT]


def strip_comments(src):
    """Delete comment spans, then tidy the whitespace each one leaves behind."""
    spans = comment_spans(src)
    if not spans:
        return src
    out, prev = [], 0
    for start, end in spans:
        out.append(src[prev:start])
        # Keep whatever code followed the '#' on the same line.
        tail = src[end:]
        nl = tail.find("\n")
        if nl != -1 and tail[:nl].strip():
            out.append(tail[:nl])
        prev = end
    out.append(src[prev:])
    text = "".join(out)

    cleaned, blanks = [], 0
    for line in text.split("\n"):
        line = line.rstrip()
        if line:
            blanks = 0
            cleaned.append(line)
        else:
            blanks += 1
            if blanks <= 1:
                cleaned.append("")
    return "\n".join(cleaned).strip("\n") + "\n"


def count_comments(src):
    return len(comment_spans(src))


def main():
    nb = json.loads(NB.read_text())
    before = after = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        before += count_comments(src)
        cell["source"] = strip_comments(src).splitlines(keepends=True)
        after += count_comments("".join(cell["source"]))
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"notebook code cells: {before} comments -> {after}; markdown and outputs untouched")

    # The part files keep every cell as a *string literal*, so tokenize cannot see the
    # comments inside them. Each cell is therefore stripped through the same helper and
    # written back into its `("code", r'''...''')` slot.
    for name in PARTS:
        p = Path(name)
        if not p.exists():
            continue
        text = p.read_text()
        ns: dict = {}
        exec(compile(text, name, "exec"), ns)
        cells = next(v for k, v in ns.items() if k.startswith("CELLS_"))
        n = 0
        for idx, (kind, body) in enumerate(cells):
            if kind == "code":
                n += count_comments(body)
                cells[idx] = (kind, strip_comments(body))
        # Round-trip through repr so quoting and indentation are preserved exactly.
        rebuilt = [f"CELLS_{name[-5].upper()} = ["]
        for kind, body in cells:
            if kind == "md":
                rebuilt.append(f'("md", r"""{body}"""),')
            else:
                rebuilt.append(f"(\"code\", r'''{body}'''),")
        rebuilt.append("]")
        p.write_text("\n".join(rebuilt) + "\n")
        print(f"{name}: {n} comments -> {count_comments(''.join(c[1] for c in cells))}")


if __name__ == "__main__":
    main()
