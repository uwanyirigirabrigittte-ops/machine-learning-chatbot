"""Assemble the notebook from the cell parts and execute it end to end."""
import sys, time, argparse
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PARTS = ["nb_part_a", "nb_part_b", "nb_part_c", "nb_part_d", "nb_part_e"]


def collect():
    cells = []
    for mod in PARTS:
        try:
            m = __import__(mod)
        except ModuleNotFoundError:
            print(f"[skip] {mod} not present yet")
            continue
        cells += getattr(m, "CELLS", None) or getattr(m, f"CELLS_{mod[-1].upper()}")
        print(f"[load] {mod}: {len(cells)} cells cumulative")
    return cells


def build(path, cells):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src)
                for kind, src in cells]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    nbformat.write(nb, path)
    print(f"[write] {path} ({len(nb.cells)} cells, "
          f"{sum(1 for c in nb.cells if c.cell_type=='code')} code)")


def execute(path, timeout=3600):
    import os, nbclient
    nb = nbformat.read(str(path), as_version=4)
    client = nbclient.NotebookClient(nb, timeout=timeout,
                                     kernel_name=os.environ.get("KERNEL", "py310"),
                                     allow_errors=True, store_widget_state=True,
                                     resources={"metadata": {"path": str(pathlib_parent(path))}})
    t0 = time.time()
    client.execute()
    nbformat.write(nb, path)
    errors = [(i, o) for i, c in enumerate(nb.cells) for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    print(f"[exec] {time.time()-t0:.0f}s | error cells: {len(errors)}")
    for i, o in errors:
        print(f"  cell #{i} [{nb.cells[i].cell_type}] {o['ename']}: {o['evalue'][:300]}")
        print("   " + "\n   ".join(o.get("traceback", [])[-6:])[:1200])
    return len(errors)


def pathlib_parent(path):
    import pathlib
    return pathlib.Path(path).parent.resolve()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Brigitte_Uwanyirigira_RAG_Assignment.ipynb")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    build(a.out, collect())
    if a.run:
        execute(a.out)
