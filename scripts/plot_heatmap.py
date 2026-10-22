"""Models x datasets heatmap of force MAE vs DFT.

Columns: materials groups A, B, C (validate_materials.py), all materials (A+B+C weighted by
each group's share of the full dataset) and organic molecules (validate_model.py).
Cells a model cannot run on are grey.

Importable:
    make(rows, out_path, group_sizes=None, title=None) -> path or None
        rows: [{"name": str,
                "groups": {"A": {...}|None, "B": ..., "C": ...},   # validate_materials group dicts
                "coverage_all": float|None,                        # share of materials it can run on
                "organic_f": float|None}]                          # organic force MAE vs DFT
        group_sizes: {"A": n, "B": n, "C": n} used to weight the "all materials" column
    rows_from_folders(materials_folders, organic_folders, rename=None) -> (rows, group_sizes)
        merges validation folders; a model in several folders keeps the first occurrence

CLI:
    python scripts/plot_heatmap.py --materials validation_materials --organic validation \\
        --out heatmap_force_mae.png

Only numpy/matplotlib/json: no TensorFlow. Missing inputs are skipped with a note.
"""

import argparse
import json
from pathlib import Path

import numpy as np

COLUMNS = [("A", "Materials WITHOUT\nadapted elements\n(metals, alloys, Si…)"),
           ("B", "Materials MIXING both\n(oxides, nitrides,\nhalides…)"),
           ("C", "Materials of ONLY\nadapted elements\n(C, P, S, CHNO crystals)"),
           ("ALL", "All materials\n(base GRACE domain,\nA+B+C weighted)"),
           ("ORG", "Organic molecules\n(teacher domain,\nSPICE)")]


def values(rows, group_sizes):
    """Matrix (models x COLUMNS) of force MAE and per-cell notes."""
    vals = np.full((len(rows), len(COLUMNS)), np.nan)
    notes = {}
    for i, r in enumerate(rows):
        groups = r.get("groups") or {}
        for j, key in enumerate("ABC"):
            g = groups.get(key)
            if g is not None and not g.get("n"):
                notes[(i, j)] = "cannot run\n(elements not\nsupported)"
            if g and g.get("n"):
                vals[i, j] = g["force_mae_meV_A"]
                med = g.get("force_mae_median_structure_meV_A")
                if med and g["force_mae_meV_A"] > 5 * med:
                    notes[(i, j)] = f"median {med:.0f}"
        if group_sizes and all((groups.get(k) or {}).get("n") for k in "ABC"):
            w = np.array([group_sizes[k] for k in "ABC"], dtype=float)
            vals[i, 3] = float(np.dot(w / w.sum(), [groups[k]["force_mae_meV_A"] for k in "ABC"]))
        elif r.get("coverage_all") is not None and r["coverage_all"] < 0.999:
            notes[(i, 3)] = f"runs on {100 * r['coverage_all']:.1f}%\nof materials"
        if r.get("organic_f") is not None:
            vals[i, 4] = r["organic_f"]
    return vals, notes


def make(rows, out_path, group_sizes=None, title=None):
    rows = [r for r in rows if r]
    if not rows:
        print("[plot_heatmap] skipped: no rows")
        return None
    vals, notes = values(rows, group_sizes)
    if np.all(np.isnan(vals)):
        print("[plot_heatmap] skipped: no values")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(13, 1.15 * len(rows) + 2.0))
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(np.clip(vals, 10, 1000)), cmap=cmap, norm=LogNorm(10, 1000),
                   aspect="auto")
    for i in range(len(rows)):
        for j in range(len(COLUMNS)):
            v = vals[i, j]
            if np.isnan(v):
                note = notes.get((i, j))
                text = note if note and note.startswith("cannot") else \
                    (f"n/a\n{note}" if note else "not measured")
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color="0.35")
            else:
                big = f"{v:,.0f}" if v < 1e5 else f"{v:.1e}"
                small = f"\n({notes[(i, j)]})" if (i, j) in notes else ""
                ax.text(j, i, big + small, ha="center", va="center", fontsize=9 if small else 11,
                        fontweight="bold", color="black")
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels([c[1] for c in COLUMNS], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows], fontsize=10)
    ax.xaxis.tick_top()
    for x in (2.5, 3.5):
        ax.axvline(x, color="white", lw=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("force MAE vs DFT (meV/Å, log scale; lower = better)")
    ax.set_xlabel(title or "Force error vs DFT by model and dataset", fontsize=11, labelpad=10)
    fig.tight_layout()
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def rows_from_folders(materials_folders, organic_folders, rename=None):
    """Merge validation folders into heatmap rows (first occurrence of a model wins)."""
    rename = rename or {}
    mat, sizes = {}, None
    for f in materials_folders or []:
        path = Path(f).expanduser() / "materials_validation.json"
        if not path.exists():
            print(f"[plot_heatmap] note: missing {path}")
            continue
        m = json.loads(path.read_text())
        sizes = sizes or m.get("group_sizes_available")
        for name, r in m.get("results", {}).items():
            mat.setdefault(rename.get(name, name), r)
    org = {}
    for f in organic_folders or []:
        path = Path(f).expanduser() / "validation.json"
        if not path.exists():
            print(f"[plot_heatmap] note: missing {path}")
            continue
        for name, r in json.loads(path.read_text()).get("metrics", {}).items():
            name = "GRACE-OFF" if name == "teacher" else rename.get(name, name)
            if "vs_dft" in r:
                org.setdefault(name, r["vs_dft"]["force_mae_meV_A"])
    names = list(dict.fromkeys([*mat, *org]))
    names.sort(key=lambda n: n == "GRACE-OFF")  # teacher last
    rows = [{"name": n, "groups": mat.get(n, {}).get("groups", {}),
             "coverage_all": mat.get(n, {}).get("coverage_all"), "organic_f": org.get(n)}
            for n in names]
    return rows, sizes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--materials", nargs="+", required=True,
                   help="validate_materials.py output folder(s); a model in several keeps the first")
    p.add_argument("--organic", nargs="+", required=True, help="validate_model.py output folder(s)")
    p.add_argument("--out", default="heatmap_force_mae.png")
    args = p.parse_args()
    rows, sizes = rows_from_folders(args.materials, args.organic)
    vals, _ = values(rows, sizes)
    print(f"{'model':16s}" + "".join(f"{c[0]:>12s}" for c in COLUMNS))
    for r, v in zip(rows, vals):
        print(f"{r['name']:16s}" + "".join(f"{'n/a':>12s}" if np.isnan(x) else f"{x:12.1f}" for x in v))
    path = make(rows, args.out, sizes)
    print(f"Saved: {path}" if path else "Nothing written")


if __name__ == "__main__":
    main()
