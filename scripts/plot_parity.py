"""Force parity plots (model vs reference force components, with MAE and R²).

Importable:
    make(folder, out_dir=None, rename=None)          -> list of written PNG paths
        organic validation folder (validate_model.py: predictions.npz + validation.json):
        parity_forces.png (vs GRACE-OFF) and parity_forces_vs_dft.png (vs DFT, if labelled)
    make_materials(folder, out_path=None, models=None, rename=None) -> path or None
        materials validation folder (validate_materials.py: predictions.npz + JSON):
        grid of groups A/B/C (rows) x models (columns), model forces vs DFT
    grid(out_path, reference, panels, reference_label) -> path

CLI:
    python scripts/plot_parity.py validation/exp_003/organic
    python scripts/plot_parity.py validation/exp_003/materials --materials --models base exp_003

Only numpy/matplotlib/json: no TensorFlow. Missing inputs are skipped with a note.
"""

import argparse
import json
from pathlib import Path

import numpy as np

GROUP_LABEL = {"A": "A: no adapted elements",
               "B": "B: adapted + other elements",
               "C": "C: only adapted elements"}


def r2_score(pred, ref):
    """Coefficient of determination of pred against ref (force components pooled)."""
    pred, ref = np.asarray(pred).ravel(), np.asarray(ref).ravel()
    return float(1 - np.sum((pred - ref) ** 2) / np.sum((ref - ref.mean()) ** 2))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _stats(pred, ref):
    pred, ref = np.asarray(pred), np.asarray(ref)
    mae = 1000 * float(np.abs(pred - ref).mean())
    r2 = r2_score(pred, ref)
    mae_txt = f"{mae:.0f}" if mae < 1e4 else f"{mae:.1e}"
    if mae >= 1e4:  # diverging model: the mean is dominated by blow-ups, quote the median too
        med = 1000 * float(np.median(np.abs(pred - ref)))
        mae_txt += f" (median |ΔF| {med:.0f})"
    return f"MAE {mae_txt} meV/Å, " + (f"R² {r2:.3f}" if r2 > -10 else "R² ≪ 0 (diverges)")


def grid(out_path, reference, panels, reference_label, max_points=30000):
    """One row of parity panels. panels: [(title, predicted forces)] vs the same reference."""
    plt = _plt()
    ref = np.asarray(reference).ravel()
    sel = np.random.default_rng(0).choice(len(ref), size=min(len(ref), max_points), replace=False)
    lim = 1.1 * np.percentile(np.abs(ref), 99.5)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.9),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (title, pred) in zip(axes[0], panels):
        p = np.asarray(pred).ravel()
        ax.scatter(ref[sel], np.clip(p[sel], -lim, lim), s=2, alpha=0.25, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{reference_label} force component (eV/Å)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("model force component (eV/Å)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def make(folder, out_dir=None, rename=None, prefix=""):
    """Parity plots for an organic validation folder (validate_model.py output)."""
    folder = Path(folder).expanduser()
    out_dir = Path(out_dir).expanduser() if out_dir else folder
    rename = rename or {}
    if not (folder / "predictions.npz").exists():
        print(f"[plot_parity] skipped: no predictions.npz in {folder}")
        return []
    data = np.load(folder / "predictions.npz")
    info = json.loads((folder / "validation.json").read_text()) \
        if (folder / "validation.json").exists() else {}
    trained = list(info.get("trained", {})) or \
        [f[:-7] for f in data.files if f.endswith("_forces") and f[:-7] not in ("base", "teacher", "dft")]
    students = [n for n in ["base", *trained] if f"{n}_forces" in data.files]
    if "teacher_forces" not in data.files or not students:
        print(f"[plot_parity] skipped: no teacher/student forces in {folder}")
        return []
    label = lambda n: rename.get(n, "GRACE-OFF" if n == "teacher" else n)
    written = []
    teacher = data["teacher_forces"]
    panels = [(f"{label(n)}: {_stats(data[f'{n}_forces'], teacher)}", data[f"{n}_forces"])
              for n in students]
    written.append(grid(out_dir / f"{prefix}parity_forces.png", teacher, panels, "GRACE-OFF"))
    if "dft_forces" in data.files:
        dft = data["dft_forces"]
        panels = [(f"{label(n)}: {_stats(data[f'{n}_forces'], dft)}", data[f"{n}_forces"])
                  for n in [*students, "teacher"]]
        written.append(grid(out_dir / f"{prefix}parity_forces_vs_dft.png", dft, panels, "DFT"))
    return written


def make_materials(folder, out_path=None, models=None, rename=None):
    """Grid parity plot (groups A/B/C x models) from a validate_materials.py folder."""
    folder = Path(folder).expanduser()
    rename = rename or {}
    if not (folder / "predictions.npz").exists():
        print(f"[plot_parity] skipped: no predictions.npz in {folder}")
        return None
    data = np.load(folder / "predictions.npz")
    keys = [tuple(k.split("__")) for k in data.files if k.count("__") == 2]
    available = list(dict.fromkeys(m for m, _, _ in keys))
    models = [m for m in (models or available) if m in available]
    groups = [g for g in "ABC" if any(g == gg for _, gg, _ in keys)]
    if not models or not groups:
        print(f"[plot_parity] skipped: nothing to plot in {folder}")
        return None
    plt = _plt()
    fig, axes = plt.subplots(len(groups), len(models), figsize=(4.0 * len(models), 3.9 * len(groups)),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for row, g in enumerate(groups):
        ref_any = next((data[f"{m}__{g}__dft_f"] for m in models if f"{m}__{g}__dft_f" in data.files),
                       None)
        lim = 1.1 * np.percentile(np.abs(ref_any), 99.5) if ref_any is not None else 1.0
        for col, m in enumerate(models):
            ax = axes[row][col]
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            name = rename.get(m, m)
            if f"{m}__{g}__fp" not in data.files:
                ax.set_facecolor("#f0f0f0")
                ax.text(0, 0, "cannot run\n(elements not supported)", ha="center", va="center",
                        color="tab:red", fontsize=9)
                ax.set_title(f"{name} | group {g}", fontsize=9)
            else:
                ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
                ref, pred = data[f"{m}__{g}__dft_f"].ravel(), data[f"{m}__{g}__fp"].ravel()
                sel = rng.choice(len(ref), size=min(len(ref), 20000), replace=False)
                ax.scatter(ref[sel], np.clip(pred[sel], -lim, lim), s=2, alpha=0.25, rasterized=True)
                ax.set_title(f"{name} | group {g}\n{_stats(pred, ref)}", fontsize=9)
            ax.grid(alpha=0.25)
            if row == len(groups) - 1:
                ax.set_xlabel("DFT force component (eV/Å)", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{GROUP_LABEL[g]}\nmodel force (eV/Å)", fontsize=9)
    fig.suptitle("Inorganic materials (MPtrj DFT): force parity per group", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path).expanduser() if out_path else folder / "materials_parity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="validation output folder (with predictions.npz)")
    p.add_argument("--materials", action="store_true", help="folder is a validate_materials.py output")
    p.add_argument("--models", nargs="*", help="materials: models (columns) to include")
    p.add_argument("--out", help="output folder (organic) or PNG path (materials)")
    args = p.parse_args()
    if args.materials:
        path = make_materials(args.folder, args.out, args.models)
        print(f"Saved: {path}" if path else "Nothing written")
    else:
        paths = make(args.folder, args.out)
        print("Saved: " + ", ".join(map(str, paths)) if paths else "Nothing written")


if __name__ == "__main__":
    main()
