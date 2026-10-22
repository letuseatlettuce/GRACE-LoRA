"""Pareto fronts and the organic/materials trade-off plot.

Importable:
    make(points, out_path, teacher_f=None, title=None, ylabel=...) -> path or None
        points: [{"name": str, "params": int|None, "effort": float|None,
                  "force": float, "energy": float|None}]
        Pareto panels: force MAE (y, lower better) vs trainable parameters, training effort
        (structures x epochs) and, if available, energy MAE. Top row: full range; bottom row:
        zoom on the trained models (everything except "base").
    make_tradeoff(experiments, base, out_path, max_materials_increase=0.15, teacher_f=None)
        experiments: [{"id", "parent", "organic_f", "materials_f", "feasible"}] (ledger entries)
        base: {"organic_f", "materials_f"}; shaded region = materials within the budget
    make_from_validation(folder, out_path=None) -> path or None
        legacy: points from a validate_model.py folder + run metadata next to each SavedModel

CLI:
    python scripts/plot_pareto.py validation                # legacy validation-folder mode

Only numpy/matplotlib/json/pandas: no TensorFlow. Missing inputs are skipped with a note.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def pareto_mask(x, y):
    """True for points not dominated (lower-or-equal in both, strictly lower in one)."""
    pts = np.column_stack([x, y])
    return np.array([not np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)) for p in pts])


def run_info(run_dir: Path) -> dict:
    """{'params': int|None, 'effort': float|None} from a train_lora.py run folder (or run.json)."""
    run_dir = Path(run_dir)
    if (run_dir / "run.json").exists():
        meta = json.loads((run_dir / "run.json").read_text())
        return {"params": meta.get("trainable_parameters"),
                "effort": meta.get("n_train", 0) * meta.get("epochs", 0)}
    info = {"params": None, "effort": None}
    if (run_dir / "summary.json").exists():
        info["params"] = json.loads((run_dir / "summary.json").read_text()).get("trainable_params")
    if (run_dir / "args.json").exists():
        args = json.loads((run_dir / "args.json").read_text())
        train = Path(args.get("data_dir", "")) / "train.pkl.gz"
        if train.exists():
            try:
                import config
                info["effort"] = len(config.load_frame(train)) * args.get("epochs", 0)
            except (Exception, SystemExit) as exc:  # unreadable or untrusted pickle: effort unknown
                print(f"[plot_pareto] note: cannot read {train}: {exc}")
    return info


def _label_points(ax, xs, ys, names, front):
    """Annotate points to their right (clear of the step line, which runs left/up)."""
    for i, n in enumerate(names):
        ax.annotate(f"{n}: {ys[i]:.1f}" + ("" if front[i] else " (dominated)"), (xs[i], ys[i]),
                    textcoords="offset points", xytext=(11, -3), ha="left", va="center",
                    fontsize=8.5, bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none",
                                        "alpha": 0.8})


def make(points, out_path, teacher_f=None, title=None, ylabel="force MAE vs DFT (meV/Å)"):
    points = [p for p in points if p.get("force") is not None]
    panels = [("params", "trainable parameters (millions)", 1e-6),
              ("effort", "training effort (structures × epochs, thousands)", 1e-3),
              ("energy", "energy MAE vs DFT (meV/atom)", 1.0)]
    panels = [pn for pn in panels if sum(p.get(pn[0]) is not None for p in points) >= 2]
    if len(points) < 2 or not panels:
        print("[plot_pareto] skipped: need >= 2 points with metadata")
        return None
    plt = _plt()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = {p["name"]: colors[i % len(colors)] for i, p in enumerate(points)}
    fig, axes = plt.subplots(2, len(panels), figsize=(5.4 * len(panels), 9), squeeze=False)
    for col, (key, xlabel, scale) in enumerate(panels):
        pts = [p for p in points if p.get(key) is not None]
        x = np.array([p[key] for p in pts], dtype=float) * scale
        y = np.array([p["force"] for p in pts], dtype=float)
        names = [p["name"] for p in pts]
        front = pareto_mask(x, y)
        order = np.argsort(x[front])
        for row, ax in enumerate(axes[:, col]):
            ax.step(x[front][order], y[front][order], where="post", color="0.4", lw=1.3, zorder=1,
                    label="Pareto front")
            for i, n in enumerate(names):
                ax.scatter(x[i], y[i], s=140 if front[i] else 100, color=color[n],
                           edgecolor="black" if front[i] else "white", linewidth=1.3,
                           zorder=3 if front[i] else 4, marker="o" if front[i] else "X", label=n)
            ax.grid(alpha=0.3)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                if teacher_f is not None:
                    ax.axhline(teacher_f, color="tab:red", ls="--", lw=1, label="teacher")
                ax.set_ylim(0, y.max() * 1.15)
            else:
                trained = np.array([n != "base" for n in names])
                if trained.sum() >= 1:
                    tx, ty = x[trained], y[trained]
                    pad_y = max(np.ptp(ty), 2.0) * 0.6
                    pad_x = max(np.ptp(tx), 0.1 * max(abs(tx).max(), 1e-9)) * 0.4
                    ax.set_ylim(ty.min() - pad_y, ty.max() + pad_y)
                    ax.set_xlim(tx.min() - pad_x, tx.max() + 2.2 * pad_x)  # room for labels
                    sel = np.flatnonzero(trained)
                    _label_points(ax, tx, ty, [names[i] for i in sel], front[sel])
                ax.set_title("zoom: trained models", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=min(len(unique), 7),
               frameon=False, fontsize=9)
    fig.suptitle(title or "Pareto fronts: lower-left is better (outlined = Pareto-optimal, X = dominated)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _place_labels(ax, pts, x_span, y_span):
    """Label points, pushing labels of crowded neighbours to alternating sides."""
    order = sorted(pts, key=lambda p: (p[1], p[2]))
    for k, (name, x, y) in enumerate(order):
        crowded = any(abs(x - x2) < 0.06 * x_span and abs(y - y2) < 0.06 * y_span
                      for n2, x2, y2 in order if n2 != name)
        if crowded:
            rank = sum(1 for n2, x2, y2 in order if n2 != name and abs(x - x2) < 0.06 * x_span
                       and abs(y - y2) < 0.06 * y_span and (x2, y2) < (x, y))
            offset, ha = ((10, 10), "left") if rank % 2 else ((-10, -14), "right")
        else:
            offset, ha = (9, 7), "left"
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offset, ha=ha, fontsize=9,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                    zorder=6)


def make_tradeoff(experiments, base, out_path, max_materials_increase=0.15, teacher_f=None,
                  best=None):
    exps = [e for e in experiments
            if e.get("organic_f") is not None and e.get("materials_f") is not None]
    if not exps or not base or base.get("materials_f") is None:
        print("[plot_pareto] skipped trade-off: no experiments with monitor metrics")
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    limit = base["materials_f"] * (1 + max_materials_increase)
    xs = [e["organic_f"] for e in exps] + [base["organic_f"]] + ([teacher_f] if teacher_f else [])
    ys = [e["materials_f"] for e in exps] + [base["materials_f"]]
    x_lo, x_hi = 0, max(xs) * 1.1
    y_lo, y_hi = min(ys) * 0.85, max(ys) * 1.12
    ax.axhspan(y_lo, limit, color="tab:green", alpha=0.08, zorder=0)
    ax.axhline(limit, color="tab:green", ls="--", lw=1)
    ax.text(x_hi * 0.99, limit, f"materials budget (+{100 * max_materials_increase:.0f}% vs base)",
            ha="right", va="bottom", fontsize=8.5, color="tab:green")
    pos = {e["id"]: (e["organic_f"], e["materials_f"]) for e in exps}
    pos["base"] = (base["organic_f"], base["materials_f"])
    for e in exps:  # search path: parent -> child (first experiments start from base)
        src = pos.get(e.get("parent") or "base")
        if src:
            ax.annotate("", xy=pos[e["id"]], xytext=src,
                        arrowprops={"arrowstyle": "->", "color": "0.55", "lw": 1.1,
                                    "shrinkA": 7, "shrinkB": 7})
    ax.scatter(*pos["base"], s=170, marker="s", color="0.3", zorder=4, label="base GRACE")
    for e in exps:
        ok = bool(e.get("feasible"))
        ax.scatter(*pos[e["id"]], s=130, zorder=5, color="tab:green" if ok else "tab:orange",
                   edgecolor="black", marker="o" if ok else "X")
    if best in pos:
        ax.scatter(*pos[best], s=420, facecolor="none", edgecolor="black", lw=1.6, zorder=6,
                   label=f"best ({best})")
    _place_labels(ax, [(n, x, y) for n, (x, y) in pos.items()], x_hi - x_lo, y_hi - y_lo)
    if teacher_f:
        ax.axvline(teacher_f, color="tab:red", ls=":", lw=1)
        ax.text(teacher_f, y_hi, " teacher\n (organics)", color="tab:red", va="top", fontsize=8.5)
    ax.scatter([], [], s=110, color="tab:green", edgecolor="black", label="feasible")
    ax.scatter([], [], s=110, color="tab:orange", edgecolor="black", marker="X", label="infeasible")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("organic force MAE, monitor set (meV/Å)  ← better")
    ax.set_ylabel("materials force MAE, monitor set (meV/Å)  ← better")
    ax.set_title("Search path: organic gain vs materials retention (arrows = parent → child)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_from_validation(folder, out_path=None):
    """Legacy mode: points from a validate_model.py folder and the run folders it names."""
    folder = Path(folder).expanduser()
    path = folder / "validation.json"
    if not path.exists():
        print(f"[plot_pareto] skipped: missing {path}")
        return None
    val = json.loads(path.read_text())
    m = val["metrics"]
    if "vs_dft" not in m.get("base", {}):
        print("[plot_pareto] skipped: validation.json has no DFT metrics")
        return None
    points = [{"name": "base", "params": 0, "effort": 0,
               "force": m["base"]["vs_dft"]["force_mae_meV_A"],
               "energy": m["base"]["vs_dft"].get("energy_mae_meV_atom")}]
    for name, model_dir in val.get("trained", {}).items():
        info = run_info(Path(model_dir).parent)
        points.append({"name": name, **info, "force": m[name]["vs_dft"]["force_mae_meV_A"],
                       "energy": m[name]["vs_dft"].get("energy_mae_meV_atom")})
    teacher = m.get("teacher", {}).get("vs_dft", {}).get("force_mae_meV_A")
    return make(points, out_path or folder / "pareto_front.png", teacher_f=teacher)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="validation output folder (validation.json)")
    p.add_argument("--out", default=None, help="image path (default: <folder>/pareto_front.png)")
    args = p.parse_args()
    path = make_from_validation(args.folder, args.out)
    print(f"Saved: {path}" if path else "Nothing written")


if __name__ == "__main__":
    main()
