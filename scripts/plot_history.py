"""Training-history plots for several runs.

Reads two formats:
  --csv  name=path/history.csv   train_lora.py output (epoch, train_loss, e_mae_meV_atom,
                                 f_mae_meV_A, time_s; epoch 0 = untrained model)
  --log  name=path/history.txt   text log lines "epoch 001: train=.. val=.. val_force=.. meV/A"
Checkpoints taken from a logged run are marked with --mark name=epoch:label (repeatable).
With --validation <folder>, each model's held-out test force MAE vs DFT is drawn at the right.

Importable:
    read_csv(path) / read_log(path) -> run dict (or None if unreadable)
    make(runs, out_path, marks=None, test=None) -> path or None
        runs: {name: run dict}; marks: {name: [(epoch, label)]}; test: {model: test force MAE}
    make_monitor_curves(histories, out_path, base=None) -> path or None
        histories: {name: history.csv path}; three panels (organic / materials / only_adapted
        monitor force MAE per epoch); runs without monitor columns are noted and skipped

    python scripts/plot_history.py --csv "smoke test=runs/lora_r8/history.csv" \\
        --log "C10/C11 run=checkpoint_history.txt" --mark "C10/C11 run=10:C10" \\
        --mark "C10/C11 run=11:C11" --validation validation --out validation/training_history.png
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

LOG_LINE = re.compile(r"epoch\s+(\d+):\s*train=([\d.eE+-]+)\s+val=([\d.eE+-]+)\s+"
                      r"val_force=([\d.eE+-]+)")


MONITORS = [("organic", "organic molecules (monitor)"),
            ("materials", "materials with adapted elements (monitor)"),
            ("only_adapted", "crystals of only adapted elements (monitor)")]


def _num(v):
    try:
        return float(v) if v not in ("", "nan", None) else np.nan
    except ValueError:
        return np.nan


def read_csv(path):
    """train_lora.py history.csv -> run dict, incl. monitors {name: force array} (None if missing)."""
    path = Path(path)
    if not path.exists():
        print(f"[plot_history] note: missing {path}")
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[plot_history] note: empty {path}")
        return None
    cols = rows[0].keys()
    monitors = {c[:-len("_f_mae_meV_A")]: np.array([_num(r[c]) for r in rows])
                for c in cols if c.endswith("_f_mae_meV_A") and c != "f_mae_meV_A"}
    return {"epoch": np.array([int(float(r["epoch"])) for r in rows]),
            "train_loss": np.array([_num(r.get("train_loss")) for r in rows]),
            "val_loss": None,
            "val_force": np.array([_num(r.get("f_mae_meV_A")) for r in rows]),
            "val_energy": np.array([_num(r.get("e_mae_meV_atom")) for r in rows]),
            "monitors": monitors}


def read_log(path):
    rows = [LOG_LINE.search(line) for line in Path(path).read_text().splitlines()]
    rows = [m.groups() for m in rows if m]
    if not rows:
        print(f"[plot_history] note: no 'epoch NNN: train=.. val=.. val_force=..' lines in {path}")
        return None
    arr = np.array(rows, dtype=float)
    return {"epoch": arr[:, 0].astype(int), "train_loss": arr[:, 1], "val_loss": arr[:, 2],
            "val_force": arr[:, 3], "val_energy": None, "monitors": {}}


def make_monitor_curves(histories, out_path, base=None):
    """Monitor force MAE per epoch for every experiment, one panel per monitor set."""
    runs = {n: read_csv(p) for n, p in histories.items()}
    runs = {n: r for n, r in runs.items() if r and r["monitors"]}
    skipped = [n for n in histories if n not in runs]
    if skipped:
        print(f"[plot_history] note: no monitor columns for {', '.join(skipped)}")
    present = [(k, t) for k, t in MONITORS if any(k in r["monitors"] for r in runs.values())]
    if not runs or not present:
        print("[plot_history] skipped monitor curves: no monitor data")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, len(present), figsize=(5.6 * len(present), 4.8), squeeze=False)
    for ax, (key, title) in zip(axes[0], present):
        for k, (name, r) in enumerate(runs.items()):
            if key in r["monitors"]:
                ax.plot(r["epoch"], r["monitors"][key], "-o", ms=3.5, color=colors[k % len(colors)],
                        label=name)
        ref = (base or {}).get(f"{key}_f")
        if ref is None:  # epoch 0 of any run = untrained base
            ref = next((r["monitors"][key][0] for r in runs.values() if key in r["monitors"]), None)
        if ref is not None and np.isfinite(ref):
            ax.axhline(ref, color="0.35", ls=":", lw=1)
            ax.text(ax.get_xlim()[1], ref, "base ", ha="right", va="bottom", fontsize=8.5, color="0.35")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("force MAE (meV/Å)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    fig.suptitle("Held-out monitor sets during training (epoch 0 = untrained base GRACE)", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def split_spec(spec):
    name, _, value = spec.partition("=")
    if not value:
        raise SystemExit(f"Expected name=value, got '{spec}'")
    return name, value


def make(runs, out_path, marks=None, test=None):
    """Validation force / loss / learning-speed panels for several runs."""
    runs = {n: r for n, r in runs.items() if r is not None and len(r["epoch"])}
    if not runs:
        print("[plot_history] skipped: no runs")
        return None
    marks = marks or {}
    test = test or {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax_f, ax_l, ax_r) = plt.subplots(1, 3, figsize=(17, 5.4))
    max_epoch = max(r["epoch"].max() for r in runs.values())
    diamonds = {}  # run -> (color, [(label, held-out test MAE)])
    for k, (name, r) in enumerate(runs.items()):
        c = colors[k % len(colors)]
        ax_f.plot(r["epoch"], r["val_force"], "-o", color=c, ms=4, label=f"{name} (validation)")
        for epoch, label in marks.get(name, []):
            i = np.flatnonzero(r["epoch"] == epoch)
            if i.size:
                y = r["val_force"][i[0]]
                ax_f.scatter(epoch, y, marker="*", s=260, color=c, edgecolor="black", zorder=5)
                n_before = sum(1 for e, _ in marks[name] if e < epoch)
                ax_f.annotate(f"{label}\nval {y:.1f}", (epoch, y), textcoords="offset points",
                              xytext=(-62 + 58 * n_before, 16), fontsize=8.5,
                              arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.6})
                if label in test:
                    diamonds.setdefault(name, (c, []))[1].append((label, test[label]))
        if name in test:
            diamonds.setdefault(name, (c, []))[1].append((name, test[name]))

        trained = r["epoch"] >= 1
        ax_l.semilogy(r["epoch"][trained], r["train_loss"][trained], "-o", color=c, ms=4,
                      label=f"{name}: train loss")
        if r["val_loss"] is not None:
            ax_l.semilogy(r["epoch"], r["val_loss"], "--s", color=c, ms=4, mfc="none",
                          label=f"{name}: val loss")

        ones = np.flatnonzero(r["epoch"] == 1)
        if ones.size:
            first = r["val_force"][ones[0]]
            ax_r.plot(r["epoch"][trained], 100 * r["val_force"][trained] / first, "-o", color=c,
                      ms=4, label=name)

    if test:
        base = test.get("base")
        teacher = test.get("teacher")
        x = max_epoch + 1.2
        ax_f.axvline(max_epoch + 0.6, color="0.6", lw=0.8)
        # one diamond per model; one label per run, stacked so close values stay readable
        ordered = sorted(diamonds.values(), key=lambda cv: -np.mean([v for _, v in cv[1]]))
        for k, (c, items) in enumerate(ordered):
            for _, v in items:
                ax_f.scatter(x, v, marker="D", s=55, color=c, edgecolor="black", zorder=5)
            text = "\n".join(f"{lab}: {v:.1f}" for lab, v in items)
            ax_f.annotate(text, (x, np.mean([v for _, v in items])), textcoords="offset points",
                          xytext=(10, 6 if k == 0 else -12 * len(items) - 8), fontsize=8,
                          color=c, va="bottom")
        top = max(np.nanmax(r["val_force"]) for r in runs.values()) * 1.08
        ax_f.text(x, top, "held-out\ntest", ha="center", va="top", fontsize=8.5)
        ax_f.set_xlim(-0.5, max_epoch + 5.2)
        ax_f.set_xticks(range(0, max_epoch + 1, 2))
        if base is not None:
            ax_f.axhline(base, color="tab:gray", ls=":", lw=1)
            ax_f.text(max_epoch + 0.4, base, f"base GRACE (test) {base:.0f}", va="bottom",
                      ha="right", fontsize=8.5, color="tab:gray")
        if teacher is not None:
            ax_f.axhline(teacher, color="tab:red", ls="--", lw=1)
            ax_f.text(0.2, teacher, f"GRACE-OFF (test) {teacher:.0f}", va="bottom", fontsize=8.5,
                      color="tab:red")
        ax_f.set_ylim(0, top * 1.02)
    ax_f.set_xlabel("epoch")
    ax_f.set_ylabel("force MAE (meV/Å)")
    ax_f.set_title("validation force MAE per epoch" + (" (◆ = held-out test MAE vs DFT)" if test else ""),
                   fontsize=10)
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=8.5, loc="upper right" if not test else "lower left",
                bbox_to_anchor=None if not test else (0.0, 0.1))

    ax_l.set_xlabel("epoch")
    ax_l.set_ylabel("loss (log scale)")
    ax_l.set_title("training / validation loss (loss definitions differ between runs)",
                   fontsize=10)
    ax_l.grid(alpha=0.3, which="both")
    ax_l.legend(fontsize=8.5)

    ax_r.set_xlabel("epoch")
    ax_r.set_ylabel("validation force MAE, % of epoch 1")
    ax_r.set_title("learning speed (normalised to each run's epoch 1)", fontsize=10)
    ax_r.grid(alpha=0.3)
    ax_r.legend(fontsize=8.5)

    fig.tight_layout()
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="*", default=[], help="name=history.csv (train_lora.py)")
    p.add_argument("--log", nargs="*", default=[], help="name=history.txt (text log)")
    p.add_argument("--mark", action="append", default=[], help="name=epoch:label checkpoint marker")
    p.add_argument("--validation", help="validation folder (adds held-out test force MAE)")
    p.add_argument("--monitors", help="also write monitor curves (organic/materials/only_adapted) here")
    p.add_argument("--out", default="training_history.png")
    args = p.parse_args()

    runs = {}
    for spec in args.csv:
        name, path = split_spec(spec)
        runs[name] = read_csv(Path(path).expanduser())
    for spec in args.log:
        name, path = split_spec(spec)
        runs[name] = read_log(Path(path).expanduser())
    marks = {}
    for spec in args.mark:
        name, rest = split_spec(spec)
        epoch, _, label = rest.partition(":")
        marks.setdefault(name, []).append((int(epoch), label or f"epoch {epoch}"))
    test = {}
    if args.validation:
        val = json.loads((Path(args.validation).expanduser() / "validation.json").read_text())
        test = {n: m["vs_dft"]["force_mae_meV_A"] for n, m in val["metrics"].items() if "vs_dft" in m}
    out = make(runs, args.out, marks, test)
    for name, r in runs.items():
        if r is None:
            continue
        best = int(np.nanargmin(r["val_force"]))
        print(f"{name:14s} epochs {r['epoch'].min()}-{r['epoch'].max()}  val force "
              f"{r['val_force'][r['epoch'] >= 1][0]:.1f} -> {r['val_force'][-1]:.1f} meV/A "
              f"(best {r['val_force'][best]:.1f} at epoch {r['epoch'][best]})")
    print(f"Saved: {out}" if out else "Nothing written")
    if args.monitors:
        csvs = dict(split_spec(s) for s in args.csv)
        m = make_monitor_curves(csvs, args.monitors)
        print(f"Saved: {m}" if m else "No monitor curves written")


if __name__ == "__main__":
    main()
