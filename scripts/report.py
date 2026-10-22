"""Final report for a GRACE LoRA adaptation campaign: report.md + figures.

Reads (layout per docs/CONTRACTS.md):
    <workdir>/experiments/ledger.json             experiment ledger (configs, rationales, monitors)
    <workdir>/experiments/<id>/run/history.csv    per-epoch monitors of each experiment
    <workdir>/validation/base/{organic,materials} base GRACE + teacher reference validation
    <workdir>/validation/<id>/{organic,materials} full validation of chosen experiments
Writes:
    <out>/report.md   summary, timeline, decision log, figures with captions, limitations
    <out>/figures/*.png

Usage:
    python scripts/report.py --workdir W [--exps exp_003 exp_005] [--out W/report]
    python scripts/report.py --validation-only <out> [--title TEXT]

Validation-only mode (no ledger): <out>/organic (validate_model.py output) and/or
<out>/materials (validate_materials.py output) -> <out>/summary.md + <out>/figures/
(heatmap, organic parity vs DFT and vs teacher, materials parity grid, results table and an
automatic verdict per model). Ledger mode also links any <workdir>/validation/adhoc_* folders.

Only numpy/pandas/matplotlib/json: no TensorFlow. Missing inputs are skipped with a note.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_heatmap  # noqa: E402
import plot_history  # noqa: E402
import plot_parity  # noqa: E402
import plot_pareto  # noqa: E402

DEFAULT_OBJECTIVE = {"primary": "organic_f", "max_materials_increase": 0.15,
                     "max_only_adapted_increase": 0.75}
REFERENCE_NAMES = {"base", "teacher", "GRACE-OFF"}
CONFIG_KEYS = ["n_organic", "n_materials", "n_only_adapted", "rank", "alpha", "epochs", "lr",
               "batch_size", "oversample", "max_atoms", "seed", "adapt_elements"]


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[report] note: cannot parse {path}: {exc}")
        return None


def relative(metrics, base):
    out = {}
    if metrics.get("organic_f") is not None and base.get("organic_f"):
        out["organic_gain"] = 1 - metrics["organic_f"] / base["organic_f"]
    for key in ("materials", "only_adapted"):
        if metrics.get(f"{key}_f") is not None and base.get(f"{key}_f"):
            out[f"{key}_increase"] = metrics[f"{key}_f"] / base[f"{key}_f"] - 1
    return out


def is_feasible(entry, base, objective):
    if "feasible" in entry and entry["feasible"] is not None:
        return bool(entry["feasible"])
    rel = entry.get("relative") or relative(entry.get("metrics") or {}, base)
    if "materials_increase" not in rel:
        return False
    ok = rel["materials_increase"] <= objective["max_materials_increase"]
    if "only_adapted_increase" in rel:
        ok = ok and rel["only_adapted_increase"] <= objective["max_only_adapted_increase"]
    return ok


def pick_best(done, base, objective):
    """Feasible experiment with the lowest score (organic_f); else lowest materials increase."""
    scored = [e for e in done if (e.get("metrics") or {}).get("organic_f") is not None]
    feas = [e for e in scored if is_feasible(e, base, objective)]
    if feas:
        return min(feas, key=lambda e: e.get("score", e["metrics"]["organic_f"]))
    rel = lambda e: (e.get("relative") or relative(e["metrics"], base)).get("materials_increase", np.inf)
    return min(scored, key=rel) if scored else None


def validation_of(workdir, exp_id):
    """{'organic': json, 'materials': json, 'dir': Path, 'alias': {model_name: exp_id}}."""
    d = Path(workdir) / "validation" / exp_id
    org = load_json(d / "organic" / "validation.json")
    mat = load_json(d / "materials" / "materials_validation.json")
    names = set((org or {}).get("metrics", {})) | set((mat or {}).get("results", {}))
    alias = {n: exp_id for n in names if n not in REFERENCE_NAMES} if exp_id != "base" else {}
    return {"organic": org, "materials": mat, "dir": d, "alias": alias}


def organic_dft(val, name):
    m = ((val or {}).get("metrics") or {}).get(name, {})
    return m.get("vs_dft")


def all_materials(groups, sizes):
    if not sizes or not all((groups.get(k) or {}).get("n") for k in "ABC"):
        return None
    w = np.array([sizes[k] for k in "ABC"], dtype=float)
    return float(np.dot(w / w.sum(), [groups[k]["force_mae_meV_A"] for k in "ABC"]))


# --------------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------------
def fmt(v, digits=1, unit=""):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}{unit}"


def pct(new, old):
    if new is None or old in (None, 0):
        return "–"
    return f"{100 * (new / old - 1):+.0f}%"


def duration(seconds):
    if not seconds:
        return "–"
    m = int(round(seconds / 60))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def config_summary(cfg):
    parts = [f"{cfg.get('n_organic', '?')} organic"]
    if cfg.get("n_materials"):
        only = f" ({cfg['n_only_adapted']} only-adapted)" if cfg.get("n_only_adapted") else ""
        parts.append(f"{cfg['n_materials']} materials{only}")
    else:
        parts.append("no replay")
    parts.append(f"rank {cfg.get('rank', '?')}, {cfg.get('epochs', '?')} epochs")
    return ", ".join(parts)


def config_delta(cfg, parent_cfg):
    if parent_cfg is None:
        return config_summary(cfg)
    changes = [f"{k} {parent_cfg.get(k)}→{cfg.get(k)}" for k in CONFIG_KEYS
               if k in cfg or k in parent_cfg if cfg.get(k) != parent_cfg.get(k)]
    return "; ".join(changes) if changes else "same config"


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------
def build(workdir, out, exps=None):
    workdir, out = Path(workdir).expanduser(), Path(out).expanduser()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    ledger = load_json(workdir / "experiments" / "ledger.json")
    lines = ["# GRACE LoRA adaptation report", "",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M')} from `{workdir}`._", ""]
    if not ledger or not ledger.get("experiments"):
        print("[report] note: no ledger / no experiments; writing a minimal report")
        lines += ["No experiments found in `experiments/ledger.json` yet.", ""]
        lines += additional_validations(workdir, out)
        (out / "report.md").write_text("\n".join(lines))
        return out / "report.md"

    objective = {**DEFAULT_OBJECTIVE, **(ledger.get("objective") or {})}
    base = dict(ledger.get("base") or {})
    entries = ledger["experiments"]
    if exps:
        entries = [e for e in entries if e["id"] in exps]
    done = [e for e in entries if e.get("status") == "done"]
    by_id = {e["id"]: e for e in ledger["experiments"]}
    best = pick_best(done, base, objective)

    # validation: base/teacher reference + every experiment that was validated
    vbase = validation_of(workdir, "base")
    vexp = {e["id"]: validation_of(workdir, e["id"]) for e in done}
    validated = [i for i, v in vexp.items() if v["organic"] or v["materials"]]
    # base/teacher reference: validation/base, else the base rows inside any experiment folder
    ref_org = vbase["organic"] or next((vexp[i]["organic"] for i in validated if vexp[i]["organic"]), None)
    ref_mat = vbase["materials"] or next((vexp[i]["materials"] for i in validated if vexp[i]["materials"]), None)
    sizes = (ref_mat or {}).get("group_sizes_available")
    base_dft, teacher_dft = organic_dft(ref_org, "base"), organic_dft(ref_org, "teacher")
    base_groups = ((ref_mat or {}).get("results") or {}).get("base", {}).get("groups", {})

    # ---- summary --------------------------------------------------------------------------
    lines += ["## Summary", ""]
    if best is None:
        lines += ["No finished experiment has monitor metrics yet.", ""]
    else:
        bm = best["metrics"]
        feasible = is_feasible(best, base, objective)
        lines.append(f"**Best experiment: `{best['id']}`** ({config_summary(best.get('config', {}))})"
                     + ("" if feasible else " (no experiment met the materials budget; this one "
                        "degrades materials least)") + ".")
        lines.append("")
        rows = ["| Metric (force MAE, meV/Å) | Base GRACE | Best | Change | Teacher |", "|---|---|---|---|---|"]
        vb = vexp.get(best["id"], {})
        best_dft = organic_dft(vb.get("organic"), next(iter(vb.get("alias", {})), best["id"]))
        if best_dft and base_dft:
            rows.append(f"| Organic molecules vs DFT (full test set) | {fmt(base_dft['force_mae_meV_A'])} | "
                        f"**{fmt(best_dft['force_mae_meV_A'])}** | "
                        f"{pct(best_dft['force_mae_meV_A'], base_dft['force_mae_meV_A'])} | "
                        f"{fmt((teacher_dft or {}).get('force_mae_meV_A'))} |")
        best_groups = {}
        if vb.get("materials"):
            name = next(iter(vb["alias"]), best["id"])
            best_groups = vb["materials"]["results"].get(name, {}).get("groups", {})
            b_all, x_all = all_materials(base_groups, sizes), all_materials(best_groups, sizes)
            if b_all and x_all:
                rows.append(f"| All materials vs DFT (A+B+C weighted) | {fmt(b_all)} | **{fmt(x_all)}** | "
                            f"{pct(x_all, b_all)} | cannot run |")
            for g in "BC":
                if (best_groups.get(g) or {}).get("n") and (base_groups.get(g) or {}).get("n"):
                    b, x = base_groups[g]["force_mae_meV_A"], best_groups[g]["force_mae_meV_A"]
                    rows.append(f"| Materials group {g} vs DFT | {fmt(b)} | {fmt(x)} | {pct(x, b)} | – |")
        rows.append(f"| Organic monitor (held-out, vs teacher labels) | {fmt(base.get('organic_f'))} | "
                    f"{fmt(bm.get('organic_f'))} | {pct(bm.get('organic_f'), base.get('organic_f'))} | – |")
        rows.append(f"| Materials monitor (held-out, DFT) | {fmt(base.get('materials_f'))} | "
                    f"{fmt(bm.get('materials_f'))} | {pct(bm.get('materials_f'), base.get('materials_f'))} | – |")
        if bm.get("only_adapted_f") is not None:
            rows.append(f"| Only-adapted crystals monitor (DFT) | {fmt(base.get('only_adapted_f'))} | "
                        f"{fmt(bm.get('only_adapted_f'))} | "
                        f"{pct(bm.get('only_adapted_f'), base.get('only_adapted_f'))} | – |")
        lines += rows + [""]
        run = workdir / best.get("run_dir", "")
        summary = load_json(run / "summary.json") or {}
        if summary.get("trainable_params"):
            lines.append(f"Adapter: {summary['trainable_params']:,} trainable parameters "
                         f"({100 * summary['trainable_params'] / summary['total_params']:.1f}% of "
                         f"{summary['total_params']:,}); training time {duration(best.get('duration_s') or summary.get('train_time_s'))}.")
            lines.append("")
    lines += [f"Objective: minimise organic force error while keeping materials within "
              f"+{100 * objective['max_materials_increase']:.0f}% and only-adapted crystals within "
              f"+{100 * objective['max_only_adapted_increase']:.0f}% of base GRACE (monitor sets). "
              f"{len(done)} experiment(s) finished, "
              f"{sum(is_feasible(e, base, objective) for e in done)} feasible.", ""]

    # ---- timeline ---------------------------------------------------------------------------
    lines += ["## Experiment timeline", "",
              "| Experiment | Parent | Change vs parent | Organic | Materials | Only-adapted | Feasible | Time |",
              "|---|---|---|---|---|---|---|---|",
              f"| base | – | untrained GRACE | {fmt(base.get('organic_f'))} | {fmt(base.get('materials_f'))} | "
              f"{fmt(base.get('only_adapted_f'))} | – | – |"]
    for e in entries:
        m = e.get("metrics") or {}
        parent_cfg = (by_id.get(e.get("parent")) or {}).get("config") if e.get("parent") else None
        status = "" if e.get("status") == "done" else f" ({e.get('status', '?')})"
        star = "**" if best and e["id"] == best["id"] else ""
        feas = "✔" if is_feasible(e, base, objective) else "✘"
        lines.append(f"| {star}{e['id']}{star}{status} | {e.get('parent') or '–'} | "
                     f"{config_delta(e.get('config', {}), parent_cfg)} | {fmt(m.get('organic_f'))} | "
                     f"{fmt(m.get('materials_f'))} | {fmt(m.get('only_adapted_f'))} | {feas} | "
                     f"{duration(e.get('duration_s'))} |")
    lines += ["", "Monitor force MAE in meV/Å at the selected (best-validation) epoch. Organic monitor "
              "uses teacher labels; materials monitors use DFT.", ""]

    # ---- decision log -----------------------------------------------------------------------
    lines += ["## Decision log", ""]
    for k, e in enumerate(entries, 1):
        m = e.get("metrics") or {}
        outcome = (f"→ organic {fmt(m.get('organic_f'))}, materials {fmt(m.get('materials_f'))}"
                   + (f", only-adapted {fmt(m.get('only_adapted_f'))}" if m.get("only_adapted_f") is not None else "")
                   + f" meV/Å; {'feasible' if is_feasible(e, base, objective) else 'infeasible'}")
        if e.get("status") != "done":
            outcome = f"→ {e.get('status', 'unknown')}" + (f": {e['error']}" if e.get("error") else "")
        lines.append(f"{k}. **{e['id']}**" + (f" (from {e['parent']})" if e.get("parent") else "")
                     + f": {e.get('rationale') or '_no rationale recorded_'}  ")
        lines.append(f"   {outcome}")
        if e.get("hypothesis"):
            lines.append(f"   _Hypothesis:_ {e['hypothesis']}")
        for r in e.get("review") or []:
            lines.append(f"   > **Critic ({r.get('verdict', 'note')}):** {r.get('text', '')}")
        if e.get("healing"):
            lines.append(f"   _Self-healed {len(e['healing'])}×: "
                         + "; ".join(h.get("error_class", "error") for h in e["healing"]) + "_")
    lines.append("")
    lines += predictions_section(entries) + healing_section(entries)

    # ---- figures ----------------------------------------------------------------------------
    lines += ["## Figures", ""]

    def figure(path, caption):
        if path:
            alt = Path(path).stem.replace("_", " ")
            lines.extend([f"![{alt}](figures/{Path(path).name})", "", f"*{caption}*", ""])

    rows = []
    if ref_mat or ref_org:
        base_res = (ref_mat or {}).get("results", {})
        rows.append({"name": "base GRACE", "groups": base_res.get("base", {}).get("groups", {}),
                     "coverage_all": base_res.get("base", {}).get("coverage_all"),
                     "organic_f": (base_dft or {}).get("force_mae_meV_A")})
        for i in validated:
            v = vexp[i]
            name = next(iter(v["alias"]), i)
            res = ((v["materials"] or {}).get("results") or {}).get(name, {})
            rows.append({"name": i, "groups": res.get("groups", {}), "coverage_all": res.get("coverage_all"),
                         "organic_f": (organic_dft(v["organic"], name) or {}).get("force_mae_meV_A")})
        off = base_res.get("GRACE-OFF", {})
        if off or teacher_dft:
            rows.append({"name": "GRACE-OFF (teacher)", "groups": off.get("groups", {}),
                         "coverage_all": off.get("coverage_all", 0.0 if not off else None),
                         "organic_f": (teacher_dft or {}).get("force_mae_meV_A")})
    figure(plot_heatmap.make(rows, figs / "heatmap_force_mae.png", sizes),
           "Force error vs DFT for every validated model and dataset (log colour scale, lower is better). "
           "Grey cells: the model cannot run there (unsupported elements) or was not measured.")

    trade = [{"id": e["id"], "parent": e.get("parent"), "feasible": is_feasible(e, base, objective),
              **{k: (e.get("metrics") or {}).get(k) for k in ("organic_f", "materials_f")}} for e in done]
    figure(plot_pareto.make_tradeoff(trade, base, figs / "tradeoff.png",
                                     objective["max_materials_increase"],
                                     teacher_f=None, best=best["id"] if best else None),
           "Search path of the autonomous loop: each experiment's organic error vs materials error on the "
           "held-out monitor sets; arrows go from parent to child, green = within the materials budget.")

    histories = {e["id"]: workdir / e.get("run_dir", f"experiments/{e['id']}/run") / "history.csv"
                 for e in done}
    figure(plot_history.make_monitor_curves(histories, figs / "monitor_curves.png", base),
           "Monitor force error per epoch for every experiment (epoch 0 = untrained base GRACE); "
           "organics should fall while the two materials monitors stay near the base line.")
    runs = {i: plot_history.read_csv(p) for i, p in histories.items()}
    figure(plot_history.make(runs, figs / "training_history.png"),
           "Validation force error, training loss and learning speed per epoch (mixed validation set "
           "of each experiment, so absolute values differ between data mixes).")

    if best:
        vb = vexp.get(best["id"], {})
        rename = {n: best["id"] for n in vb.get("alias", {})}
        if vb.get("organic"):
            written = plot_parity.make(vb["dir"] / "organic", figs, rename, prefix="best_")
            for p in written:
                ref = "DFT" if "vs_dft" in p.name else "GRACE-OFF"
                figure(p, f"Organic test molecules: force parity vs {ref} for base GRACE, {best['id']}"
                          + (" and the teacher" if ref == "DFT" else "") + ".")
        if vb.get("materials"):
            figure(plot_parity.make_materials(vb["dir"] / "materials", figs / "best_parity_materials.png",
                                              ["base", *vb["alias"]], rename),
                   f"Materials (MPtrj DFT) force parity per group for base GRACE and {best['id']}: "
                   "group A must be identical, B and C show any forgetting.")

    points = [{"name": "base", "params": 0, "effort": 0, "force": base.get("organic_f")}]
    for e in done:
        summ = load_json(workdir / e.get("run_dir", "") / "summary.json") or {}
        cfg = e.get("config", {})
        effort = (cfg.get("n_organic", 0) + cfg.get("n_materials", 0)) * cfg.get("epochs", 0) or None
        points.append({"name": e["id"], "params": summ.get("trainable_params"), "effort": effort,
                       "force": (e.get("metrics") or {}).get("organic_f")})
    figure(plot_pareto.make(points, figs / "pareto.png",
                            ylabel="organic force MAE, monitor (meV/Å)",
                            title="Pareto fronts: organic error vs adapter size and training effort"),
           "Cost-accuracy trade-off: organic monitor error vs trainable parameters and training effort "
           "(structures × epochs); outlined points are Pareto-optimal.")

    lines += additional_validations(workdir, out)

    # ---- limitations ------------------------------------------------------------------------
    lines += ["## Limitations", ""]
    lim = []
    if best:
        vb = vexp.get(best["id"], {})
        name = next(iter(vb.get("alias", {})), best["id"])
        groups = (((vb.get("materials") or {}).get("results") or {}).get(name) or {}).get("groups", {})
        if (groups.get("C") or {}).get("n") and (base_groups.get("C") or {}).get("n"):
            r = groups["C"]["force_mae_meV_A"] / base_groups["C"]["force_mae_meV_A"]
            lim.append(f"Crystals made only of adapted elements (group C) remain {r:.1f}× base GRACE's force "
                       f"error ({fmt(groups['C']['force_mae_meV_A'])} vs "
                       f"{fmt(base_groups['C']['force_mae_meV_A'])} meV/Å): the same element rows serve two "
                       "DFT levels of theory.")
        elif best["metrics"].get("only_adapted_f") and base.get("only_adapted_f"):
            r = best["metrics"]["only_adapted_f"] / base["only_adapted_f"]
            lim.append(f"Only-adapted crystals (monitor) remain {r:.1f}× base GRACE's force error.")
        best_dft = organic_dft(vb.get("organic"), name)
        if best_dft and teacher_dft:
            r = best_dft["force_mae_meV_A"] / teacher_dft["force_mae_meV_A"]
            lim.append(f"Organic force error is still {r:.1f}× the teacher's "
                       f"({fmt(best_dft['force_mae_meV_A'])} vs {fmt(teacher_dft['force_mae_meV_A'])} meV/Å "
                       "vs DFT): the adapter trades peak accuracy for breadth and cost.")
    n_inf = sum(not is_feasible(e, base, objective) for e in done)
    if n_inf:
        lim.append(f"{n_inf} of {len(done)} experiment(s) violated the materials budget.")
    failed = [e["id"] for e in entries if e.get("status") == "failed"]
    if failed:
        lim.append(f"Failed experiments: {', '.join(failed)}.")
    not_validated = [e["id"] for e in done if e["id"] not in validated]
    if not_validated:
        lim.append(f"Not fully validated (monitor sets only): {', '.join(not_validated)}.")
    lim.append("Monitor sets are small held-out samples (≤150 structures each); headline numbers come "
               "from the full validation where available.")
    lines += [f"- {x}" for x in lim] + [""]

    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path


def predictions_section(entries):
    """'Predictions vs outcomes' (only if any entry has prediction_check)."""
    rows, errs = [], []
    for e in entries:
        for k, c in (e.get("prediction_check") or {}).items():
            if not isinstance(c, dict):
                continue
            rel = c.get("rel_error")
            if isinstance(rel, (int, float)):
                errs.append(abs(rel))
            rows.append(f"| {e['id']} | {k} | {fmt(c.get('predicted'))} | {fmt(c.get('actual'))} | "
                        f"{fmt(c.get('error'))} | "
                        + (f"{100 * rel:+.0f}%" if isinstance(rel, (int, float)) else "–") + " |")
    if not rows:
        return []
    scores = [f"{e['id']} {e['prediction_score']:.2f}" for e in entries
              if isinstance(e.get("prediction_score"), (int, float))]
    out = ["## Predictions vs outcomes", "",
           "Before each experiment the agent predicted its monitor errors; after it, the predictions "
           "were checked.", "",
           "| Experiment | Quantity | Predicted | Actual | Error | Rel. error |", "|---|---|---|---|---|---|",
           *rows, ""]
    if errs:
        out.append(f"Mean |relative error|: **{100 * np.mean(errs):.0f}%** over {len(errs)} predictions."
                   + (f" Prediction scores: {', '.join(scores)}." if scores else ""))
        out.append("")
    return out


def healing_section(entries):
    """'Self-healing events' table (only if any entry has healing events)."""
    rows = []
    for e in entries:
        for h in e.get("healing") or []:
            change = "; ".join(f"{k} {v[0]}→{v[1]}" if isinstance(v, (list, tuple)) and len(v) == 2 else f"{k} {v}"
                               for k, v in (h.get("change") or {}).items())
            rows.append(f"| {e['id']} | {h.get('attempt', '?')} | `{h.get('error_class', 'error')}` | "
                        f"{change or 'retry'} | {h.get('time', '–')} |")
    if not rows:
        return []
    return ["## Self-healing events", "",
            "Failures the agent diagnosed and repaired automatically before retrying.", "",
            "| Experiment | Attempt | Error class | Change | Time |", "|---|---|---|---|---|", *rows, ""]


def additional_validations(workdir, out):
    """Markdown section linking <workdir>/validation/adhoc_* folders (validation-only runs)."""
    vdir = Path(workdir) / "validation"
    folders = sorted(d for d in vdir.glob("adhoc_*") if d.is_dir()) if vdir.exists() else []
    if not folders:
        return []
    lines = ["## Additional validations", "",
             "Ad-hoc validations of models outside the experiment loop (`pipeline.py validate-models`):", ""]
    for d in folders:
        target = d / "summary.md" if (d / "summary.md").exists() else d
        title = ""
        if target.is_file():
            first = next((l for l in target.read_text().splitlines() if l.startswith("# ")), "")
            title = f": {first[2:].strip()}" if first else ""
        lines.append(f"- [{d.name}]({os.path.relpath(target, out)}){title}")
    return lines + [""]


# --------------------------------------------------------------------------------------------
# validation-only summary (no ledger)
# --------------------------------------------------------------------------------------------
def _ratio_text(x, ref, what):
    """'3.1× better than base on organics' / 'within 11% of base on ...' / 'worse ... by +N%'."""
    if x is None or not ref:
        return None
    change = x / ref - 1
    if change <= -0.15:
        return f"{ref / x:.1f}× better than base on {what}" if x / ref < 0.67 else \
            f"{-100 * change:.0f}% better than base on {what}"
    if abs(change) <= 0.05:
        return f"matches base on {what} ({100 * change:+.0f}%)"
    if change <= 0.15:
        return f"within {100 * change:.0f}% of base on {what}"
    return f"worse than base on {what} by +{100 * change:.0f}%"


def build_validation_only(out, title=None):
    out = Path(out).expanduser()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    org = load_json(out / "organic" / "validation.json")
    mat = load_json(out / "materials" / "materials_validation.json")
    lines = [f"# {title or 'GRACE model validation'}", "",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M')} from `{out}`._", ""]
    if not org:
        print("[report] note: no organic/validation.json; organic part skipped")
    if not mat:
        print("[report] note: no materials/materials_validation.json; materials part skipped")
    if not org and not mat:
        lines += ["No validation outputs found (`organic/validation.json`, "
                  "`materials/materials_validation.json`).", ""]
        path = out / "summary.md"
        path.write_text("\n".join(lines))
        return path

    rows, sizes = plot_heatmap.rows_from_folders([out / "materials"] if mat else [],
                                                 [out / "organic"] if org else [])
    org_metrics = (org or {}).get("metrics", {})
    mat_results = (mat or {}).get("results", {})
    get_org = lambda n: org_metrics.get("teacher" if n == "GRACE-OFF" else n, {})
    base_row = next((r for r in rows if r["name"] == "base"), None)
    base_all = all_materials(base_row["groups"], sizes) if base_row else None

    # ---- data sources -------------------------------------------------------------------------
    lines += ["## Data", ""]
    if org:
        lines.append(f"- Organic: {org.get('n_structures', '?')} structures from `{org.get('structures', '?')}` "
                     "(force/energy vs DFT; teacher = GRACE-OFF).")
    if mat:
        gs = mat.get("group_sizes_available", {})
        n_eval = {g: max((r['groups'].get(g, {}) or {}).get('n_group', 0) for r in mat_results.values())
                  for g in "ABC"}
        lines.append(f"- Materials: `{mat.get('data', '?')}` (DFT); evaluated A {n_eval['A']}, B {n_eval['B']}, "
                     f"C {n_eval['C']} structures (available {gs.get('A', '?')}/{gs.get('B', '?')}/{gs.get('C', '?')}). "
                     f"Adapted elements: {' '.join(mat.get('adapted_elements', []))}.")
    lines.append("")

    # ---- results table -----------------------------------------------------------------------
    lines += ["## Results", "",
              "Force MAE vs DFT in meV/Å (R² in brackets); energy MAE in meV/atom; stress MAE in GPa. "
              "“All mat.” weights groups A/B/C by their share of the full materials set.", "",
              "| Model | Organic F (R²) | Organic E | Mat. A | Mat. B (R²) | Mat. C (R²) | All mat. | "
              "Stress B | vs base: organic | vs base: all mat. |",
              "|---|---|---|---|---|---|---|---|---|---|"]

    def cell(g, r2=False):
        if not g or not g.get("n"):
            return "cannot run" if g is not None and g.get("n") == 0 else "–"
        txt = fmt(g["force_mae_meV_A"]) if g["force_mae_meV_A"] < 1e4 else f"{g['force_mae_meV_A']:.1e}"
        return txt + (f" ({g['force_r2']:.3f})" if r2 and g.get("force_r2") is not None and g["force_r2"] > -10
                      else "")

    base_org = (get_org("base").get("vs_dft") or {}).get("force_mae_meV_A")
    verdicts = []
    for r in rows:
        n = r["name"]
        o = get_org(n).get("vs_dft") or {}
        g = r["groups"] or {}
        allm = all_materials(g, sizes)
        res = mat_results.get(n, {})
        lines.append(
            f"| {'**' + n + '**' if n == 'base' else n} | "
            + (f"{fmt(o.get('force_mae_meV_A'))} ({o['force_r2']:.3f})" if o.get("force_r2") is not None
               else fmt(o.get("force_mae_meV_A")))
            + f" | {fmt(o.get('energy_mae_meV_atom'))} | {cell(g.get('A'))} | {cell(g.get('B'), True)} | "
            f"{cell(g.get('C'), True)} | "
            + (fmt(allm) if allm is not None else
               (f"runs on {100 * res['coverage_all']:.1f}%" if res.get("coverage_all") is not None else "–"))
            + f" | {fmt((g.get('B') or {}).get('stress_mae_GPa'), 3)} | "
            f"{pct(o.get('force_mae_meV_A'), base_org) if n != 'base' else '–'} | "
            f"{pct(allm, base_all) if n != 'base' and allm is not None else '–'} |")

        # ---- verdict ----
        if n == "base":
            continue
        parts = []
        t = _ratio_text(o.get("force_mae_meV_A"), base_org, "organic molecules")
        if t:
            parts.append(t)
        if allm is not None:
            parts.append(_ratio_text(allm, base_all, "all materials"))
            for grp, label in (("B", "materials with adapted + other elements (B)"),
                               ("C", "crystals of only adapted elements (C)")):
                gb, gx = (base_row or {}).get("groups", {}).get(grp), g.get(grp)
                if gb and gx and gb.get("n") and gx.get("n") and gx["force_mae_meV_A"] / gb["force_mae_meV_A"] > 1.15:
                    parts.append(f"{grp}: +{100 * (gx['force_mae_meV_A'] / gb['force_mae_meV_A'] - 1):.0f}% vs base")
            if (g.get("A") or {}).get("n") and ((base_row or {}).get("groups", {}).get("A") or {}).get("n"):
                if abs(g["A"]["force_mae_meV_A"] - base_row["groups"]["A"]["force_mae_meV_A"]) < 0.05:
                    parts.append("identical to base on materials without adapted elements (A)")
        elif res.get("coverage_all") is not None:
            parts.append(f"cannot run on {100 * (1 - res['coverage_all']):.1f}% of materials "
                         "(elements outside its training)")
            gc = g.get("C") or {}
            if gc.get("n") and gc.get("force_mae_median_structure_meV_A") and \
                    gc["force_mae_meV_A"] > 5 * gc["force_mae_median_structure_meV_A"]:
                parts.append(f"unstable on the materials it can run on (group C median "
                             f"{gc['force_mae_median_structure_meV_A']:.0f} meV/Å, mean {gc['force_mae_meV_A']:.1e})")
        teacher_o = (get_org("GRACE-OFF").get("vs_dft") or {}).get("force_mae_meV_A")
        if n != "GRACE-OFF" and teacher_o and o.get("force_mae_meV_A"):
            parts.append(f"organic error {o['force_mae_meV_A'] / teacher_o:.1f}× the teacher's")
        if parts:
            verdicts.append(f"- **{n}**: " + "; ".join(p for p in parts if p) + ".")
    lines += ["", "## Verdict", ""] + (verdicts or ["- Only the base model was validated."]) + [""]

    # ---- figures -----------------------------------------------------------------------------
    lines += ["## Figures", ""]

    def figure(path, caption):
        if path:
            lines.extend([f"![{Path(path).stem.replace('_', ' ')}](figures/{Path(path).name})", "",
                          f"*{caption}*", ""])

    figure(plot_heatmap.make(rows, figs / "heatmap_force_mae.png", sizes),
           "Force error vs DFT per model and dataset (log colour scale, lower is better); grey = the model "
           "cannot run there or it was not measured.")
    if org:
        for p in plot_parity.make(out / "organic", figs):
            ref = "DFT" if "vs_dft" in p.name else "GRACE-OFF"
            figure(p, f"Organic molecules: force parity vs {ref} for every validated model.")
    if mat:
        figure(plot_parity.make_materials(out / "materials", figs / "parity_materials.png"),
               "Materials (MPtrj DFT): force parity per group (rows A/B/C) and model (columns).")
    path = out / "summary.md"
    path.write_text("\n".join(lines))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", help="pipeline workdir (experiments/, validation/)")
    p.add_argument("--exps", nargs="*", help="only these experiments (default: all)")
    p.add_argument("--out", help="output folder (default: <workdir>/report)")
    p.add_argument("--validation-only", metavar="OUT",
                   help="no ledger: summarise <OUT>/organic and <OUT>/materials into <OUT>/summary.md")
    p.add_argument("--title", help="title for --validation-only")
    args = p.parse_args()
    if args.validation_only:
        out = Path(args.validation_only).expanduser()
        path = build_validation_only(out, args.title)
        figs = sorted((out / "figures").glob("*.png"))
        print(f"Saved: {path}  ({len(figs)} figures: {', '.join(f.name for f in figs)})")
        return
    if not args.workdir:
        p.error("--workdir is required (or use --validation-only <out>)")
    out = Path(args.out).expanduser() if args.out else Path(args.workdir).expanduser() / "report"
    path = build(args.workdir, out, args.exps)
    figs = sorted((out / "figures").glob("*.png"))
    print(f"Saved: {path}  ({len(figs)} figures: {', '.join(f.name for f in figs)})")


if __name__ == "__main__":
    main()
