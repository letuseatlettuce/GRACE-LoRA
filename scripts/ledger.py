"""Experiment ledger for the autonomous LoRA loop: <workdir>/experiments/ledger.json.

Stdlib only, so it runs with any Python. Format (see docs/CONTRACTS.md):
    {"version": 1,
     "objective": {"primary": "organic_f", "max_materials_increase": 0.15,
                   "max_only_adapted_increase": 0.75},
     "base": {"organic_f": ..., "materials_f": ..., "only_adapted_f": ...},
     "experiments": [{"id": "exp_001", "parent": null, "status": "done", "config": {...},
                      "metrics": {...}, "relative": {...}, "feasible": true, "score": ...}]}

    python scripts/ledger.py --workdir W show [--json]
    python scripts/ledger.py --workdir W best
    python scripts/ledger.py --workdir W get exp_003
    python scripts/ledger.py --workdir W set-objective max_materials_increase=0.10
    python scripts/ledger.py --workdir W annotate exp_003 review "plausible" --verdict accept
    python scripts/ledger.py --workdir W predictions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERSION = 1
DEFAULT_OBJECTIVE = {"primary": "organic_f", "max_materials_increase": 0.15,
                     "max_only_adapted_increase": 0.75}
MONITORS = ("organic", "materials", "only_adapted")

# experiment config defaults (CONTRACTS.md); experiment.py validates against these keys
DEFAULT_CONFIG = {
    "n_organic": 1000, "n_materials": 600, "n_materials_val": 100, "n_only_adapted": 170,
    "rank": 8, "alpha": None, "epochs": 10, "batch_size": 8, "max_buckets": 4, "lr": None,
    "max_atoms": 30, "oversample": 3.0, "seed": 0, "device": "cpu-jit",
    "adapt_elements": None,
}


def settings_objective(workdir) -> dict:
    """Objective limits from pipeline.json (repo root first, then the workdir; same lookup as
    experiment.py), restricted to known keys. {} when not configured."""
    for path in (REPO / "pipeline.json", Path(workdir).expanduser() / "pipeline.json"):
        if path.exists():
            try:
                obj = json.loads(path.read_text()).get("objective") or {}
            except (json.JSONDecodeError, AttributeError):
                return {}
            return {k: v for k, v in obj.items() if k in DEFAULT_OBJECTIVE}
    return {}


def configured_objective(workdir) -> dict:
    return {**DEFAULT_OBJECTIVE, **settings_objective(workdir)}


def ledger_path(workdir) -> Path:
    return Path(workdir).expanduser() / "experiments" / "ledger.json"


def new_ledger(workdir=None) -> dict:
    objective = configured_objective(workdir) if workdir is not None else dict(DEFAULT_OBJECTIVE)
    return {"version": VERSION, "objective": objective, "base": {}, "experiments": []}


def load(workdir) -> dict:
    """Return the ledger (a fresh empty one if the file does not exist yet)."""
    path = ledger_path(workdir)
    if not path.exists():
        return new_ledger(workdir)
    ledger = json.loads(path.read_text())
    ledger.setdefault("version", VERSION)
    ledger.setdefault("objective", dict(DEFAULT_OBJECTIVE))
    for key, value in DEFAULT_OBJECTIVE.items():
        ledger["objective"].setdefault(key, value)
    ledger.setdefault("base", {})
    ledger.setdefault("experiments", [])
    return ledger


def save(workdir, ledger: dict) -> Path:
    """Atomic write: temp file in the same folder, then rename."""
    path = ledger_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".ledger.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(ledger, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def get(ledger: dict, exp_id: str):
    return next((e for e in ledger["experiments"] if e["id"] == exp_id), None)


def append(workdir, entry: dict) -> dict:
    """Add (or replace, same id) an entry and save; returns the updated ledger."""
    ledger = load(workdir)
    ledger["experiments"] = [e for e in ledger["experiments"] if e["id"] != entry["id"]]
    ledger["experiments"].append(entry)
    ledger["experiments"].sort(key=lambda e: e["id"])
    save(workdir, ledger)
    return ledger


def update_entry(workdir, exp_id: str, **fields) -> dict:
    """Merge fields into an existing entry (creating it if missing) and save."""
    ledger = load(workdir)
    entry = get(ledger, exp_id)
    if entry is None:
        entry = {"id": exp_id}
        ledger["experiments"].append(entry)
        ledger["experiments"].sort(key=lambda e: e["id"])
    entry.update(fields)
    save(workdir, ledger)
    return ledger


def next_id(ledger: dict) -> str:
    """exp_001, exp_002, ... (one more than the highest existing number)."""
    numbers = [int(e["id"].split("_")[-1]) for e in ledger["experiments"]
               if e.get("id", "").startswith("exp_") and e["id"].split("_")[-1].isdigit()]
    return f"exp_{(max(numbers) + 1 if numbers else 1):03d}"


def relative_metrics(metrics: dict, base: dict) -> dict:
    """organic_gain = 1 - organic_f/base; <monitor>_increase = x_f/base - 1."""
    rel = {}
    if metrics.get("organic_f") is not None and base.get("organic_f"):
        rel["organic_gain"] = 1 - metrics["organic_f"] / base["organic_f"]
    for name in ("materials", "only_adapted"):
        value, ref = metrics.get(f"{name}_f"), base.get(f"{name}_f")
        if value is not None and ref:
            rel[f"{name}_increase"] = value / ref - 1
    return rel


def is_feasible(relative: dict, objective: dict) -> bool:
    """Feasible = materials and only-adapted increases within the objective's limits.
    A monitor that is missing does not count against feasibility."""
    mat = relative.get("materials_increase")
    only = relative.get("only_adapted_increase")
    return ((mat is None or mat <= objective["max_materials_increase"])
            and (only is None or only <= objective["max_only_adapted_increase"]))


def evaluate_entry(entry: dict, ledger: dict) -> dict:
    """Fill relative / feasible / score of a finished entry from its metrics."""
    metrics = entry.get("metrics") or {}
    entry["relative"] = relative_metrics(metrics, ledger.get("base") or {})
    entry["feasible"] = is_feasible(entry["relative"], ledger["objective"])
    entry["score"] = metrics.get(ledger["objective"].get("primary", "organic_f"))
    return entry


def rescore(ledger: dict) -> dict:
    """Recompute relative/feasible/score for every finished entry (after objective or base
    changes)."""
    for entry in ledger["experiments"]:
        if entry.get("status") == "done" and entry.get("metrics"):
            evaluate_entry(entry, ledger)
    return ledger


VERDICTS = ("accept", "doubt", "reject")


def prediction_check(prediction: dict, metrics: dict):
    """({key: {predicted, actual, error, rel_error}}, mean |rel_error|) for the predicted
    monitors that were measured; rel_error = (actual - predicted) / actual."""
    check = {}
    for key, predicted in (prediction or {}).items():
        actual = (metrics or {}).get(key)
        if actual is None:
            continue
        error = actual - predicted
        check[key] = {"predicted": predicted, "actual": actual, "error": error,
                      "rel_error": error / actual if actual else None}
    rels = [abs(c["rel_error"]) for c in check.values() if c["rel_error"] is not None]
    return check, (sum(rels) / len(rels) if rels else None)


def annotate(workdir, exp_id: str, field: str, text: str, verdict=None) -> dict:
    """Append {time, text, verdict} to a list field of an entry (e.g. 'review', 'notes')."""
    if verdict is not None and verdict not in VERDICTS:
        raise SystemExit(f"--verdict must be one of {VERDICTS}")
    ledger = load(workdir)
    entry = get(ledger, exp_id)
    if entry is None:
        raise SystemExit(f"No experiment {exp_id}")
    current = entry.setdefault(field, [])
    if not isinstance(current, list):
        raise SystemExit(f"field '{field}' of {exp_id} is not a list; choose another field name")
    import time
    note = {"time": time.strftime("%Y-%m-%d %H:%M"), "text": text, "verdict": verdict}
    current.append(note)
    save(workdir, ledger)
    return note


def sync_objective(workdir, verbose=True) -> dict:
    """Make the ledger's objective match pipeline.json (if it configures one): update,
    re-score every entry, save, and print a one-line note. Returns the ledger."""
    ledger = load(workdir)
    wanted = configured_objective(workdir)
    if not settings_objective(workdir) or wanted == ledger["objective"]:
        return ledger
    changes = {k: [ledger["objective"].get(k), v] for k, v in wanted.items()
               if ledger["objective"].get(k) != v}
    ledger["objective"] = wanted
    rescore(ledger)
    if ledger_path(workdir).exists():
        save(workdir, ledger)
    if verbose:
        print("Objective updated from pipeline.json: "
              + ", ".join(f"{k} {o} -> {n}" for k, (o, n) in changes.items())
              + " (feasibility and best re-scored)")
    return ledger


def best(ledger: dict):
    """Best = feasible experiment with the lowest score; if none is feasible, the finished
    experiment with the lowest materials increase. None if nothing finished."""
    done = [e for e in ledger["experiments"]
            if e.get("status") == "done" and e.get("score") is not None]
    if not done:
        return None
    feasible = [e for e in done if e.get("feasible")]
    if feasible:
        return min(feasible, key=lambda e: e["score"])
    return min(done, key=lambda e: (e.get("relative") or {}).get("materials_increase", float("inf")))


def config_deltas(config: dict) -> str:
    """Compact 'key=value' list of settings that differ from the defaults."""
    parts = [f"{k}={v}" for k, v in (config or {}).items()
             if k in DEFAULT_CONFIG and v != DEFAULT_CONFIG[k]]
    return " ".join(parts) or "defaults"


def _fmt(value, spec):
    return format(value, spec) if isinstance(value, (int, float)) else "-"


def _pct(value):
    return f"{100 * value:+.0f}%" if isinstance(value, (int, float)) else "-"


def _extras(e: dict) -> str:
    out = []
    if e.get("healing"):
        out.append("healed: " + ",".join(h["error_class"] for h in e["healing"]))
    if isinstance(e.get("prediction_score"), (int, float)):
        out.append(f"pred {100 * e['prediction_score']:.0f}%")
    if e.get("review"):
        verdicts = [r.get("verdict") for r in e["review"] if r.get("verdict")]
        out.append(f"review {verdicts[-1]}" if verdicts else "reviewed")
    return ("  [" + "; ".join(out) + "]") if out else ""


def format_predictions(ledger: dict) -> str:
    lines = [f"{'id':9s} {'monitor':15s} {'predicted':>10s} {'actual':>9s} {'error':>9s} "
             f"{'rel':>7s}  hypothesis"]
    lines.append("-" * len(lines[0]))
    for e in ledger["experiments"]:
        if not e.get("prediction"):
            continue
        check = e.get("prediction_check") or {}
        hypothesis = (e.get("hypothesis") or "")[:60]
        for i, (key, predicted) in enumerate(e["prediction"].items()):
            c = check.get(key) or {}
            lines.append(f"{(e['id'] if i == 0 else ''):9s} {key:15s} {predicted:10.1f} "
                         f"{_fmt(c.get('actual'), '9.1f')} {_fmt(c.get('error'), '+9.1f')} "
                         f"{_pct(c.get('rel_error')):>7s}  {hypothesis if i == 0 else ''}")
        if isinstance(e.get("prediction_score"), (int, float)):
            lines.append(f"{'':9s} {'mean |rel err|':15s} {'':10s} {'':9s} {'':9s} "
                         f"{100 * e['prediction_score']:6.0f}%")
    if len(lines) == 2:
        lines.append("(no experiment has predictions)")
    return "\n".join(lines)


def format_table(ledger: dict) -> str:
    base, obj = ledger.get("base") or {}, ledger["objective"]
    best_entry = best(ledger)
    lines = [f"Objective: minimise {obj['primary']} with materials <= "
             f"+{100 * obj['max_materials_increase']:.0f}% and only-adapted <= "
             f"+{100 * obj['max_only_adapted_increase']:.0f}% vs base",
             "Base (epoch 0): " + ("  ".join(f"{m} {_fmt(base.get(f'{m}_f'), '.1f')}"
                                            for m in MONITORS) if base else "not set yet"),
             ""]
    header = (f"{'id':10s} {'parent':8s} {'status':11s} {'organic':>8s} {'materials':>17s} "
              f"{'only_adapted':>17s} {'feasible':>8s} {'time':>7s}  config (vs defaults)")
    lines += [header, "-" * len(header)]
    for e in ledger["experiments"]:
        m, r = e.get("metrics") or {}, e.get("relative") or {}
        mark = " *" if best_entry is not None and e["id"] == best_entry["id"] else ""
        if e.get("review"):
            mark += "✎"
        duration = e.get("duration_s")
        lines.append(
            f"{e['id'] + mark:10s} {str(e.get('parent') or '-'):8s} {e.get('status', '?'):11s} "
            f"{_fmt(m.get('organic_f'), '8.1f')} "
            f"{_fmt(m.get('materials_f'), '8.1f')} {_pct(r.get('materials_increase')):>8s} "
            f"{_fmt(m.get('only_adapted_f'), '8.1f')} {_pct(r.get('only_adapted_increase')):>8s} "
            f"{('yes' if e.get('feasible') else 'no') if e.get('status') == 'done' else '-':>8s} "
            f"{(f'{duration / 60:.0f} min' if isinstance(duration, (int, float)) else '-'):>7s}  "
            f"{config_deltas(e.get('config'))}" + _extras(e))
    lines.append("")
    lines.append("force MAE in meV/A; * = best; ✎ = reviewed; % = change vs base; "
                 "pred = mean |relative error| of the predictions")
    return "\n".join(lines)


def _parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default=".", help="pipeline workdir (contains experiments/)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="table of all experiments")
    s.add_argument("--json", action="store_true")
    sub.add_parser("best", help="best experiment (JSON)")
    g = sub.add_parser("get", help="one experiment (JSON)")
    g.add_argument("exp_id")
    o = sub.add_parser("set-objective", help="change objective limits, e.g. "
                                             "max_materials_increase=0.10")
    o.add_argument("assignments", nargs="+")
    a = sub.add_parser("annotate", help="append a note/review to a list field of an entry")
    a.add_argument("exp_id")
    a.add_argument("field", help="e.g. review (critic) or notes")
    a.add_argument("text")
    a.add_argument("--verdict", choices=VERDICTS, default=None)
    sub.add_parser("predictions", help="predicted vs actual monitors for all experiments")
    args = p.parse_args(argv)
    ledger = load(args.workdir)

    if args.cmd == "show":
        print(json.dumps(ledger, indent=2) if args.json else format_table(ledger))
    elif args.cmd == "best":
        entry = best(ledger)
        if entry is None:
            raise SystemExit("No finished experiment yet.")
        print(json.dumps(entry, indent=2))
    elif args.cmd == "get":
        entry = get(ledger, args.exp_id)
        if entry is None:
            raise SystemExit(f"No experiment {args.exp_id}")
        print(json.dumps(entry, indent=2))
    elif args.cmd == "set-objective":
        for assignment in args.assignments:
            key, _, value = assignment.partition("=")
            if key not in DEFAULT_OBJECTIVE or not value:
                raise SystemExit(f"Use key=value with key in {sorted(DEFAULT_OBJECTIVE)}")
            ledger["objective"][key] = _parse_value(value)
        save(args.workdir, rescore(ledger))
        print(json.dumps(ledger["objective"], indent=2))
        if settings_objective(args.workdir):
            print("NOTE: pipeline.json also sets an objective; it is applied again at the start "
                  "of the next experiment. Change it there to make this permanent.")
    elif args.cmd == "annotate":
        note = annotate(args.workdir, args.exp_id, args.field, args.text, args.verdict)
        print(json.dumps({"id": args.exp_id, "field": args.field, **note}))
    elif args.cmd == "predictions":
        print(format_predictions(ledger))


if __name__ == "__main__":
    sys.exit(main())
