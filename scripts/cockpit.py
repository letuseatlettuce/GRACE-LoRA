"""Agent cockpit: one self-contained HTML page showing the autonomous loop live.

    python scripts/cockpit.py --workdir W [--out W/report/cockpit.html] [--watch SECONDS]
    python scripts/cockpit.py --workdir W --artifact [--out W/report/cockpit_artifact/cockpit.html]

Reads <workdir>/experiments/ledger.json and each run's (possibly partial) history.csv and
writes ONE HTML file: inline CSS (light/dark aware), figures embedded as base64 PNG, no
network access. The page refreshes itself every 20 s; with --watch N the file is rebuilt every
N seconds until Ctrl+C. Partial or missing data never crashes it: each section degrades to a note.

--artifact writes a version for publishing as a Claude artifact: a page fragment (the artifact
skeleton adds <html>/<head>/<body>) with theme tokens for the viewer's light/dark setting, figures
as separate files in figures/ next to the page (publish them with the page), no auto-refresh
(republishing updates open viewers) and no local paths.

Sections: status header, current experiment (rationale, hypothesis, progress), search path
(trade-off scatter), monitor curves (incl. the running experiment), ledger table, decision log
with critic reviews and self-healing events, predicted vs actual.
"""

import argparse
import base64
import html
import json
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_history  # noqa: E402
import plot_pareto  # noqa: E402
import report  # noqa: E402

REFRESH_S = 20
esc = lambda x: html.escape(str(x))

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--muted:#667085;--line:#e4e7ec;--accent:#2563eb;
--ok:#15803d;--okbg:#dcfce7;--bad:#b45309;--badbg:#fef3c7;--err:#b91c1c;--errbg:#fee2e2;
--run:#1d4ed8;--runbg:#dbeafe}
@media (prefers-color-scheme:dark){:root{--bg:#0f141b;--card:#171e28;--fg:#e6e9ef;--muted:#98a2b3;
--line:#2a3442;--accent:#60a5fa;--ok:#4ade80;--okbg:#123020;--bad:#fbbf24;--badbg:#3a2a0a;
--err:#f87171;--errbg:#3b1212;--run:#93c5fd;--runbg:#132a4a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px}
h1{font-size:22px;margin:0}h2{font-size:16px;margin:0 0 10px;color:var(--muted);
text-transform:uppercase;letter-spacing:.05em}
.top{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.sub{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.kpi .v{font-size:24px;font-weight:650}.kpi .l{color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.two{grid-template-columns:1fr}}
img.fig{width:100%;height:auto;background:#fff;border-radius:8px;padding:4px}
table{width:100%;border-collapse:collapse;font-size:13.5px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.tw{overflow-x:auto}
.b{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.b.ok{background:var(--okbg);color:var(--ok)}.b.bad{background:var(--badbg);color:var(--bad)}
.b.err{background:var(--errbg);color:var(--err)}.b.run{background:var(--runbg);color:var(--run)}
.log{border-left:3px solid var(--line);padding:2px 0 2px 14px;margin:0 0 14px}
.log.best{border-color:var(--ok)}.log h3{font-size:15px;margin:0 0 4px}
.q{color:var(--muted);font-size:13px;margin:3px 0}.rev{background:var(--bg);border-radius:8px;padding:6px 10px;margin:6px 0;font-size:13.5px}
.heal{font-size:13px;margin:4px 0}.note{color:var(--muted);font-style:italic}
.bar{height:8px;background:var(--line);border-radius:999px;overflow:hidden;margin:8px 0}
.bar>div{height:100%;background:var(--accent)}code{font-size:12.5px;overflow-wrap:anywhere}
.two>*,.grid>*{min-width:0}.wrap{overflow-wrap:anywhere}
@media(max-width:600px){.wrap{padding:12px}.kpi .v{font-size:20px}.grid{grid-template-columns:repeat(2,1fr)}}
"""

# Artifact version: colours as theme tokens (light on :root; dark for the OS setting unless the
# viewer chose light, and again for an explicit dark choice), side gutters clear of safe areas.
ART_FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:'
             'wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')
_DARK = """--bg:#0f1418;--card:#161d23;--fg:#e3e8ec;--muted:#93a1ad;--line:#26313a;--accent:#5fb3c9;
--ok:#4ade80;--okbg:#12301f;--bad:#f5b84a;--badbg:#3a2a0d;--err:#f58787;--errbg:#3b1515;
--run:#8ec5ff;--runbg:#132a44;color-scheme:dark"""
ART_CSS = """
/* layout: status tiles on top, then current experiment beside the search path, curves, ledger, log */
:root{--bg:#f4f6f7;--card:#ffffff;--fg:#17212b;--muted:#5b6876;--line:#dde3e8;--accent:#1f7a8c;
--ok:#15803d;--okbg:#dcf3e4;--bad:#a15c07;--badbg:#fbefd6;--err:#b42323;--errbg:#fbe1e1;
--run:#1d4f91;--runbg:#dde9fa;--sans:"IBM Plex Sans",-apple-system,"Segoe UI",Roboto,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){""" + _DARK + """}}
:root[data-theme="dark"]{""" + _DARK + """}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:15px/1.5 var(--sans)}
.wrap{max-width:1280px;margin:0 auto;padding-block:20px;
padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font-size:22px;margin:0;font-weight:600;text-wrap:balance}
h2{font:600 12.5px/1.3 var(--mono);margin:0 0 10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.top{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.sub{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .v{font:600 24px/1.2 var(--mono);font-variant-numeric:tabular-nums}.kpi .l{color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.two{grid-template-columns:1fr}}
img.fig{width:100%;max-width:100%;height:auto;background:#ffffff;border-radius:6px;padding:4px}
table{width:100%;border-collapse:collapse;font-size:13.5px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
td.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.tw{overflow-x:auto}
.b{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.b.ok{background:var(--okbg);color:var(--ok)}.b.bad{background:var(--badbg);color:var(--bad)}
.b.err{background:var(--errbg);color:var(--err)}.b.run{background:var(--runbg);color:var(--run)}
.log{border-left:3px solid var(--line);padding:2px 0 2px 14px;margin:0 0 14px}
.log.best{border-color:var(--ok)}.log h3{font-size:15px;margin:0 0 4px}
.q{color:var(--muted);font-size:13px;margin:3px 0}.rev{background:var(--bg);border-radius:8px;padding:6px 10px;margin:6px 0;font-size:13.5px}
.heal{font-size:13px;margin:4px 0}.note{color:var(--muted);font-style:italic}
.bar{height:8px;background:var(--line);border-radius:999px;overflow:hidden;margin:8px 0}
.bar>div{height:100%;background:var(--accent)}code{font:12.5px var(--mono);overflow-wrap:anywhere}
.two>*,.grid>*{min-width:0}.wrap{overflow-wrap:anywhere}
@media(max-width:600px){.kpi .v{font-size:20px}.grid{grid-template-columns:repeat(2,1fr)}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""
ASSET_DIR = None  # artifact mode: figures are copied here and referenced relatively


# --------------------------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------------------------
def parse_time(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp()
    except (TypeError, ValueError):
        return None


def badge(entry, base, objective):
    status = entry.get("status", "?")
    if status == "running":
        return '<span class="b run">running</span>'
    if status != "done":
        return f'<span class="b err">{esc(status)}</span>'
    ok = report.is_feasible(entry, base, objective)
    return f'<span class="b {"ok" if ok else "bad"}">{"feasible" if ok else "infeasible"}</span>'


def png_tag(path, alt):
    if not path or not Path(path).exists():
        return f'<p class="note">{esc(alt)}: not available yet.</p>'
    if ASSET_DIR is not None:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        target = ASSET_DIR / Path(path).name
        target.write_bytes(Path(path).read_bytes())
        return f'<img class="fig" alt="{esc(alt)}" src="figures/{esc(target.name)}">'
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img class="fig" alt="{esc(alt)}" src="data:image/png;base64,{data}">'


def safe(fn, *args):
    """Run a section builder; on any error render a short note instead of crashing."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - the cockpit must never crash
        traceback.print_exc()
        return f'<div class="card"><p class="note">Section unavailable ({esc(type(exc).__name__)}: {esc(exc)})</p></div>'


def run_progress(workdir, entry):
    """(epochs done, epochs planned, seconds per epoch or None) from a run's history.csv."""
    run = plot_history.read_csv(Path(workdir) / entry.get("run_dir", "") / "history.csv")
    planned = (entry.get("config") or {}).get("epochs")
    if not run:
        return 0, planned, None
    done = int(run["epoch"].max())
    times = [t for t, e in zip(*_times(workdir, entry)) if e >= 1]
    return done, planned, (float(np.mean(times[1:] or times)) if times else None)


def _times(workdir, entry):
    import csv
    path = Path(workdir) / entry.get("run_dir", "") / "history.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [float(r.get("time_s") or 0) for r in rows], [int(float(r["epoch"])) for r in rows]


# --------------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------------
def header(ctx):
    ledger, entries, base, objective = ctx["ledger"], ctx["entries"], ctx["base"], ctx["objective"]
    done = [e for e in entries if e.get("status") == "done"]
    running = [e for e in entries if e.get("status") == "running"]
    failed = [e for e in entries if e.get("status") not in ("done", "running")]
    starts = [t for t in (parse_time(e.get("time")) for e in entries) if t]
    ends = [parse_time(e.get("time")) + (e.get("duration_s") or 0) for e in done if parse_time(e.get("time"))]
    elapsed = ((time.time() if running else max(ends or [time.time()])) - min(starts)) if starts else None
    best = ctx["best"]
    kpis = [(f"{len(done)}", "experiments done"), (f"{len(running)}", "running"),
            (f"{len(failed)}", "failed / interrupted"),
            (report.duration(elapsed) if elapsed else "–", "elapsed")]
    if best and base.get("organic_f"):
        bm = best["metrics"]
        kpis.append((f"{base['organic_f'] / bm['organic_f']:.1f}×", f"organics vs base ({esc(best['id'])})"))
        if bm.get("materials_f") and base.get("materials_f"):
            kpis.append((report.pct(bm["materials_f"], base["materials_f"]), "materials vs base"))
    cells = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in kpis)
    goal = (f"Goal: minimise organic error, materials within +{100 * objective['max_materials_increase']:.0f}%"
            f" and only-adapted crystals within +{100 * objective['max_only_adapted_increase']:.0f}% of base.")
    return f'<div class="grid">{cells}</div><p class="sub">{esc(goal)}</p>'


def current(ctx):
    entries = ctx["entries"]
    if not entries:
        return '<div class="card"><h2>Current experiment</h2><p class="note">No experiment started yet.</p></div>'
    e = next((x for x in reversed(entries) if x.get("status") == "running"), entries[-1])
    parent_cfg = (ctx["by_id"].get(e.get("parent")) or {}).get("config")
    parts = [f'<div class="card"><h2>{"Running now" if e.get("status") == "running" else "Last experiment"}</h2>',
             f'<h3 style="margin:0 0 6px">{esc(e["id"])} {badge(e, ctx["base"], ctx["objective"])}'
             f' <span class="sub">from {esc(e.get("parent") or "base")} · started {esc(e.get("time", "?"))}</span></h3>',
             f'<p class="q"><b>Change:</b> {esc(report.config_delta(e.get("config", {}), parent_cfg))}</p>',
             f'<p><b>Rationale:</b> {esc(e.get("rationale") or "–")}</p>']
    if e.get("hypothesis"):
        parts.append(f'<p><b>Hypothesis:</b> {esc(e["hypothesis"])}</p>')
    if e.get("prediction"):
        pred = ", ".join(f"{esc(k)} ≈ {report.fmt(v) if isinstance(v, (int, float)) else esc(v)}"
                         for k, v in e["prediction"].items())
        parts.append(f'<p class="q"><b>Predicted:</b> {pred}</p>')
    if e.get("status") == "running":
        done, planned, per = run_progress(ctx["workdir"], e)
        if planned:
            frac = min(done / planned, 1.0)
            eta = f" · ~{report.duration(per * (planned - done))} left" if per and done < planned else ""
            parts.append(f'<div class="bar"><div style="width:{100 * frac:.0f}%"></div></div>'
                         f'<p class="sub">epoch {done}/{planned}{eta}</p>')
    elif e.get("metrics"):
        m = e["metrics"]
        parts.append(f'<p class="q"><b>Outcome:</b> organic {report.fmt(m.get("organic_f"))}, materials '
                     f'{report.fmt(m.get("materials_f"))}, only-adapted {report.fmt(m.get("only_adapted_f"))} meV/Å</p>')
    if e.get("error"):
        parts.append(f'<p class="q" style="color:var(--err)"><b>Error:</b> {esc(e["error"])}</p>')
    return "".join(parts) + "</div>"


def figures(ctx, tmp):
    entries, base, objective = ctx["entries"], ctx["base"], ctx["objective"]
    done = [e for e in entries if e.get("status") == "done"]
    trade = [{"id": e["id"], "parent": e.get("parent"), "feasible": report.is_feasible(e, base, objective),
              **{k: (e.get("metrics") or {}).get(k) for k in ("organic_f", "materials_f")}} for e in done]
    t = plot_pareto.make_tradeoff(trade, base, Path(tmp) / "tradeoff.png", objective["max_materials_increase"],
                                  best=ctx["best"]["id"] if ctx["best"] else None)
    hist = {}
    for e in entries:
        if e.get("status") in ("done", "running"):
            label = e["id"] + (" (running)" if e.get("status") == "running" else "")
            hist[label] = Path(ctx["workdir"]) / e.get("run_dir", f"experiments/{e['id']}/run") / "history.csv"
    m = plot_history.make_monitor_curves(hist, Path(tmp) / "monitors.png", base)
    search = ('<div class="card"><h2>Search path</h2>' + png_tag(t, "trade-off scatter") +
              '<p class="sub">Organic vs materials error on held-out monitors; arrows parent → child; '
              'green band = materials budget.</p></div>')
    monitors = ('<div class="card"><h2>Monitors per epoch</h2>' + png_tag(m, "monitor curves") +
                '<p class="sub">Epoch 0 = untrained base GRACE; the running experiment updates live.</p></div>')
    return search, monitors


def ledger_table(ctx):
    base, objective = ctx["base"], ctx["objective"]
    rows = [f'<tr><td>base</td><td>–</td><td>untrained GRACE</td><td class="n">{report.fmt(base.get("organic_f"))}</td>'
            f'<td class="n">{report.fmt(base.get("materials_f"))}</td><td class="n">{report.fmt(base.get("only_adapted_f"))}</td>'
            f'<td>–</td><td>–</td><td>–</td></tr>']
    for e in ctx["entries"]:
        m = e.get("metrics") or {}
        parent_cfg = (ctx["by_id"].get(e.get("parent")) or {}).get("config")
        best = ctx["best"] and e["id"] == ctx["best"]["id"]
        score = e.get("prediction_score")
        rows.append(
            f'<tr><td>{"<b>" if best else ""}{esc(e["id"])}{" ★</b>" if best else ""}</td>'
            f'<td>{esc(e.get("parent") or "–")}</td>'
            f'<td>{esc(report.config_delta(e.get("config", {}), parent_cfg))}</td>'
            f'<td class="n">{report.fmt(m.get("organic_f"))}</td><td class="n">{report.fmt(m.get("materials_f"))}</td>'
            f'<td class="n">{report.fmt(m.get("only_adapted_f"))}</td><td>{badge(e, base, objective)}</td>'
            f'<td class="n">{report.duration(e.get("duration_s"))}</td>'
            f'<td class="n">{report.fmt(score, 2) if isinstance(score, (int, float)) else "–"}</td></tr>')
    return ('<div class="card"><h2>Ledger</h2><div class="tw"><table><tr><th>Experiment</th><th>Parent</th>'
            '<th>Change vs parent</th><th>Organic</th><th>Materials</th><th>Only-adapted</th><th>Status</th>'
            '<th>Time</th><th>Prediction score</th></tr>' + "".join(rows) + '</table></div>'
            '<p class="sub">Monitor force MAE (meV/Å) at the selected epoch. ★ = best so far.</p></div>')


def decision_log(ctx):
    items = []
    for e in reversed(ctx["entries"]):  # newest first
        m = e.get("metrics") or {}
        best = ctx["best"] and e["id"] == ctx["best"]["id"]
        out = [f'<div class="log{" best" if best else ""}"><h3>{esc(e["id"])} {badge(e, ctx["base"], ctx["objective"])}'
               f' <span class="sub">{esc(e.get("time", ""))}</span></h3>',
               f'<p>{esc(e.get("rationale") or "No rationale recorded.")}</p>']
        if e.get("hypothesis"):
            out.append(f'<p class="q"><b>Hypothesis:</b> {esc(e["hypothesis"])}</p>')
        if m:
            out.append(f'<p class="q"><b>Outcome:</b> organic {report.fmt(m.get("organic_f"))}, materials '
                       f'{report.fmt(m.get("materials_f"))}, only-adapted {report.fmt(m.get("only_adapted_f"))} meV/Å</p>')
        for k, c in (e.get("prediction_check") or {}).items():
            if isinstance(c, dict):
                rel = c.get("rel_error")
                out.append(f'<p class="q">Predicted {esc(k)} {report.fmt(c.get("predicted"))} → actual '
                           f'{report.fmt(c.get("actual"))}' + (f' ({100 * rel:+.0f}%)' if isinstance(rel, (int, float)) else "")
                           + '</p>')
        for r in e.get("review") or []:
            verdict = str(r.get("verdict", "")).lower()
            cls = "ok" if verdict in ("accept", "accepted", "approve", "approved", "ok", "pass", "agree") else \
                ("err" if verdict in ("reject", "rejected", "fail", "block") else "bad")
            out.append(f'<div class="rev"><span class="b {cls}">critic: {esc(r.get("verdict", "note"))}</span> '
                       f'{esc(r.get("text", ""))} <span class="sub">{esc(r.get("time", ""))}</span></div>')
        for h in e.get("healing") or []:
            change = ", ".join(f"{esc(k)} {esc(v[0])}→{esc(v[1])}" if isinstance(v, (list, tuple)) and len(v) == 2
                               else f"{esc(k)} {esc(v)}" for k, v in (h.get("change") or {}).items())
            out.append(f'<p class="heal">🩹 self-healing attempt {esc(h.get("attempt", "?"))}: '
                       f'<code>{esc(h.get("error_class", "error"))}</code> → {change or "retry"} '
                       f'<span class="sub">{esc(h.get("time", ""))}</span></p>')
        items.append("".join(out) + "</div>")
    body = "".join(items) or '<p class="note">No decisions yet.</p>'
    return f'<div class="card"><h2>Decision log (newest first)</h2>{body}</div>'


def predictions(ctx):
    rows, errs = [], []
    for e in ctx["entries"]:
        for k, c in (e.get("prediction_check") or {}).items():
            if not isinstance(c, dict):
                continue
            rel = c.get("rel_error")
            if isinstance(rel, (int, float)):
                errs.append(abs(rel))
            rows.append(f'<tr><td>{esc(e["id"])}</td><td>{esc(k)}</td><td class="n">{report.fmt(c.get("predicted"))}</td>'
                        f'<td class="n">{report.fmt(c.get("actual"))}</td><td class="n">{report.fmt(c.get("error"))}</td>'
                        f'<td class="n">{f"{100 * rel:+.0f}%" if isinstance(rel, (int, float)) else "–"}</td></tr>')
    if not rows:
        return ""
    mean = f'Mean |relative error| {100 * np.mean(errs):.0f}% over {len(errs)} predictions.' if errs else ""
    return ('<div class="card"><h2>Predicted vs actual</h2><div class="tw"><table><tr><th>Experiment</th>'
            '<th>Quantity</th><th>Predicted</th><th>Actual</th><th>Error</th><th>Rel. error</th></tr>'
            + "".join(rows) + f'</table></div><p class="sub">{mean}</p></div>')


# --------------------------------------------------------------------------------------------
def build(workdir, out, artifact=False):
    """Write the cockpit. artifact=True: a publishable page + figures/ next to it (see docstring)."""
    global ASSET_DIR
    workdir, out = Path(workdir).expanduser(), Path(out).expanduser()
    ASSET_DIR = out.parent / "figures" if artifact else None
    ledger = report.load_json(workdir / "experiments" / "ledger.json") or {}
    entries = ledger.get("experiments") or []
    base = dict(ledger.get("base") or {})
    if not base:  # before the first experiment finishes: epoch 0 of any run is the base model
        for e in entries:
            run = plot_history.read_csv(workdir / e.get("run_dir", "") / "history.csv")
            if run and run["monitors"]:
                base = {f"{k}_f": float(v[0]) for k, v in run["monitors"].items()}
                break
    objective = {**report.DEFAULT_OBJECTIVE, **(ledger.get("objective") or {})}
    done = [e for e in entries if e.get("status") == "done"]
    ctx = {"workdir": workdir, "ledger": ledger, "entries": entries, "base": base, "objective": objective,
           "by_id": {e.get("id"): e for e in entries},
           "best": report.pick_best(done, base, objective) if done and base else None}
    with tempfile.TemporaryDirectory() as tmp:
        figs = safe(figures, ctx, tmp)
        search, monitors = figs if isinstance(figs, tuple) else (figs, "")
        body = (safe(header, ctx)
                + f'<div class="two">{safe(current, ctx)}{search}</div>'  # text next to the scatter
                + monitors                                              # 3 panels: full width
                + safe(predictions, ctx) + safe(ledger_table, ctx) + safe(decision_log, ctx))
    if artifact:
        page = (f'<title>GRACE LoRA Cockpit</title>'
                f'<meta name="claude:viewport-fit-sides" content="cover">{ART_FONTS}'
                f'<style>{ART_CSS}</style><div class="wrap"><div class="top">'
                f'<h1>GRACE LoRA agent cockpit</h1><span class="sub">updated '
                f'{time.strftime("%Y-%m-%d %H:%M")} · republished by the agent after each experiment'
                f'</span></div>{body}</div>')
        ASSET_DIR = None
    else:
        page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<meta http-equiv="refresh" content="{REFRESH_S}"><title>GRACE LoRA agent cockpit</title>'
                f'<style>{CSS}</style></head><body><div class="wrap"><div class="top">'
                f'<h1>GRACE LoRA · agent cockpit</h1><span class="sub">updated {time.strftime("%Y-%m-%d %H:%M:%S")}'
                f' · auto-refresh {REFRESH_S} s · <code>{esc(workdir)}</code></span></div>'
                f'{body}</div></body></html>')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_suffix(".tmp.html")
    tmp_out.write_text(page, encoding="utf-8")
    tmp_out.replace(out)  # atomic: a browser refresh never sees a half-written page
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True)
    p.add_argument("--out", help="HTML path (default: <workdir>/report/cockpit.html)")
    p.add_argument("--watch", type=float, help="rebuild every N seconds until Ctrl+C")
    p.add_argument("--artifact", action="store_true",
                   help="write a publishable Claude-artifact version (+ figures/) instead")
    args = p.parse_args()
    default = ("cockpit_artifact/cockpit.html" if args.artifact else "cockpit.html")
    out = Path(args.out).expanduser() if args.out else Path(args.workdir).expanduser() / "report" / default
    if args.artifact:
        page = build(args.workdir, out, artifact=True)
        figs = sorted((page.parent / "figures").glob("*.png"))
        print(f"Saved: {page}  (publish together with: {', '.join(f'figures/{f.name}' for f in figs)})")
        return
    if not args.watch:
        print(f"Saved: {build(args.workdir, out)}")
        return
    print(f"Watching {args.workdir}: rebuilding {out} every {args.watch:.0f} s (Ctrl+C to stop)")
    try:
        while True:
            try:
                build(args.workdir, out)
            except Exception:  # noqa: BLE001 - keep watching even if one rebuild fails
                traceback.print_exc()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
