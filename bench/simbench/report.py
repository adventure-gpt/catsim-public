"""The results as tables (bench_results/report.md) and charts (bench_results/charts/*.png, a light and a dark version).

    python -m simbench report [--out bench_results]

Every table says what each simulator was given and what was scored, and lists every simulator that was run on this
machine, with "not supported" or "not run" said plainly. Charts need matplotlib (`pip install matplotlib`); without it the
tables are still written.
"""
from __future__ import annotations

import json
from pathlib import Path

ORDER = ["catsim", "catsim-py", "rocketsim-v3", "rocketsim-v2", "rocketgpu-cpu", "rocketgpu", "rlmax"]
NAMES = {"catsim": "catSim", "catsim-py": "catSim (Python)", "rocketsim-v3": "RocketSim v3", "rocketsim-v2": "RocketSim v2 (Python)",
         "rocketgpu-cpu": "RocketGPU (CPU)", "rocketgpu": "RocketGPU (CUDA)", "rlmax": "RLMAX"}
# one fixed colour per simulator in every chart (categorical slots in their validated order)
LIGHT = ["#2a78d6", "#6250d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DARK = ["#3987e5", "#9085e9", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

LEVELS = {
    "A": ("A. Physics only", "the game's full internal state at each segment's start (bodies, contact caches, broadphase, every "
          "timer), then the recorded controls and frame timing"),
    "B": ("B. From the public state", "RocketSim v3's CarState and BallState at each segment's start (what a packet shows plus "
          "v3's own timers: physics + guessing the hidden state), then the recorded controls and frame timing"),
    "Bp": ("B, packet fields only", "only the fields an RLBot packet lists at each segment's start (physics + guessing more of "
           "the hidden state), then the recorded controls and frame timing"),
    "C": ("C. Whole game, random choices given", "the state at the first kickoff, then only the recorded controls, frame timing "
          "and the game's random choices (each kickoff and respawn spot)"),
    "D": ("D. Whole game, random stream given", "the state at the first kickoff, the recorded controls and frame timing, and the "
          "random stream's starting state: the simulator picks every kickoff and respawn spot itself"),
}
SUITE_SCORED = ("every car and the ball every tick (up to 120 ticks a segment): bit-exact position, velocity, angular "
                "velocity and rotation; the suite's normalized error (pos/2 uu, vel/3 uu/s, ang vel/0.05, axes/0.05)")


def load(out: Path) -> list:
    recs = []
    for p in sorted(out.glob("**/results.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    return recs


def latest(recs, key):
    d = {}
    for r in recs:
        d[key(r)] = r
    return d


def sim_sort(names):
    return sorted(names, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER))


def nice(n):
    return NAMES.get(n, n)


def fmt_k(x):
    return f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"


def speed_tables(recs) -> list[str]:
    sp = [r for r in recs if r.get("kind") == "speed"]
    if not sp:
        return []
    out = []
    last = latest(sp, lambda r: (r["sim"], r.get("mode"), r.get("arenas"), r.get("threads")))
    for device in ("cpu", "gpu"):
        rows = [r for r in last.values() if r.get("device", "cpu") == device and r.get("mode")]
        if not rows:
            continue
        sims_here = sim_sort({r["sim"] for r in rows})
        m = rows[0]["machine"]
        hw = (", ".join(m.get("gpus") or []) or "GPU") if device == "gpu" else f"{m['cpu']}, {m['threads']} threads"
        out.append(f"### Speed, {device.upper()} simulators ({hw}; {m['os'].split(' (')[0]})\n")
        out.append("Arena-ticks per second, median of the rounds (the spread in brackets). Identical work for every simulator: "
                   "the same start states and controls, 8 ticks per decision, every body read each decision.\n")
        configs = sorted({(r["mode"], r["arenas"], r["threads"]) for r in rows}, key=lambda c: (c[0], c[1], c[2]))
        out.append("| mode | arenas | threads | " + " | ".join(nice(s) for s in sims_here) + " |")
        out.append("|---|---:|---:|" + "---:|" * len(sims_here))
        for mode, arenas, threads in configs:
            cells = []
            for s in sims_here:
                r = last.get((s, mode, arenas, threads))
                if r is None:
                    cells.append("not run")
                elif r.get("status") == "ok":
                    cells.append(f"{fmt_k(r['arena_ticks_per_s'])} ({fmt_k(r['min'])}-{fmt_k(r['max'])})")
                else:
                    cells.append(r.get("status", "?"))
            out.append(f"| {mode} | {arenas} | {threads if device == 'cpu' else '-'} | " + " | ".join(cells) + " |")
        out.append("")
        out.append(versions([r for r in rows if r.get("status") == "ok"]))
    not_run = [r for r in sp if r.get("status") == "not run"]
    if not_run:
        out.append("Not run here: " + "; ".join(f"{nice(r['sim'])} ({r.get('why', '')})" for r in latest(not_run, lambda r: r["sim"]).values()) + "\n")
    return out


def versions(rows) -> str:
    seen = {}
    for r in rows:
        seen.setdefault(r["sim"], r)
    lines = []
    for s in sim_sort(seen):
        r = seen[s]
        ch = "; ".join(r.get("choices") or [])
        lines.append(f"- **{nice(s)}**: {r.get('version') or '?'}. {r.get('about') or ''}" + (f" Adapter: {ch}." if ch else ""))
    return "\n".join(lines) + "\n"


def accuracy_tables(recs) -> list[str]:
    acc = [r for r in recs if r.get("kind") == "accuracy"]
    out = []
    last = latest(acc, lambda r: (r["sim"], r["level"], r.get("suite")))
    for level in ("A", "B", "Bp", "C", "D"):
        rows = [r for r in last.values() if r["level"] == level]
        if not rows:
            continue
        title, given = LEVELS[level]
        for suite in sorted({r.get("suite") for r in rows}):
            rs = [r for r in rows if r.get("suite") == suite]
            out.append(f"### {title} ({suite})\n")
            out.append(f"Given: {given}.  ")
            if level in ("C", "D"):
                out.append("Scored: every compared packet row to the end of the recording (positions, velocities, rotation, as "
                           "the game rounds them); the first row that drifts.\n")
                out.append("| simulator | recordings | exact to the end | packet rows exact | ticks |")
                out.append("|---|---:|---:|---:|---:|")
                for r in sorted(rs, key=lambda r: ORDER.index(r["sim"]) if r["sim"] in ORDER else 99):
                    s = r.get("summary") or {}
                    if r.get("status") != "ok":
                        out.append(f"| {nice(r['sim'])} | {r.get('status')}: {r.get('why', '')} | | | |")
                        continue
                    out.append(f"| {nice(r['sim'])} | {s.get('recordings', '')} | {s.get('exact_to_the_end', '')} | "
                               f"{s.get('exact_rows', 0):,} of {s.get('compared_rows', 0):,} | {s.get('ticks', 0):,} |")
                assists = next((r.get("assists") for r in rs if r.get("assists")), None)
                if assists:
                    out.append("\nAssists (what the recording hands every simulator): " + "; ".join(assists) + ".")
            else:
                out.append(f"Scored: {SUITE_SCORED}.\n")
                out.append("| simulator | exact body-ticks | exact segments | segment pass | mean error k1 / k30 / k120 | body-ticks |")
                out.append("|---|---:|---:|---:|---|---:|")
                for r in sorted(rs, key=lambda r: ORDER.index(r["sim"]) if r["sim"] in ORDER else 99):
                    if r.get("status") != "ok":
                        out.append(f"| {nice(r['sim'])} | {r.get('status')} | {r.get('why', '')} | | | |")
                        continue
                    s = r["summary"]["all"]
                    me = s["mean_error"]
                    out.append(f"| {nice(r['sim'])} | {s['exact_body_ticks']:.4f}% | {s['exact_segments']:.3f}% | {s['segment_pass']:.3f}% | "
                               f"{me['1']:.4f} / {me['30']:.4f} / {me['120']:.4f} | {s['body_ticks']:,} |")
            out.append("")
            out.append(versions([r for r in rs if r.get("status") == "ok"]))
    return out


def rlpr_table(recs) -> list[str]:
    rl = [r for r in recs if r.get("kind") == "rlpr"]
    if not rl:
        return []
    out = ["### RocketSim's own replay metric (rlpr_metric, unchanged)\n",
           "Given: an RLPR tick record at each 120-tick segment's start (bodies, jump and flip state and clocks, flip torque, "
           "boost, wheel contacts and normals, held controls; no contact history), then the recorded controls; or one tick "
           "from every record (one-tick).  Scored: RocketSim's own normalized error per car-tick, its own tolerances "
           "(pass = error under 1).\n",
           "| simulator | 120-tick pass | 120-tick mean norm | one-tick pass | one-tick mean norm | car-ticks |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(latest(rl, lambda r: r["sim"]).values(), key=lambda r: ORDER.index(r["sim"]) if r["sim"] in ORDER else 99):
        if r.get("status") != "ok":
            out.append(f"| {nice(r['sim'])} | {r.get('status')} | {r.get('why', '')} | | | |")
            continue
        a, b = r["results"]["segments_120"]["total"], r["results"]["one_tick"]["total"]
        out.append(f"| {nice(r['sim'])} | {a['pass_pct']:.3f}% | {a['mean_norm']:.4f} | {b['pass_pct']:.3f}% | {b['mean_norm']:.4f} | {a['support']:,} |")
    out.append("")
    return out


def charts(recs, out: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(charts skipped: pip install matplotlib)")
        return []
    made = []
    out.mkdir(parents=True, exist_ok=True)
    sp = [r for r in recs if r.get("kind") == "speed" and r.get("status") == "ok"]
    last = latest(sp, lambda r: (r["sim"], r["mode"], r["arenas"], r["threads"]))
    acc = latest([r for r in recs if r.get("kind") == "accuracy" and r.get("status") == "ok" and r.get("suite") == "full-state suite"],
                 lambda r: (r["sim"], r["level"]))
    for theme, colors, bg, ink, muted, grid in (("light", LIGHT, "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"),
                                               ("dark", DARK, "#1a1a19", "#f0efec", "#c3c2b7", "#3a3936")):
        def axes(title, ylabel):
            fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(grid)
            ax.tick_params(colors=muted, labelsize=9)
            ax.yaxis.grid(True, color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_title(title, color=ink, fontsize=12, loc="left", pad=26)
            ax.set_ylabel(ylabel, color=muted, fontsize=9)
            return fig, ax

        def bars(ax, groups, series, value, label):
            w = 0.8 / max(len(series), 1)
            for k, s in enumerate(series):
                xs, ys = [], []
                for g, grp in enumerate(groups):
                    v = value(s, grp)
                    if v is not None:
                        xs.append(g - 0.4 + w * (k + 0.5))
                        ys.append(v)
                c = colors[ORDER.index(s) % len(colors)] if s in ORDER else colors[-1]
                ax.bar(xs, ys, w * 0.9, color=c, label=nice(s), edgecolor=bg, linewidth=1)
                for x, y in zip(xs, ys):
                    ax.text(x, y, label(y), ha="center", va="bottom", fontsize=7, color=muted)
            ax.set_xticks(range(len(groups)))
            top = ax.get_ylim()[1]
            ax.set_ylim(0, top * 1.08)
            leg = ax.legend(frameon=False, fontsize=8, ncol=min(len(series), 4), loc="lower left", bbox_to_anchor=(0, 1.0),
                            borderaxespad=0.2, handlelength=1.2)
            for t in leg.get_texts():
                t.set_color(ink)

        # CPU speed at the training-style load point (1024 arenas, all threads)
        cpu = [r for r in last.values() if r.get("device", "cpu") == "cpu" and r["arenas"] == 1024]
        if cpu:
            series = sim_sort({r["sim"] for r in cpu})
            modes = ["1v1", "2v2", "3v3"]
            m = cpu[0]["machine"]
            fig, ax = axes(f"CPU speed: 1,024 arenas, {cpu[0]['threads']} threads ({m['cpu'].split('@')[0].strip()})",
                           "million arena-ticks per second")
            get = lambda s, mode: next((r["arena_ticks_per_s"] / 1e6 for r in cpu if r["sim"] == s and r["mode"] == mode), None)
            bars(ax, modes, series, get, lambda y: f"{y:.2f}")
            ax.set_xticklabels(modes, color=ink)
            p = out / f"speed_cpu_{theme}.png"
            fig.tight_layout()
            fig.savefig(p, facecolor=bg)
            plt.close(fig)
            made.append(p)
        gpu = [r for r in last.values() if r.get("device") == "gpu"]
        if gpu:
            series = sim_sort({r["sim"] for r in gpu})
            sizes = sorted({r["arenas"] for r in gpu})
            for mode in ("1v1", "3v3"):
                rows = [r for r in gpu if r["mode"] == mode]
                if not rows:
                    continue
                m = rows[0]["machine"]
                fig, ax = axes(f"GPU speed, {mode} ({', '.join(x.split(',')[0] for x in m.get('gpus') or ['GPU'])})",
                               "million arena-ticks per second")
                get = lambda s, n: next((r["arena_ticks_per_s"] / 1e6 for r in rows if r["sim"] == s and r["arenas"] == n), None)
                bars(ax, [str(n) for n in sizes], series, lambda s, n: get(s, int(n)), lambda y: f"{y:.1f}")
                ax.set_xticklabels([f"{n:,} arenas" for n in sizes], color=ink)
                p = out / f"speed_gpu_{mode}_{theme}.png"
                fig.tight_layout()
                fig.savefig(p, facecolor=bg)
                plt.close(fig)
                made.append(p)
        if acc:
            levels = [lv for lv in ("A", "B") if any(k[1] == lv for k in acc)]
            series = sim_sort({k[0] for k in acc})
            fig, ax = axes("Accuracy on the full-state suite: bit-exact body-ticks", "% of body-ticks exact")
            get = lambda s, lv: acc[(s, lv)]["summary"]["all"]["exact_body_ticks"] if (s, lv) in acc else None
            bars(ax, levels, series, get, lambda y: f"{y:.1f}")
            ax.set_xticklabels([LEVELS[lv][0] for lv in levels], color=ink, fontsize=9)
            ax.set_ylim(0, 110)
            p = out / f"accuracy_{theme}.png"
            fig.tight_layout()
            fig.savefig(p, facecolor=bg)
            plt.close(fig)
            made.append(p)
    return made


def summary_table(recs) -> list[str]:
    """One row per measure, one column per simulator: the headline numbers of every benchmark that was run."""
    sims_all = sim_sort({r["sim"] for r in recs if r.get("sim")})
    sims_all = [s for s in sims_all if s != "catsim-py"] + (["catsim-py"] if "catsim-py" in sims_all else [])
    rows = []
    sp = latest([r for r in recs if r.get("kind") == "speed"], lambda r: (r["sim"], r.get("mode"), r.get("arenas"), r.get("threads")))
    big = [r for r in sp.values() if r.get("status") == "ok" and r.get("device", "cpu") == "cpu" and r.get("arenas") == 1024]
    if big:
        t = big[0]["threads"]
        for mode in ("1v1", "3v3"):
            cells = []
            for s in sims_all:
                r = sp.get((s, mode, 1024, t))
                dev = next((x.get("device") for x in sp.values() if x["sim"] == s and x.get("device")), "cpu")
                cells.append(fmt_k(r["arena_ticks_per_s"]) if r and r.get("status") == "ok" else ("GPU simulator" if dev == "gpu" else "-"))
            rows.append((f"CPU speed, {mode}, 1,024 arenas, {t} threads (arena-ticks/s)", cells))
        for mode in ("1v1", "3v3"):
            cells = []
            for s in sims_all:
                r = sp.get((s, mode, 1, 1))
                cells.append(fmt_k(r["arena_ticks_per_s"]) if r and r.get("status") == "ok" else "-")
            rows.append((f"CPU speed, {mode}, one arena, one thread", cells))
    gpu = [r for r in sp.values() if r.get("status") == "ok" and r.get("device") == "gpu"]
    if gpu:
        n = max(r["arenas"] for r in gpu)
        for mode in ("1v1", "3v3"):
            cells = []
            for s in sims_all:
                r = sp.get((s, mode, n, 1))
                cells.append(fmt_k(r["arena_ticks_per_s"]) if r and r.get("status") == "ok" else "-")
            rows.append((f"GPU speed, {mode}, {n:,} arenas", cells))
    acc = latest([r for r in recs if r.get("kind") == "accuracy"], lambda r: (r["sim"], r["level"], r.get("suite")))
    for level, label in (("A", "A. physics only: exact body-ticks / segment pass"), ("B", "B. public state: exact body-ticks / segment pass"),
                         ("Bp", "B, packet fields only: exact / pass")):
        if not any(k[1] == level for k in acc):
            continue
        cells = []
        for s in sims_all:
            r = acc.get((s, level, "full-state suite"))
            if r is None:
                cells.append("-")
            elif r.get("status") != "ok":
                cells.append(r.get("status", "?"))
            else:
                a = r["summary"]["all"]
                cells.append(f"{a['exact_body_ticks']:.2f}% / {a['segment_pass']:.2f}%")
        rows.append((label, cells))
    rl = latest([r for r in recs if r.get("kind") == "rlpr"], lambda r: r["sim"])
    if rl:
        cells = []
        for s in sims_all:
            r = rl.get(s)
            if r is None:
                cells.append("-")
            elif r.get("status") != "ok":
                cells.append(r.get("status", "?"))
            else:
                a = r["results"]["segments_120"]["total"]
                cells.append(f"{a['pass_pct']:.2f}% / {a['mean_norm']:.2f}")
        rows.append(("RocketSim's rlpr_metric, 120-tick segments: pass / mean norm", cells))
    for level, label in (("C", "C. whole games, choices given: exact to the end"), ("D", "D. whole games, random stream given: exact to the end")):
        if not any(k[1] == level for k in acc):
            continue
        cells = []
        for s in sims_all:
            r = acc.get((s, level, "our recordings"))
            if r is None:
                cells.append("-")
            elif r.get("status") != "ok":
                cells.append(r.get("status", "?"))
            else:
                a = r["summary"]
                cells.append(f"{a['exact_to_the_end']} of {a['recordings']} games")
        rows.append((label, cells))
    if not rows:
        return []
    out = ["| | " + " | ".join(nice(s) for s in sims_all) + " |", "|---|" + "---|" * len(sims_all)]
    out += [f"| {label} | " + " | ".join(cells) + " |" for label, cells in rows]
    return out + [""]


def main(out: Path) -> None:
    recs = load(Path(out))
    if not recs:
        raise SystemExit(f"no results in {out}: run a benchmark first")
    text = ["# Benchmark results\n"]
    ms = {json.dumps(r.get("machine"), sort_keys=True) for r in recs if r.get("machine")}
    for m in ms:
        m = json.loads(m)
        text.append(f"Machine: {m['cpu']}, {m['threads']} threads, {m['ram_gb']} GB, {m['os'].split(' (')[0]}"
                    + (f"; {', '.join(m['gpus'])}" if m.get("gpus") else "") + "\n")
    text += ["## Summary\n"] + summary_table(recs) + ["## Details\n"] + speed_tables(recs) + accuracy_tables(recs) + rlpr_table(recs)
    p = Path(out) / "report.md"
    p.write_text("\n".join(text), encoding="utf-8")
    made = charts(recs, Path(out) / "charts")
    print("\n".join(text))
    print(f"\nwritten: {p}" + "".join(f"\n         {c}" for c in made))
