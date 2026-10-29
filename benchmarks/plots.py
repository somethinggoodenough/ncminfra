"""Render benchmark figures exclusively from recorded, validated measurements."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.core import SCHEMA, digest, mean_std, paired_ratios, token_rows_ratio, write_json

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"eager": "#475569", "cache": "#2563eb", "candidate": "#e87638"}
LABELS = {"eager": "Eager", "cache": "Fixed upstream cache", "candidate": "Candidate"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "savefig.bbox": "tight"})


def read_jsonl(path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def read_run(directory, mode):
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    if manifest.get("schema") != SCHEMA or run.get("schema") != SCHEMA or manifest.get("mode") != mode:
        raise ValueError(f"{directory}: wrong schema or mode (expected {mode})")
    for key, filename in (("manifest_sha256", "manifest.json"), ("windows_sha256", "windows.jsonl"),
                          ("quality_sha256", "quality.jsonl")):
        if run.get(key) != digest(directory / filename):
            raise ValueError(f"{directory}: {filename} hash mismatch")
    if run.get("status") not in ("COMPLETE", "PARTIAL", "FAILED"):
        raise ValueError(f"{directory}: unrecognized run status")
    if run.get("provenance_valid") is not True:
        raise ValueError(f"{directory}: asset provenance was not valid at the end of the run")
    return {"directory": directory, "manifest": manifest, "run": run,
            "windows": read_jsonl(directory / "windows.jsonl"),
            "quality": read_jsonl(directory / "quality.jsonl")}


def qualified_cells(data):
    needed = len(data["manifest"]["workload"]["sample_ids"])
    groups = defaultdict(list)
    for row in data["quality"]:
        groups[(row["cfg"], row["batch_size"])].append(row)
    return {key for key, rows in groups.items()
            if rows and all(row.get("passed") and row.get("coverage") == needed for row in rows)}


def grouped_windows(data):
    valid_cells = qualified_cells(data)
    groups = defaultdict(list)
    for row in data["windows"]:
        if row.get("status") != "PASS" or (row["cfg"], row["batch_size"]) not in valid_cells:
            continue
        key = (row["system"], row["cfg"], row["batch_size"], row.get("arrival_rate_per_second"))
        groups[key].append(row)
    expected = data["manifest"]["protocol"]["windows_per_cell"]
    # Partial, duplicate or mismatched windows cannot become a completed curve.
    return {key: sorted(rows, key=lambda row: row["window"]) for key, rows in groups.items()
            if len(rows) == expected and sorted(r["window"] for r in rows) == list(range(expected))}


def save(fig, output, name):
    fig.subplots_adjust(bottom=.22)
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}")
    plt.close(fig)


def step0_plots(path, output):
    data = json.loads(path.read_text())
    levels = data["per_level"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for branch, label, color in (("cond", "Conditional", "#2563eb"),
                                 ("null", "Null", "#e87638"),
                                 ("guided", "CFG combined", "#7c3aed")):
        ax.plot([r["level_index"] for r in levels],
                [100*r[branch]["relative_l2"] for r in levels],
                marker="o", linewidth=2, label=label, color=color)
    ax.set(xlabel="Concept level (zero-based)", ylabel="Relative L2 error (%)",
           title="Step 0: legacy cache vs eager by level")
    ax.set_xticks([r["level_index"] for r in levels])
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=3)
    fig.text(.12, .01, "WMT14 · CFG=1 · CPU/FP32 · nine prompts · free-running trajectories", fontsize=8, color="#475569")
    save(fig, output, "step0_relative_l2")

    prompts = data["prompts"]
    x = np.asarray([p["effective_token_length"] for p in prompts], dtype=float)
    y = np.asarray([p["level1"]["cond"]["rmse"] for p in prompts], dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(x, y, s=55, color="#2563eb", zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(0, 96, 100)
    ax.plot(line_x, slope*line_x + intercept, color="#e87638", linewidth=1.8,
            label=f"Linear fit · Pearson r = {r:.3f}")
    for p, xval, yval in zip(prompts, x, y):
        ax.annotate(str(p["index"]), (xval, yval), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Effective prompt length (tokens)", ylabel="Conditional logits RMSE",
           title="Step 0: first affected level vs prompt length", xlim=(-3, 100), ylim=bottom_zero(y))
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.text(.12, .01, "Level 1 · WMT14 · CFG=1 · CPU/FP32 · labels are dataset row IDs", fontsize=8, color="#475569")
    save(fig, output, "step0_level1_rmse_vs_prompt")
    return {"prompts": len(prompts), "pearson_r": r, "source_sha256": digest(path)}


def bottom_zero(values):
    return (0, max(values)*1.18 if max(values) else 1)


def comparable(data, groups, cfg, batch, rate=None):
    """Require the same complete set of paired windows for each requested system."""
    return all((system, cfg, batch, rate) in groups for system in data["manifest"]["systems"])


def e2e_plots(data, output):
    groups = grouped_windows(data)
    systems = data["manifest"]["systems"]
    cfgs = sorted({p["cfg"] for p in data["manifest"]["plan"]})
    batches = sorted({p["batch_size"] for p in data["manifest"]["plan"]})
    cfgs = [cfg for cfg in cfgs if any(comparable(data, groups, cfg, batch) for batch in batches)]
    if not cfgs:
        return []
    figures = []
    for kind in ("throughput", "memory"):
        fig, axes = plt.subplots(1, len(cfgs), figsize=(6.3*len(cfgs), 4.7), squeeze=False)
        for ax, cfg in zip(axes[0], cfgs):
            eligible = [batch for batch in batches if comparable(data, groups, cfg, batch)]
            for system in systems:
                ys, errors = [], []
                for batch in eligible:
                    rows = groups[(system, cfg, batch, None)]
                    values = [r["requests_per_second"] if kind == "throughput"
                              else r["peak_allocated_bytes"] / 2**30 for r in rows]
                    stats = mean_std(values)
                    ys.append(stats["median"])
                    errors.append(stats["stddev"])
                label = LABELS[system]
                if kind == "memory" and eligible:
                    label += f" (max measured B={max(eligible)})"
                ax.errorbar(eligible, ys, yerr=errors, color=COLORS[system], marker="o",
                            linewidth=2, capsize=3, label=label)
            ax.set_xscale("log", base=2)
            ax.set_xticks(eligible, labels=[str(b) for b in eligible])
            ax.set_xlabel("Batch size (log scale)")
            ax.set_title(f"CFG = {cfg:g}")
            ax.grid(axis="y", alpha=.25)
            if kind == "memory":
                ax.axhline(8, linestyle="--", color="#b91c1c", label="8 GiB reference")
                failures = [r for r in data["windows"] if r.get("cfg") == cfg and r.get("status") == "OOM"]
                if failures:
                    ax.text(.02, .03, "OOM: " + ", ".join(f"{r.get('system')} B={r['batch_size']}" for r in failures),
                            transform=ax.transAxes, fontsize=8, color="#b91c1c")
                ax.set_ylabel("Peak allocated GPU memory (GiB)")
            else:
                ax.set_ylabel("Completed requests / second")
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle("Throughput vs batch" if kind == "throughput" else "Peak memory vs batch")
        note = "Medians; error bars are window standard deviations. Complete paired cells only."
        if data["manifest"]["protocol"].get("exploratory"):
            note += " Fewer than five windows: exploratory."
        fig.text(.08, .02, note,
                 fontsize=8, color="#475569")
        save(fig, output, kind + "_vs_batch")
        figures.append(kind + "_vs_batch")
    return figures


def profile_window_medians(row):
    spans = defaultdict(list)
    for batch in row["batches"]:
        for span in batch["spans"]:
            spans[span["name"]].append(span["gpu_ms"])
    return {name: statistics.median(values) for name, values in spans.items()}


def profile_plots(data, output, cfg, batch):
    groups = grouped_windows(data)
    present = [s for s in data["manifest"]["systems"] if (s, cfg, batch, None) in groups]
    if "eager" not in present or len(present) < 2:
        return []
    level_lengths = data["manifest"]["workload"]["levels"]
    names = [f"level_{i}" for i in range(len(level_lengths))]
    windows = {s: [profile_window_medians(r) for r in groups[(s, cfg, batch, None)]] for s in present}
    if any(any(not all(name in w for name in [*names, "reconstruction", "decoder"]) for w in records)
           for records in windows.values()):
        raise ValueError("profile spans are incomplete")
    categories = names + (["cache_prefill"] if any("cache_prefill" in w for rows in windows.values() for w in rows) else []) + ["reconstruction", "decoder"]
    xpos = np.arange(len(categories))
    width = .8 / len(present)
    fig, ax = plt.subplots(figsize=(14, 5))
    for idx, system in enumerate(present):
        values, errors = [], []
        for name in categories:
            times = [w[name] for w in windows[system] if name in w]
            values.append(statistics.median(times) if times else 0.)
            errors.append(statistics.stdev(times) if len(times) > 1 else 0.)
        ax.bar(xpos + (idx - (len(present)-1)/2)*width, values, width,
               yerr=errors, capsize=2, color=COLORS[system], label=LABELS[system])
    ax.set_xticks(xpos, [f"L{i}" for i in range(len(names))] +
                  (["KV prefill"] if "cache_prefill" in categories else []) + ["Rebuild\nlatent", "Decoder\n+ argmax"])
    ax.set(ylabel="GPU stream span (ms)", title=f"Per-level stack and decoder time · CFG={cfg:g}, B={batch}")
    ax.legend(frameon=False, ncol=len(present))
    ax.grid(axis="y", alpha=.25)
    missing = [LABELS[s] for s in data["manifest"]["systems"] if s not in present]
    note = "Diagnostic CUDA events, median across paired windows; error bars are window standard deviations."
    if missing:
        note += " Missing profile: " + ", ".join(missing) + "."
    if data["manifest"]["protocol"].get("exploratory"):
        note += " Fewer than five windows: exploratory."
    fig.text(.08, .02, note,
             fontsize=8, color="#475569")
    save(fig, output, "per_level_time")
    figures = ["per_level_time"]

    total_names = [f"level_total_{i}" for i in range(len(level_lengths))]
    total_systems = [s for s in present if all(all(name in w for name in total_names)
                                               for w in windows[s])]
    if "eager" in total_systems and len(total_systems) >= 2:
        categories = total_names + ["reconstruction", "decoder"]
        xpos = np.arange(len(categories))
        width = .8 / len(total_systems)
        fig, ax = plt.subplots(figsize=(13.5, 5))
        for idx, system in enumerate(total_systems):
            values, errors = [], []
            for name in categories:
                times = [w[name] for w in windows[system]]
                values.append(statistics.median(times))
                errors.append(statistics.stdev(times) if len(times) > 1 else 0.)
            ax.bar(xpos + (idx - (len(total_systems)-1)/2)*width, values, width,
                   yerr=errors, capsize=2, color=COLORS[system], label=LABELS[system])
        ax.set_xticks(xpos, [f"L{i}" for i in range(len(total_names))] + ["Rebuild\nlatent", "Decoder\n+ argmax"])
        ax.set(ylabel="GPU stream span (ms)",
               title=f"Per-level total time · CFG={cfg:g}, B={batch}")
        ax.legend(frameon=False, ncol=len(total_systems))
        ax.grid(axis="y", alpha=.25)
        note = "Level 1 includes cache prefill; reconstruction is shown separately."
        missing_totals = [LABELS[s] for s in present if s not in total_systems]
        if missing_totals:
            note += " Missing total-level spans: " + ", ".join(missing_totals) + "."
        fig.text(.08, .02, note, fontsize=8, color="#475569")
        save(fig, output, "per_level_total_time")
        figures.append("per_level_total_time")

    prefix = data["manifest"]["workload"]["prefix_length"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    predicted_values, measured_values = [], []
    missing_rows = []
    for system in [s for s in present if s != "eager"]:
        if system == "cache":
            predicted = [token_rows_ratio(prefix, length, i == 0) for i, length in enumerate(level_lengths)]
        else:
            rows = data["manifest"].get("candidate_profile_token_rows")
            if rows is None:
                missing_rows.append(LABELS[system])
                continue
            predicted = [(prefix if i == 0 else prefix + length) / rows[i]
                         for i, length in enumerate(level_lengths)]
        measured = [statistics.median(paired_ratios([w[name] for w in windows["eager"]],
                                                     [w[name] for w in windows[system]]))
                    for name in names]
        ax.scatter(predicted, measured, s=38, color=COLORS[system], label=LABELS[system])
        for i, (x, y) in enumerate(zip(predicted, measured)):
            ax.annotate(f"L{i}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        predicted_values.extend(predicted)
        measured_values.extend(measured)
    if predicted_values:
        extent = max(predicted_values + measured_values) * 1.05
        ax.plot([0, extent], [0, extent], "--", color="#64748b", label="y = x")
        ax.set(xlabel="Predicted stack speedup from token rows", ylabel="Measured stack speedup",
               title=f"Predicted vs measured · CFG={cfg:g}, B={batch}", xlim=(0, extent), ylim=(0, extent))
        ax.legend(frameon=False)
        ax.grid(alpha=.25)
        note = "Row model excludes cache prefill, decoder, and launch overhead; L0 predicts 1×."
        if missing_rows:
            note += " Missing row model: " + ", ".join(missing_rows) + "."
        fig.text(.1, .01, note,
                 fontsize=8, color="#475569")
        save(fig, output, "predicted_vs_measured")
        figures.append("predicted_vs_measured")
    else:
        plt.close(fig)
    return figures


def serving_plot(data, output, cfg, batch):
    groups = grouped_windows(data)
    systems = data["manifest"]["systems"]
    rates = sorted({p["arrival_rate_per_second"] for p in data["manifest"]["plan"]
                    if p["cfg"] == cfg and p["batch_size"] == batch})
    rates = [rate for rate in rates if comparable(data, groups, cfg, batch, rate)]
    if not rates:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), squeeze=False)
    exploratory = False
    for ax, stat in zip(axes[0], ("p50", "p99")):
        for system in systems:
            x, y, err = [], [], []
            for rate in rates:
                records = groups[(system, cfg, batch, rate)]
                exploratory |= any(r.get("p99_exploratory") for r in records)
                x.append(statistics.median(r["requests_per_second"] for r in records))
                summary = mean_std([r["latency_ms"][stat] for r in records])
                y.append(summary["median"])
                err.append(summary["stddev"])
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linewidth=2,
                        color=COLORS[system], label=LABELS[system])
        ax.set(xlabel="Completed requests / second", ylabel=f"{stat} request latency (ms)",
               title=f"{stat} · CFG={cfg:g}, max B={batch}")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    note = "Open-loop scheduled arrivals to output; median across windows, error bars = window stddev."
    if exploratory:
        note += " p99 exploratory: fewer than 10,000 requests/window."
    if data["manifest"]["protocol"].get("exploratory"):
        note += " Fewer than five windows: exploratory."
    fig.text(.08, .01, note, fontsize=8, color="#475569")
    fig.suptitle("Serving latency vs throughput")
    save(fig, output, "latency_vs_throughput")
    return ["latency_vs_throughput"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="new figure directory")
    ap.add_argument("--step0", type=Path, default=ROOT / "reports/raw/2026-10-27-wmt14-kv-cache-probe.json")
    ap.add_argument("--e2e", type=Path)
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--serving", type=Path)
    ap.add_argument("--plot-batch", type=int, default=4)
    ap.add_argument("--plot-cfg", type=float, default=1.)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    figures, inputs = [], {}
    if a.step0:
        inputs["step0"] = step0_plots(a.step0, a.out)
        figures += ["step0_relative_l2", "step0_level1_rmse_vs_prompt"]
    for key in ("e2e", "profile", "serving"):
        path = getattr(a, key)
        if path is None:
            continue
        data = read_run(path, key)
        inputs[key] = {"path": str(path.resolve()), "manifest_sha256": digest(path / "manifest.json"),
                       "run_sha256": digest(path / "run.json"), "run_status": data["run"]["status"]}
        if key == "e2e":
            figures += e2e_plots(data, a.out)
        elif key == "profile":
            figures += profile_plots(data, a.out, a.plot_cfg, a.plot_batch)
        else:
            figures += serving_plot(data, a.out, a.plot_cfg, a.plot_batch)
    figure_hashes = {p.name: digest(p) for p in a.out.iterdir() if p.is_file()}
    summary = {"schema": SCHEMA, "command": sys.argv, "matplotlib": matplotlib.__version__,
               "source_sha256": {"plots.py": digest(Path(__file__)),
                                 "core.py": digest(Path(__file__).with_name("core.py"))},
               "inputs": inputs, "figures": figures, "figure_sha256": figure_hashes,
               "plot_cfg": a.plot_cfg, "plot_batch": a.plot_batch}
    write_json(a.out / "figure_manifest.json", summary)
    for name in figures:
        print(f"{a.out / name}.png and .svg")
    if not a.e2e or not a.profile or not a.serving:
        print("Missing performance-run inputs are intentionally not plotted.")


if __name__ == "__main__":
    main()
