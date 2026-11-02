"""Project NVTX stack launches onto an Nsight Systems SQLite kernel trace.

Usage: python -m benchmarks.nsys --trace capture.sqlite benchmark_run --output new_dir
Repeat --trace to put both configurations in the same table. No CUDA dependency.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sqlite3
import statistics

from benchmarks.core import digest, write_json

PREFIX = "ncm_stack:"
IDENTITY = ("system", "batch_size", "cfg", "window", "profile_batch", "stack")
GROUP = ("trace", "system", "batch_size", "cfg", "stack", "token_rows")


def is_stack(name):
    return name == "cache_prefill" or (
        name.startswith("level_") and name.removeprefix("level_").isdigit())


def key(row):
    return tuple(row[field] for field in IDENTITY)


def expected_stacks(run):
    """Require a complete, hash-verified profile and bind trace labels to it."""
    result = json.loads((run / "run.json").read_text())
    if result["status"] != "COMPLETE" or result.get("provenance_valid") is not True:
        raise ValueError(f"{run}: requires COMPLETE with valid provenance")
    for name in ("manifest", "windows", "quality"):
        suffix = ".json" if name == "manifest" else ".jsonl"
        if digest(run / (name + suffix)) != result[name + "_sha256"]:
            raise ValueError(f"{run}: {name} hash mismatch")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["mode"] != "profile" or not manifest["protocol"].get("nvtx"):
        raise ValueError(f"{run}: requires a --mode profile --nvtx run")
    planned = {(system, cell["batch_size"], cell["cfg"], window)
               for cell in manifest["plan"] for window, order in enumerate(cell["orders"])
               for system in order}
    seen, expected = set(), {}
    for line in (run / "windows.jsonl").read_text().splitlines():
        record = json.loads(line)
        cell = tuple(record[field] for field in IDENTITY[:4])
        if cell in seen or record["status"] != "PASS" or not record.get("probe_output_exact"):
            raise ValueError(f"{run}: duplicate or failed profile window {cell}")
        seen.add(cell)
        if len(record["batches"]) != manifest["protocol"]["profile_batches_per_window"]:
            raise ValueError(f"{run}: incomplete profile window {cell}")
        for i, batch in enumerate(record["batches"]):
            names = {span["name"] for span in batch["spans"] if is_stack(span["name"])}
            required = {f"level_{level}" for level in range(len(manifest["workload"]["levels"]))}
            if record["system"] == "cache":
                required.add("cache_prefill")
            if not required <= names:
                raise ValueError(f"{run}: missing stack spans in {cell}, batch {i}")
            for span in batch["spans"]:
                if not is_stack(span["name"]):
                    continue
                identity = {**dict(zip(IDENTITY[:4], cell)), "batch_size": len(batch["indices"]),
                            "profile_batch": i, "stack": span["name"]}
                if key(identity) in expected:
                    raise ValueError(f"{run}: duplicate stack {identity}")
                expected[key(identity)] = {**identity, "input_shape": span["input_shape"],
                    "event_gpu_ms": span["gpu_ms"], "cpu_enqueue_ms": span["cpu_enqueue_ms"]}
    if seen != planned or not expected:
        raise ValueError(f"{run}: profile windows do not match the manifest plan")
    return result["manifest_sha256"], expected


def require_columns(db, table, required):
    columns = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
    if not set(required) <= columns:
        raise ValueError(f"Unsupported or missing {table}: need columns {sorted(set(required) - columns)}")


def project(db, manifest_sha, expected):
    """Match API calls by CPU thread/range, then kernels by process/correlation.

    globalTid encodes the thread in its low 24 bits. Keep the higher bits to
    avoid confusing equal correlation IDs in different processes/VMs.
    CPU range end is deliberately NOT used as the GPU timestamp cutoff.
    """
    require_columns(db, "StringIds", ("id", "value"))
    require_columns(db, "NVTX_EVENTS", ("start", "end", "globalTid", "text", "textId"))
    kernel_fields = ("start", "end", "globalPid", "correlationId", "deviceId",
                     "contextId", "streamId", "demangledName")
    require_columns(db, "CUPTI_ACTIVITY_KIND_KERNEL", kernel_fields)
    threads, spans, seen = defaultdict(list), [], set()
    query = """SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) AS label
               FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
               WHERE COALESCE(n.text, s.value) LIKE 'ncm_stack:%' ORDER BY n.start"""
    for row in db.execute(query):
        label = json.loads(row["label"][len(PREFIX):])
        if label.get("manifest_sha256") != manifest_sha:
            raise ValueError("SQLite contains stacks from a different benchmark manifest")
        identity = key(label)
        if identity not in expected or identity in seen:
            raise ValueError(f"Unexpected/duplicate NVTX stack: {identity}")
        if label["input_shape"] != expected[identity]["input_shape"]:
            raise ValueError(f"NVTX input shape mismatch: {identity}")
        if row["end"] is None or row["end"] <= row["start"]:
            raise ValueError(f"Truncated NVTX range: {identity}")
        shape = label["input_shape"]
        if not shape or len(shape) < 2 or any(not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError(f"Invalid stack input shape: {shape}")
        seen.add(identity)
        span = {**expected[identity], "token_rows": math.prod(shape[:-1]),
                "cpu_start_ns": row["start"], "cpu_end_ns": row["end"], "kernels": []}
        thread = threads[row["globalTid"]]
        if thread and thread[-1]["cpu_end_ns"] > row["start"]:
            raise ValueError("Overlapping stack NVTX ranges are not supported")
        thread.append(span)
        spans.append(span)
    if seen != set(expected):
        raise ValueError(f"Trace missing {len(set(expected) - seen)} expected stack ranges")
    starts = {tid: [span["cpu_start_ns"] for span in group] for tid, group in threads.items()}
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    api_tables = [table for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
                  if table in tables]
    if not api_tables:
        raise ValueError("No CUDA API trace; collect with --trace=cuda,nvtx")
    launches = {}
    for table in api_tables:
        require_columns(db, table, ("start", "end", "globalTid", "correlationId", "nameId"))
        for tid, group in threads.items():
            query = f"""SELECT a.*, s.value AS api_name FROM {table} a
                        LEFT JOIN StringIds s ON a.nameId = s.id
                        WHERE a.globalTid = ? AND a.start >= ? AND a.end <= ?"""
            for api in db.execute(query, (tid, group[0]["cpu_start_ns"], group[-1]["cpu_end_ns"])):
                index = bisect_right(starts[tid], api["start"]) - 1
                if index < 0 or api["end"] > group[index]["cpu_end_ns"]:
                    continue
                if api["correlationId"] is None:
                    continue
                identity = (tid >> 24, api["correlationId"])
                previous = launches.get(identity)
                if previous is not None:
                    if previous[0] is not group[index]:
                        raise ValueError("Ambiguous CUDA correlation ID across stacks")
                    # A runtime launch can wrap a driver call. Count its kernels once.
                    continue
                launches[identity] = (group[index], dict(api))
    query = """SELECT k.*, s.value AS kernel_name FROM CUPTI_ACTIVITY_KIND_KERNEL k
               LEFT JOIN StringIds s ON k.demangledName = s.id
               WHERE k.start >= ? ORDER BY k.start"""
    for kernel in db.execute(query, (min(span["cpu_start_ns"] for span in spans),)):
        if kernel["globalPid"] is None:
            raise ValueError("Kernel trace lacks process IDs; cannot correlate safely")
        match = launches.get((kernel["globalPid"] >> 24, kernel["correlationId"]))
        if match is None:
            continue
        span, api = match
        if kernel["end"] <= kernel["start"] or not kernel["kernel_name"]:
            raise ValueError("Incomplete kernel timing/name in trace")
        span["kernels"].append({"start_ns": kernel["start"], "end_ns": kernel["end"],
            "kernel": kernel["kernel_name"], "stream": (kernel["globalPid"], kernel["deviceId"],
                kernel["contextId"], kernel["streamId"]),
            "api": api["api_name"], "api_start_ns": api["start"], "api_end_ns": api["end"]})
    return spans


def decompose(spans, trace):
    rows, kernels, gaps = [], [], []
    for span in spans:
        events = span["kernels"]
        if not events:
            raise ValueError(f"No kernels for {key(span)}; CUDA trace is missing or incomplete")
        if len({event["stream"] for event in events}) != 1:
            raise ValueError(f"Multiple GPU streams for {key(span)}; single-stream decomposition required")
        identity = {"trace": str(trace), **{field: span[field] for field in IDENTITY},
                    "token_rows": span["token_rows"]}
        kernel_ns, gap_ns, pre_api_ns = 0, 0, 0
        for i, event in enumerate(events):
            duration = event["end_ns"] - event["start_ns"]
            kernel_ns += duration
            kernels.append({**identity, "ordinal": i, **event, "kernel_us": duration / 1e3,
                "launch_api_us": (event["api_end_ns"] - event["api_start_ns"]) / 1e3,
                "gpu_start_minus_api_end_us": (event["start_ns"] - event["api_end_ns"]) / 1e3})
            if i:
                previous = events[i - 1]
                gap = event["start_ns"] - previous["end_ns"]
                if gap < 0:
                    raise ValueError(f"Overlapping kernels for {key(span)}; cannot sum gaps")
                # This is a timeline partition, not a causal attribution to CPU/driver.
                pre_api = max(0, min(event["start_ns"], event["api_start_ns"]) - previous["end_ns"])
                gap_ns += gap
                pre_api_ns += pre_api
                gaps.append({**identity, "next_ordinal": i, "previous_kernel": previous["kernel"],
                    "next_kernel": event["kernel"], "start_ns": previous["end_ns"],
                    "end_ns": event["start_ns"], "gap_us": gap / 1e3,
                    "before_next_api_us": pre_api / 1e3, "after_next_api_start_us": (gap - pre_api) / 1e3})
        gpu_ns = events[-1]["end_ns"] - events[0]["start_ns"]
        if kernel_ns + gap_ns != gpu_ns:
            raise ValueError("Kernel/gap decomposition does not close")
        rows.append({**identity, "kernel_count": len(events), "kernel_ms": kernel_ns / 1e6,
            "gap_ms": gap_ns / 1e6, "gpu_span_ms": gpu_ns / 1e6,
            "mean_kernel_us": kernel_ns / len(events) / 1e3,
            "gap_before_next_api_ms": pre_api_ns / 1e6,
            "gap_after_next_api_start_ms": (gap_ns - pre_api_ns) / 1e6,
            "event_gpu_ms": span["event_gpu_ms"], "cpu_enqueue_ms": span["cpu_enqueue_ms"],
            "event_minus_kernel_span_ms": span["event_gpu_ms"] - gpu_ns / 1e6,
            "nvtx_cpu_ms": (span["cpu_end_ns"] - span["cpu_start_ns"]) / 1e6})
    return rows, kernels, gaps


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in GROUP)].append(row)
    result = []
    for identity, group in groups.items():
        row = dict(zip(GROUP, identity))
        row.update(instances=len(group), windows=len({item["window"] for item in group}),
                   kernel_count_min=min(item["kernel_count"] for item in group),
                   kernel_count_max=max(item["kernel_count"] for item in group))
        for field in ("kernel_count", "kernel_ms", "gap_ms", "gpu_span_ms", "event_gpu_ms",
                      "event_minus_kernel_span_ms", "cpu_enqueue_ms",
                      "gap_before_next_api_ms", "gap_after_next_api_start_ms"):
            row[field + "_mean"] = statistics.mean(item[field] for item in group)
        row["pooled_kernel_us"] = sum(item["kernel_ms"] for item in group) * 1000 / sum(
            item["kernel_count"] for item in group)
        result.append(row)
    return result


def summarize_kernels(kernels, summary):
    instances = {tuple(row[field] for field in GROUP): row["instances"] for row in summary}
    groups = defaultdict(lambda: [0, 0])
    for event in kernels:
        identity = tuple(event[field] for field in GROUP) + (event["kernel"],)
        total = groups[identity]
        total[0] += 1
        total[1] += event["end_ns"] - event["start_ns"]
    rows = []
    for identity, (count, duration_ns) in groups.items():
        samples = instances[identity[:-1]]
        rows.append({**dict(zip((*GROUP, "kernel"), identity)), "stack_instances": samples,
            "total_calls": count, "calls_per_stack": count / samples,
            "kernel_ms_per_stack": duration_ns / samples / 1e6,
            "mean_kernel_us": duration_ns / count / 1e3})
    return sorted(rows, key=lambda row: (tuple(row[field] for field in GROUP), -row["kernel_ms_per_stack"]))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", nargs=2, action="append", required=True, type=Path,
                    metavar=("SQLITE", "RUN"), help="SQLite export and its benchmark run directory")
    ap.add_argument("--output", type=Path, required=True, help="new output directory")
    a = ap.parse_args(argv)
    rows, kernels, gaps, inputs = [], [], [], []
    for trace, run in a.trace:
        manifest_sha, expected = expected_stacks(run)
        with sqlite3.connect(trace.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            stacks, calls, pauses = decompose(project(db, manifest_sha, expected), trace)
        rows.extend(stacks)
        kernels.extend(calls)
        gaps.extend(pauses)
        inputs.append({"sqlite": str(trace.resolve()), "sqlite_sha256": digest(trace),
                       "run": str(run.resolve()), "manifest_sha256": manifest_sha,
                       "run_sha256": digest(run / "run.json")})
    summary = summarize(rows)
    a.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("stacks", rows), ("summary", summary), ("kernels", kernels),
                       ("kernel_summary", summarize_kernels(kernels, summary)), ("gaps", gaps)):
        write_csv(a.output / f"{name}.csv", data)
    lines = ["# Nsight Systems per-stack breakdown", "",
        "Arithmetic means per stack invocation; N × X / 1000 + Σ gap = GPU span (ms).",
        "GPU span runs from the first kernel start to the last kernel end; it excludes boundary wait.",
        "Σ gap is time without an attributed stack kernel, not proof of GPU-wide idle or CPU launch cost.",
        "kernel_summary.csv ranks kernel names by time per stack; kernels.csv and gaps.csv retain individual timestamps.", "",
        "| Trace | System | B | CFG | Stack | Rows | Samples | N | X (µs) | Σ kernel (ms) | Σ gap (ms) | GPU span (ms) | Event span (ms) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {Path(row['trace']).name} | {row['system']} | {row['batch_size']} | {row['cfg']:g} | "
            f"{row['stack']} | {row['token_rows']} | {row['instances']} | {row['kernel_count_mean']:.1f} | "
            f"{row['pooled_kernel_us']:.3f} | {row['kernel_ms_mean']:.4f} | {row['gap_ms_mean']:.4f} | "
            f"{row['gpu_span_ms_mean']:.4f} | {row['event_gpu_ms_mean']:.4f} |")
    (a.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(a.output / "manifest.json", {"schema": "ncm-nsys-stack-v1", "inputs": inputs,
        "analyzer_sha256": digest(Path(__file__)), "aggregation": "arithmetic mean per invocation",
        "gap_definition": "sum of next kernel start minus previous kernel end, one stream",
        "limitations": "Trace overhead, memory operations, other streams and boundary waits require timeline inspection.",
        "outputs": {path.name: digest(path) for path in a.output.iterdir() if path.is_file()}})
    print(a.output / "report.md")


if __name__ == "__main__":
    main()
