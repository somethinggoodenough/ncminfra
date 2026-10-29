"""Pure measurement, trace, provenance and aggregation helpers."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
from pathlib import Path

SCHEMA = "ncm-benchmark-v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(paths: list[Path]) -> dict[str, str]:
    unique = sorted({p.resolve() for p in paths})
    return {str(path): digest(path) for path in unique}


def verify_fingerprint(expected: dict[str, str]) -> None:
    actual = fingerprint([Path(path) for path in expected])
    if actual != expected:
        changed = [p for p in expected if expected[p] != actual.get(p)]
        raise RuntimeError(f"Source changed during the run: {changed}")


def git_info(path: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], text=True,
                              capture_output=True, check=True).stdout
    try:
        return {"head": git("rev-parse", "HEAD").strip(),
                "status": git("status", "--porcelain=v1"),
                "diff": git("diff", "--binary", "HEAD")}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}


def write_json(path: Path, value: object) -> None:
    # The parent run directory is uniquely created, so this does not overwrite
    # any other run. Atomic replacement protects readers of progress files.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def batch_ids(n: int, size: int) -> list[list[int]]:
    if n <= 0 or size <= 0:
        raise ValueError("sample count and batch size must be positive")
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def balanced_order(names: list[str], windows: int, seed: int) -> list[list[str]]:
    if len(names) != len(set(names)) or not names or windows < 1:
        raise ValueError("systems must be nonempty and unique, windows positive")
    base = names[:]
    random.Random(seed).shuffle(base)
    # Two systems alternate AB/BA; three systems rotate ABC/BCA/CAB.
    return [base[w % len(base):] + base[:w % len(base)] for w in range(windows)]


def warmup_stable(times_ms: list[float], group: int = 5, tolerance: float = .05) -> bool:
    if group <= 0 or tolerance < 0:
        raise ValueError("invalid warmup stability settings")
    if len(times_ms) < 2 * group:
        return False
    before, after = statistics.median(times_ms[-2*group:-group]), statistics.median(times_ms[-group:])
    return before > 0 and after > 0 and abs(after / before - 1) <= tolerance


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("invalid percentile input")
    v = sorted(values)
    pos = q * (len(v) - 1)
    low, high = math.floor(pos), math.ceil(pos)
    return v[low] + (v[high] - v[low]) * (pos - low)


def mean_std(values: list[float]) -> dict:
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("invalid measurements")
    return {"median": statistics.median(values),
            "mean": statistics.mean(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.,
            "min": min(values), "max": max(values), "n": len(values)}


def paired_ratios(numerator: list[float], denominator: list[float]) -> list[float]:
    if not numerator or len(numerator) != len(denominator):
        raise ValueError("paired windows are missing or unequal")
    if any(not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b <= 0
           for a, b in zip(numerator, denominator)):
        raise ValueError("paired timings must be finite and positive")
    return [a / b for a, b in zip(numerator, denominator)]


def token_rows_ratio(prefix: int, level: int, is_level0: bool = False) -> float:
    """Per-stack linear row model; excludes KV prefill and non-stack work."""
    if prefix < 0 or level <= 0:
        raise ValueError("invalid lengths")
    return 1.0 if is_level0 else (prefix + level) / level


def arrival_offsets(count: int, rate: float, seed: int) -> list[float]:
    """Deterministic open-loop Poisson arrival trace, in seconds from t=0."""
    if count <= 0 or rate <= 0 or not math.isfinite(rate):
        raise ValueError("invalid arrival trace")
    rng = random.Random(seed)
    result, now = [], 0.0
    for _ in range(count):
        now += rng.expovariate(rate)
        result.append(now)
    return result


def output_token_count(rows: list[list[int]], eos: int, pad: int) -> int:
    count = 0
    for row in rows:
        stop = row.index(eos) if eos in row else len(row)
        count += sum(token != pad and token != eos for token in row[:stop])
    return count
