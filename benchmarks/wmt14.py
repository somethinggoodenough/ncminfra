"""WMT14 CUDA benchmark. Run with ``python -m benchmarks.wmt14``.

The default experiment measures the patched upstream model's eager and cache
paths. An optional candidate adapter exposes build(runtime).generate(...).
Timing, correctness and profiling are separate phases; no speedup is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import itertools
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from omegaconf import OmegaConf

from benchmarks.core import (SCHEMA, arrival_offsets, balanced_order, batch_ids, digest, fingerprint,
                             git_info, mean_std, output_token_count, percentile,
                             verify_fingerprint, warmup_stable, write_json)

ROOT = Path(__file__).resolve().parents[1]
NCM = ROOT / "upstream/ncm"
sys.path.insert(0, str(NCM))


def csv_values(text, kind):
    result = [kind(item) for item in text.split(",") if item]
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("comma-separated values must be nonempty and unique")
    return result


def args_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True,
                    help="new output directory (must not already exist)")
    ap.add_argument("--mode", choices=["e2e", "profile", "serving"], default="e2e")
    ap.add_argument("--systems", default="eager,cache",
                    help="eager,cache and optional candidate; each appears once")
    ap.add_argument("--candidate", type=Path, help="adapter .py implementing build(runtime)")
    ap.add_argument("--candidate-source", type=Path, action="append", default=[],
                    help="additional source files used by the candidate; may repeat")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--cfg", default="0,1")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--profile-batches", type=int, default=16)
    ap.add_argument("--arrival-rates", default="", help="serving offered request rates per second")
    ap.add_argument("--requests", type=int, default=10000, help="requests per serving window")
    ap.add_argument("--warmup-min", type=int, default=10)
    ap.add_argument("--warmup-max", type=int, default=50)
    ap.add_argument("--warmup-group", type=int, default=5)
    ap.add_argument("--warmup-tolerance", type=float, default=.05)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", type=Path, default=ROOT / "artifacts/wmt14/config.yaml")
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/wmt14")
    ap.add_argument("--sources", type=Path, default=ROOT / "data/wmt14/sources.pickle")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/qwen3-0.6b-tokenizer")
    return ap


class Runtime:
    def __init__(self, a):
        # Imported only after the driver has fingerprinted these sources.
        from models.next_concept import NextConceptTransformer
        from tokenization.vqvae_tokenizer import VQVAETokenizer

        self.device = torch.device(a.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        self.cfg = OmegaConf.load(a.config)
        if self.cfg.task != "wmt14" or int(self.cfg.model.prefix_len) != 96:
            raise ValueError("this driver requires WMT14 with prefix_len=96")
        if self.cfg.checkpoints.generator != "generator.pt" or self.cfg.checkpoints.tokenizer != "tokenizer.pt":
            raise ValueError("expected WMT14 generator.pt and tokenizer.pt checkpoint names")
        self.cfg.tokenizer.pre_tokenizer.model_id = str(a.tokenizer.resolve())
        self.cfg.tokenizer.vqvae.checkpoint_path = str((a.checkpoints / "tokenizer.pt").resolve())
        self.tk = VQVAETokenizer.from_config(self.cfg.tokenizer)
        self.tk.vqvae = self.tk.vqvae.to(self.device).eval()
        self.net = NextConceptTransformer(self.tk.vqvae, self.cfg.model).to(self.device).eval()
        state = torch.load(a.checkpoints / "generator.pt", map_location="cpu", weights_only=False)
        self.net.load_state_dict(state["model"], strict=True)
        del state
        self.pad = self.tk.pre_tokenizer.pad_token_id
        self.eos = self.tk.pre_tokenizer.eos_token_id
        self.P = int(self.net.prefix_len)
        self.levels = list(map(int, self.net.levels))
        if self.levels != [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]:
            raise ValueError(f"unexpected WMT14 level schedule: {self.levels}")
        if self.net.prefix_pad_id != self.pad:
            raise ValueError("model and tokenizer pad IDs do not match")
        if not self.net.predict_level0 or self.net.query_level0:
            raise ValueError("expected WMT14 predict-level0 configuration")
        with a.sources.open("rb") as file:
            sources = pickle.load(file)
        if not 0 < a.n <= len(sources):
            raise ValueError(f"n must be 1..{len(sources)}")
        ids = [self.tk.pre_tokenizer.encode(s) for s in sources[:a.n]]
        self.lengths = [len(row) for row in ids]
        if any(not row for row in ids):
            raise ValueError("empty prompts are not supported")
        self.prefixes = torch.tensor([(row[:self.P] + [self.pad] * self.P)[:self.P]
                                      for row in ids], dtype=torch.long)
        self.batches = {}
        self._tensor_versions = self.tensor_versions()

    def tensor_versions(self):
        return [(id(tensor), tensor._version) for module in (self.net, self.tk.vqvae)
                for tensor in itertools.chain(module.parameters(), module.buffers())]

    def check_parameters(self):
        if self.tensor_versions() != self._tensor_versions:
            raise RuntimeError("model parameters or buffers changed during the benchmark")
        if any(m.training for module in (self.net, self.tk.vqvae) for m in module.modules()):
            raise RuntimeError("model left evaluation mode")

    def check_state(self):
        self.check_parameters()
        if getattr(self.net, "_cfg", None) is not None:
            raise RuntimeError("candidate left CFG state active")
        if any(b.attn.caching or b.attn.cached_k is not None or b.attn.cached_v is not None
               for b in self.net.transformer.blocks):
            raise RuntimeError("candidate left upstream KV cache active")

    def cpu_batch(self, indices):
        return self.prefixes[indices]

    def generate(self, system, prefix, cfg, candidate):
        if system == "candidate":
            return candidate.generate(prefix, cfg, self.eos)
        return self.net.generate(None, prefix, temperature=1., top_k=1, top_p=0.,
                                 use_kv_cache=(system == "cache"),
                                 cfg_scale=cfg, cfg_null_eos=self.eos)

    @torch.no_grad()
    def run(self, system, cpu_prefix, cfg, candidate, capture_codes=False, profile=None):
        # Boundary: CPU tokenized, padded IDs -> CPU final token IDs. D2H of
        # codes only occurs in quality/profile validation, never in E2E timing.
        self.last_system = system
        prefix = cpu_prefix.to(self.device)
        if profile:
            profile.install(self.net, system)
        try:
            if profile and system == "candidate":
                codes = candidate.generate_profiled(prefix, cfg, self.eos, profile)
            else:
                codes = self.generate(system, prefix, cfg, candidate)
            expected = (len(cpu_prefix), sum(self.levels))
            if codes.shape != expected or codes.dtype != torch.long or codes.device != self.device:
                raise ValueError(f"{system} returned invalid codes: {codes.shape}, {codes.dtype}, {codes.device}")
            if profile:
                profile.start_reconstruction()
            if system == "candidate" and callable(getattr(candidate, "reconstruct", None)):
                z_final = candidate.reconstruct(codes)
            else:
                _, zlevels = self.tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
                z_final = zlevels[-1]
            if profile:
                profile.end_reconstruction()
            if profile:
                profile.start_decoder()
            if system == "candidate" and callable(getattr(candidate, "decode_ids", None)):
                ids = candidate.decode_ids(z_final)
            else:
                ids = self.tk.vqvae.decoder(z_final).argmax(-1)
            if profile:
                profile.end_decoder()
            if ids.shape != (len(cpu_prefix), self.levels[-1]) or ids.dtype != torch.long or ids.device != self.device:
                raise ValueError(f"{system} returned invalid final IDs: {ids.shape}, {ids.dtype}, {ids.device}")
            ids = ids.to("cpu")
            captured = codes.to("cpu") if capture_codes else None
            return ids, captured
        finally:
            if profile:
                profile.remove()


class NativeProfile:
    """Events around generator stacks and decoder; one CUDA stream, no sync per span."""
    def __init__(self):
        self.handles, self.stacks = [], []
        self.active = None
        self.decoder = None
        self.reconstruction = None
        self.level_totals = []
        self.open_spans = []
        self.net = None

    @staticmethod
    def event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def install(self, net, system):
        self.system, self.levels = system, list(map(int, net.levels))
        if system == "candidate":
            return
        self.net = net
        self.sample_in_instance = "_sample_level" in net.__dict__
        self.saved_sample = net.__dict__.get("_sample_level")
        original_sample = net._sample_level
        self.level_start = self.event()
        self.level_cpu_start_ns = time.perf_counter_ns()

        def sample_and_mark(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            cpu_end_ns = time.perf_counter_ns()
            end = self.event()
            self.level_totals.append({"name": f"level_total_{len(self.level_totals)}",
                "start": self.level_start, "end": end,
                "cpu_start_ns": self.level_cpu_start_ns, "cpu_end_ns": cpu_end_ns})
            self.level_start = end
            self.level_cpu_start_ns = time.perf_counter_ns()
            return result

        net._sample_level = sample_and_mark
        blocks = net.transformer.blocks

        def before(module, inputs):
            if self.active is not None:
                raise RuntimeError("overlapping transformer stacks")
            self.active = {"start": self.event(), "cpu_start_ns": time.perf_counter_ns(),
                           "input_shape": list(inputs[0].shape)}

        def after(module, inputs, output):
            if self.active is None:
                raise RuntimeError("missing stack start")
            self.active["cpu_end_ns"] = time.perf_counter_ns()
            self.active["end"] = self.event()
            self.stacks.append(self.active)
            self.active = None

        try:
            self.handles.append(blocks[0].register_forward_pre_hook(before))
            self.handles.append(blocks[-1].register_forward_hook(after))
        except Exception:
            self.remove()
            raise

    @contextmanager
    def span(self, name, input_shape=None):
        """Candidate adapter's properly nested GPU scope, on the current stream."""
        if self.system != "candidate":
            raise RuntimeError("manual spans are for candidate profiling")
        item = {"name": str(name), "start": self.event(),
                "cpu_start_ns": time.perf_counter_ns(),
                "input_shape": list(input_shape) if input_shape is not None else None}
        self.open_spans.append(item)
        try:
            yield
        finally:
            if not self.open_spans or self.open_spans[-1] is not item:
                raise RuntimeError("candidate profile spans must close in nested order")
            self.open_spans.pop()
            item["cpu_end_ns"] = time.perf_counter_ns()
            item["end"] = self.event()
            self.stacks.append(item)

    def start_decoder(self):
        self.decoder = {"start": self.event(), "cpu_start_ns": time.perf_counter_ns()}

    def end_decoder(self):
        self.decoder["cpu_end_ns"] = time.perf_counter_ns()
        self.decoder["end"] = self.event()

    def start_reconstruction(self):
        self.reconstruction = {"start": self.event(), "cpu_start_ns": time.perf_counter_ns()}

    def end_reconstruction(self):
        self.reconstruction["cpu_end_ns"] = time.perf_counter_ns()
        self.reconstruction["end"] = self.event()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.net is not None:
            if self.sample_in_instance:
                self.net._sample_level = self.saved_sample
            else:
                del self.net._sample_level
            self.net = None

    def result(self):
        expected = len(self.levels) + (self.system == "cache")
        if self.active is not None or self.open_spans or self.decoder is None or self.reconstruction is None:
            raise RuntimeError("unclosed profile span or missing decoder")
        if self.system == "candidate":
            names = [item["name"] for item in self.stacks]
            required = [f"level_{i}" for i in range(len(self.levels))]
            if any(names.count(name) != 1 for name in required) or len(names) != len(set(names)):
                raise RuntimeError(f"candidate must record each level exactly once: {names}")
        elif len(self.stacks) != expected or len(self.level_totals) != len(self.levels):
            raise RuntimeError(f"unexpected stack call count: {len(self.stacks)}, expected {expected}")
        spans = []
        for index, item in enumerate(self.stacks):
            if self.system == "candidate":
                name = item["name"]
            elif self.system == "cache" and index == 1:
                name = "cache_prefill"
            else:
                name = f"level_{index - (self.system == 'cache' and index > 1)}"
            spans.append({"name": name, "input_shape": item["input_shape"],
                          "gpu_ms": item["start"].elapsed_time(item["end"]),
                          "cpu_enqueue_ms": (item["cpu_end_ns"] - item["cpu_start_ns"]) / 1e6})
        for item in self.level_totals:
            spans.append({"name": item["name"],
                          "gpu_ms": item["start"].elapsed_time(item["end"]),
                          "cpu_enqueue_ms": (item["cpu_end_ns"] - item["cpu_start_ns"]) / 1e6})
        for name, item in (("reconstruction", self.reconstruction), ("decoder", self.decoder)):
            spans.append({"name": name, "gpu_ms": item["start"].elapsed_time(item["end"]),
                          "cpu_enqueue_ms": (item["cpu_end_ns"] - item["cpu_start_ns"]) / 1e6})
        return spans


def load_candidate(path, runtime):
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("ncm_benchmark_candidate", path.resolve())
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load candidate {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = module.build(runtime)
    if not callable(getattr(candidate, "generate", None)):
        raise ValueError("candidate must expose generate(prefix, cfg, eos)")
    runtime.check_state()
    return candidate


def candidate_activate(runtime, candidate):
    if getattr(runtime, "candidate_active", False):
        raise RuntimeError("candidate is already active")
    if candidate is not None and callable(getattr(candidate, "activate", None)):
        candidate.activate()
    runtime.candidate_active = candidate is not None
    runtime.check_state()


def candidate_deactivate(runtime, candidate, force=False):
    if candidate is not None and (force or getattr(runtime, "candidate_active", False)) and callable(getattr(candidate, "deactivate", None)):
        candidate.deactivate()
    runtime.candidate_active = False
    torch.cuda.synchronize(runtime.device)
    runtime.check_state()
    live = torch.cuda.memory_allocated(runtime.device)
    if live > runtime.baseline_allocated:
        raise RuntimeError(f"candidate left {live - runtime.baseline_allocated} GPU bytes resident; "
                           "provide deactivate() to release candidate-owned state before other systems")


def candidate_cleanup(runtime, candidate):
    # Preserve an in-flight failure (especially OOM); teardown errors should
    # only replace a successful result. The cell stops on the original error.
    original_error = sys.exc_info()[0] is not None
    try:
        candidate_deactivate(runtime, candidate)
    except Exception:
        if not original_error:
            raise


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup(runtime, a, system, cfg, batches, candidate, seed):
    samples = []
    for i in range(a.warmup_max):
        set_seed(seed + i)
        torch.cuda.synchronize(runtime.device)
        start = time.perf_counter_ns()
        runtime.run(system, runtime.cpu_batch(batches[i % len(batches)]), cfg, candidate)
        torch.cuda.synchronize(runtime.device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
        if len(samples) >= a.warmup_min and warmup_stable(samples, a.warmup_group, a.warmup_tolerance):
            return {"stable": True, "batch_ms": samples}
    return {"stable": False, "batch_ms": samples}


def quality(runtime, a, systems, batches, cfg, candidate):
    """Full-coverage reference comparison; excluded from performance windows."""
    results = {system: {"code_mismatches": 0, "id_mismatches": 0} for system in systems}
    records, reference = [], []
    for batch_index, indices in enumerate(batches):
        prefix = runtime.cpu_batch(indices)
        set_seed(a.seed + batch_index)
        ref_ids, ref_codes = runtime.run("eager", prefix, cfg, candidate, capture_codes=True)
        reference.append((ref_ids, ref_codes))
        records.append({"indices": indices, "reference_codes_sha256": tensor_sha(ref_codes),
                        "reference_ids_sha256": tensor_sha(ref_ids), "comparisons": {}})
        if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(batches):
            print(f"quality cfg={cfg:g} B={a.current_batch} eager {batch_index + 1}/{len(batches)} batches", flush=True)
    for system in systems:
        if system == "eager":
            continue
        runtime.last_system = system
        if system == "candidate":
            candidate_activate(runtime, candidate)
        try:
            for batch_index, indices in enumerate(batches):
                ref_ids, ref_codes = reference[batch_index]
                set_seed(a.seed + batch_index)
                ids, codes = runtime.run(system, runtime.cpu_batch(indices), cfg, candidate, capture_codes=True)
                code_diff = codes.ne(ref_codes).sum().item()
                id_diff = ids.ne(ref_ids).sum().item()
                results[system]["code_mismatches"] += code_diff
                results[system]["id_mismatches"] += id_diff
                records[batch_index]["comparisons"][system] = {"code_mismatches": code_diff, "id_mismatches": id_diff,
                    "codes_sha256": tensor_sha(codes), "ids_sha256": tensor_sha(ids)}
                if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(batches):
                    print(f"quality cfg={cfg:g} B={a.current_batch} {system} {batch_index + 1}/{len(batches)} batches", flush=True)
        finally:
            if system == "candidate":
                candidate_cleanup(runtime, candidate)
    runtime.check_state()
    return {"coverage": sum(map(len, batches)), "results": results, "batches": records,
            "passed": all(v["code_mismatches"] == 0 and v["id_mismatches"] == 0
                          for v in results.values())}


def run_window(runtime, a, system, cfg, batches, candidate, window):
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    records, seen, outputs = [], [], []
    started_utc = datetime.now(timezone.utc).isoformat()
    begin_window = time.perf_counter_ns()
    for i, indices in enumerate(batches):
        set_seed(a.seed + i)
        torch.cuda.synchronize(runtime.device)
        cpu_prefix = runtime.cpu_batch(indices)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_ns = time.perf_counter_ns()
        start_event.record()
        ids, _ = runtime.run(system, cpu_prefix, cfg, candidate)
        end_event.record()
        torch.cuda.synchronize(runtime.device)
        end_ns = time.perf_counter_ns()
        records.append({"indices": indices, "batch_size": len(indices),
                        "wall_ms": (end_ns - start_ns) / 1e6,
                        "gpu_stream_ms": start_event.elapsed_time(end_event)})
        outputs.append(ids)
        seen.extend(indices)
    end_window = time.perf_counter_ns()
    elapsed = (end_window - begin_window) / 1e9
    if seen != list(range(a.n)) or elapsed <= 0:
        raise RuntimeError("invalid request coverage or time")
    for record, ids in zip(records, outputs):
        record["output_tokens"] = output_token_count(ids.tolist(), runtime.eos, runtime.pad)
        record["ids_sha256"] = tensor_sha(ids)
    runtime.check_state()
    return {"status": "PASS", "system": system, "cfg": cfg, "batch_size": a.current_batch,
            "window": window, "samples": len(seen), "wall_seconds": elapsed,
            "started_utc": started_utc, "start_monotonic_ns": begin_window,
            "end_monotonic_ns": end_window,
            "requests_per_second": len(seen) / elapsed,
            "output_tokens_per_second": sum(r["output_tokens"] for r in records) / elapsed,
            "latency_ms": {"p50": percentile([r["wall_ms"] for r in records], .5),
                           "p99": percentile([r["wall_ms"] for r in records], .99),
                           "unit": "batch, synchronous offline; no queue"},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
            "batches": records}


def profile_window(runtime, a, system, cfg, batches, candidate, window):
    if system == "candidate" and not callable(getattr(candidate, "generate_profiled", None)):
        return {"status": "UNSUPPORTED_PROFILE", "reason": "candidate has no generate_profiled method",
                "system": system, "cfg": cfg, "batch_size": a.current_batch, "window": window}
    records = []
    for i in range(a.profile_batches):
        prefix = runtime.cpu_batch(batches[i % len(batches)])
        set_seed(a.seed + i)
        plain_ids, plain_codes = runtime.run(system, prefix, cfg, candidate, capture_codes=True)
        set_seed(a.seed + i)
        probe = NativeProfile()
        ids, codes = runtime.run(system, prefix, cfg, candidate, capture_codes=True, profile=probe)
        torch.cuda.synchronize(runtime.device)
        if not torch.equal(ids, plain_ids) or not torch.equal(codes, plain_codes):
            raise RuntimeError("profile instrumentation changed outputs")
        records.append({"indices": batches[i % len(batches)], "spans": probe.result()})
    return {"status": "PASS", "system": system, "cfg": cfg, "batch_size": a.current_batch,
            "window": window, "probe_output_exact": True, "batches": records}


def serving_window(runtime, a, system, cfg, candidate, window, rate):
    """Single worker, predetermined open-loop arrivals and real queued execution.

    Codes and IDs are copied to CPU inside the measured service boundary to
    allow an exact, post-window eager replay of the actual dynamic batches.
    """
    offsets = arrival_offsets(a.requests, rate, a.seed + window)
    scheduled_ns = [round(x * 1e9) for x in offsets]
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter_ns()
    next_request = 0
    batches = []
    while next_request < a.requests:
        now_ns = time.perf_counter_ns() - start
        if now_ns < scheduled_ns[next_request]:
            time.sleep(min((scheduled_ns[next_request] - now_ns) / 1e9, .05))
            continue
        end = next_request
        while (end < a.requests and end - next_request < a.current_batch and
               scheduled_ns[end] <= now_ns):
            end += 1
        requests = list(range(next_request, end))
        source_ids = [i % a.n for i in requests]
        cpu_prefix = runtime.cpu_batch(source_ids)
        seed = a.seed + window * 1000000 + len(batches)
        set_seed(seed)
        torch.cuda.synchronize(runtime.device)
        dispatched_ns = time.perf_counter_ns() - start
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        ids, codes = runtime.run(system, cpu_prefix, cfg, candidate, capture_codes=True)
        event_end.record()
        torch.cuda.synchronize(runtime.device)
        completed_ns = time.perf_counter_ns() - start
        batches.append({"request_ids": requests, "source_ids": source_ids, "seed": seed,
                        "dispatched_ns": dispatched_ns, "completed_ns": completed_ns,
                        "gpu_stream_ms": event_start.elapsed_time(event_end),
                        "_codes": codes, "_ids": ids})
        next_request = end
    peak_alloc = torch.cuda.max_memory_allocated(runtime.device)
    peak_reserved = torch.cuda.max_memory_reserved(runtime.device)
    duration = (batches[-1]["completed_ns"] - scheduled_ns[0]) / 1e9
    request_rows = []
    for batch in batches:
        rows = batch["_ids"].tolist()
        for j, request_id in enumerate(batch["request_ids"]):
            arrival_ns = scheduled_ns[request_id]
            request_rows.append({"request_id": request_id, "source_id": batch["source_ids"][j],
                "arrival_ns": arrival_ns, "dispatch_ns": batch["dispatched_ns"],
                "completion_ns": batch["completed_ns"],
                "latency_ms": (batch["completed_ns"] - arrival_ns) / 1e6,
                "queue_ms": (batch["dispatched_ns"] - arrival_ns) / 1e6,
                "output_tokens": output_token_count([rows[j]], runtime.eos, runtime.pad)})
    if duration <= 0 or [x["request_id"] for x in request_rows] != list(range(a.requests)):
        raise RuntimeError("invalid serving request coverage or time")
    if system == "candidate":
        # The request trace and outputs are already on CPU. Release any
        # candidate-only working set before the untimed eager replay.
        candidate_deactivate(runtime, candidate)
    mismatched_codes = mismatched_ids = 0
    for batch in batches:
        set_seed(batch["seed"])
        ref_ids, ref_codes = runtime.run("eager", runtime.cpu_batch(batch["source_ids"]), cfg,
                                         candidate, capture_codes=True)
        codes = batch.pop("_codes")
        ids = batch.pop("_ids")
        mismatched_codes += codes.ne(ref_codes).sum().item()
        mismatched_ids += ids.ne(ref_ids).sum().item()
        batch.update(codes_sha256=tensor_sha(codes), ids_sha256=tensor_sha(ids))
    runtime.check_state()
    latencies = [row["latency_ms"] for row in request_rows]
    return {"status": "PASS" if mismatched_codes == mismatched_ids == 0 else "QUALITY_FAIL",
        "system": system, "cfg": cfg, "batch_size": a.current_batch, "window": window,
        "started_utc": started_utc, "start_monotonic_ns": start,
        "arrival_rate_per_second": rate, "arrival_process": "seeded Poisson, open loop",
        "boundary": "scheduled arrival to completed final IDs and captured codes on CPU",
        "samples": a.requests, "duration_seconds": duration,
        "requests_per_second": a.requests / duration,
        "latency_ms": {"p50": percentile(latencies, .5), "p99": percentile(latencies, .99)},
        "p99_exploratory": a.requests < 10000,
        "peak_allocated_bytes": peak_alloc, "peak_reserved_bytes": peak_reserved,
        "code_mismatches": mismatched_codes, "id_mismatches": mismatched_ids,
        "requests": request_rows, "batches": batches}


def download_metadata(path):
    """Read optional Hugging Face local-dir metadata; file hashes remain authoritative."""
    path = path.resolve()
    for base in (path.parent, path.parent.parent):
        relative = path.relative_to(base)
        meta = base / ".cache/huggingface/download" / (str(relative) + ".metadata")
        if meta.is_file():
            lines = meta.read_text().splitlines()
            return {"revision": lines[0] if lines else None,
                    "etag": lines[1] if len(lines) > 1 else None,
                    "metadata_sha256": digest(meta),
                    "note": "local download metadata; file SHA-256 recorded separately"}
    return None


def manifest(a, runtime, systems, candidate, candidate_path, plan, sources, assets):
    try:
        nvidia = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                                          "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        nvidia = f"unavailable: {exc}"
    candidate_rows = None
    if candidate is not None and callable(getattr(candidate, "profile_token_rows", None)):
        candidate_rows = list(candidate.profile_token_rows(runtime.P, runtime.levels))
        if len(candidate_rows) != len(runtime.levels) or any(not isinstance(n, int) or n <= 0 for n in candidate_rows):
            raise ValueError("candidate.profile_token_rows must return one positive integer per level")
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "mode": a.mode, "systems": systems, "candidate_path": str(candidate_path) if candidate_path else None,
        "plan": plan, "candidate_profile_token_rows": candidate_rows,
        "device": str(runtime.device), "gpu": torch.cuda.get_device_name(runtime.device),
        "gpu_properties": str(torch.cuda.get_device_properties(runtime.device)), "nvidia_smi": nvidia,
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "sdpa": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                 "math": torch.backends.cuda.math_sdp_enabled(),
                 "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled()},
        "dtype": "float32", "autocast": False, "num_threads": torch.get_num_threads(),
        "packages": {x: importlib.metadata.version(x) for x in ("torch", "numpy", "transformers", "omegaconf", "hydra-core")},
        "environment_packages": sorted(
            ({"name": d.metadata.get("Name", "unknown"), "version": d.version}
             for d in importlib.metadata.distributions()),
            key=lambda row: (row["name"].lower(), row["version"])),
        "root_git": git_info(ROOT), "upstream_git": git_info(NCM),
        "source_fingerprint": sources, "asset_hashes": assets,
        "download_metadata": {path: download_metadata(Path(path)) for path in assets},
        "baseline_resident_allocated_bytes": runtime.baseline_allocated,
        "model_configuration": OmegaConf.to_container(runtime.cfg, resolve=True),
        "workload": {"sample_ids": list(range(a.n)), "token_lengths": runtime.lengths,
            "truncated": sum(x > runtime.P for x in runtime.lengths),
            "padding_tokens": sum(max(0, runtime.P - x) for x in runtime.lengths),
            "prefix_ids_sha256": tensor_sha(runtime.prefixes),
            "prefix_length": runtime.P, "levels": runtime.levels, "eos": runtime.eos, "pad": runtime.pad,
            "temperature": 1., "top_k": 1, "top_p": 0., "seed": a.seed},
        "protocol": {"windows_per_cell": a.windows, "order": "seeded rotating paired windows",
            "warmup_min": a.warmup_min, "warmup_max": a.warmup_max,
            "warmup_group": a.warmup_group, "warmup_tolerance": a.warmup_tolerance,
            "exploratory": a.windows < 5,
            "window_process": "shared process/model; each window individually warmed; candidate activation outside timing, deactivated between systems",
            "e2e_boundary": "pretokenized CPU padded IDs through H2D, generate, reconstruct, decoder, argmax and D2H final IDs",
            "gpu_event_scope": "single current CUDA stream, includes idle gaps; not kernel-only or multistream timing",
            "e2e_probe": False, "quality": "all selected samples, exact concept codes and token IDs vs eager",
            "window_overhead": "wall throughput includes seed resets, synchronization and lightweight batch bookkeeping; hashes/counts computed after timing",
            "profile": "per-level total and stack ranges overlap; level 1 total includes cache prefill; reconstruction and decoder+argmax separate",
            "serving": "single worker, open-loop scheduled arrivals, queue and service; code and ID D2H included, eager replay excluded"}}


def main(argv=None):
    a = args_parser().parse_args(argv)
    systems = csv_values(a.systems, str)
    batches = csv_values(a.batches, int)
    cfgs = csv_values(a.cfg, float)
    rates = csv_values(a.arrival_rates, float) if a.arrival_rates else []
    if not set(systems) <= {"eager", "cache", "candidate"} or "eager" not in systems:
        raise ValueError("systems must include eager and may include cache,candidate")
    if ("candidate" in systems) != (a.candidate is not None):
        raise ValueError("candidate system and --candidate must be supplied together")
    if any(x <= 0 for x in batches) or any(x not in (0., 1.) for x in cfgs):
        raise ValueError("batch sizes must be positive and CFG must be 0 or 1")
    if a.windows < 1 or a.n < 1 or a.profile_batches < 1 or a.threads < 1:
        raise ValueError("window, sample, profile and thread counts must be positive")
    if a.warmup_group < 1 or a.warmup_min < 2*a.warmup_group or a.warmup_max < a.warmup_min:
        raise ValueError("warmup requires max >= min >= 2*group")
    if not 0 <= a.warmup_tolerance < 1:
        raise ValueError("warmup tolerance must be in [0,1)")
    if (a.mode == "serving" and (not rates or any(not math.isfinite(x) or x <= 0 for x in rates) or a.requests < 1)) or (a.mode != "serving" and rates):
        raise ValueError("serving requires positive --arrival-rates and --requests; other modes do not use rates")
    if a.candidate_source and not a.candidate:
        raise ValueError("--candidate-source requires --candidate")
    if not torch.cuda.is_available() or not a.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for benchmark measurements; no CPU timings are substituted")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    # Explicit settings, same for every implementation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = list((ROOT / "benchmarks").glob("*.py"))
    source_files += [p for base in (NCM / "models", NCM / "tokenization", NCM / "utils")
                     for p in base.rglob("*.py")]
    if a.candidate:
        source_files.append(a.candidate.resolve())
        source_files.extend(p.resolve() for p in a.candidate_source)
    source_hashes = fingerprint(source_files)
    assets = [a.config, a.sources, a.checkpoints / "generator.pt", a.checkpoints / "tokenizer.pt"]
    assets.extend(p for p in a.tokenizer.iterdir() if p.is_file())
    asset_hashes = fingerprint(assets)
    started = time.perf_counter()
    runtime = Runtime(a)
    torch.cuda.synchronize(runtime.device)
    runtime.baseline_allocated = torch.cuda.memory_allocated(runtime.device)
    candidate = load_candidate(a.candidate, runtime)
    candidate_deactivate(runtime, candidate, force=True)
    torch.cuda.synchronize(runtime.device)
    load_seconds = time.perf_counter() - started
    verify_fingerprint(source_hashes)
    verify_fingerprint(asset_hashes)
    plan = []
    for cfg in cfgs:
        for batch in batches:
            for rate in (rates if a.mode == "serving" else [None]):
                plan.append({"cfg": cfg, "batch_size": batch, "arrival_rate_per_second": rate,
                             "orders": balanced_order(systems, a.windows, a.seed + batch + int(cfg * 1000))})
    m = manifest(a, runtime, systems, candidate, a.candidate, plan, source_hashes, asset_hashes)
    candidate_deactivate(runtime, candidate, force=True)
    m["load_seconds"] = load_seconds
    write_json(output / "manifest.json", m)
    window_path, quality_path = output / "windows.jsonl", output / "quality.jsonl"
    failed = False
    unsupported = False
    with window_path.open("x") as window_file, quality_path.open("x") as quality_file:
        for cell in plan:
            a.current_batch = cell["batch_size"]
            cfg = cell["cfg"]
            ids_batches = batch_ids(a.n, a.current_batch)
            current_system, current_window = None, None
            runtime.last_system = None
            try:
                verify_fingerprint(source_hashes)
                q = quality(runtime, a, systems, ids_batches, cfg, candidate)
                q.update(cfg=cfg, batch_size=a.current_batch)
                quality_file.write(json.dumps(q, allow_nan=False) + "\n")
                quality_file.flush()
                if not q["passed"] or q["coverage"] != a.n:
                    raise RuntimeError("full-coverage quality gate failed")
                for window, order in enumerate(cell["orders"]):
                    for system in order:
                        current_system, current_window = system, window
                        verify_fingerprint(source_hashes)
                        print(f"start cfg={cfg:g} B={a.current_batch} window={window} {system}", flush=True)
                        if system == "candidate":
                            candidate_activate(runtime, candidate)
                        try:
                            w = warmup(runtime, a, system, cfg, ids_batches, candidate, a.seed + 100000 + window * 1000)
                            if not w["stable"]:
                                record = {"status": "UNSTABLE_WARMUP", "system": system,
                                          "cfg": cfg, "batch_size": a.current_batch, "window": window}
                            elif a.mode == "e2e":
                                record = run_window(runtime, a, system, cfg, ids_batches, candidate, window)
                                expected = [row["reference_ids_sha256"] for row in q["batches"]]
                                observed = [row["ids_sha256"] for row in record["batches"]]
                                if observed != expected:
                                    record["status"] = "QUALITY_FAIL"
                                    record["reason"] = "measured final IDs differ from quality reference"
                            elif a.mode == "profile":
                                record = profile_window(runtime, a, system, cfg, ids_batches, candidate, window)
                            else:
                                record = serving_window(runtime, a, system, cfg, candidate, window,
                                                        cell["arrival_rate_per_second"])
                        finally:
                            if system == "candidate":
                                candidate_cleanup(runtime, candidate)
                        record["warmup"] = w
                        window_file.write(json.dumps(record, allow_nan=False) + "\n")
                        window_file.flush()
                        rate_label = f" rate={cell['arrival_rate_per_second']:g}" if a.mode == "serving" else ""
                        print(f"cfg={cfg:g} batch={a.current_batch}{rate_label} window={window} {system} {record['status']}", flush=True)
                        if record["status"] not in ("PASS", "UNSUPPORTED_PROFILE"):
                            failed = True
                        if record["status"] == "UNSUPPORTED_PROFILE":
                            unsupported = True
                verify_fingerprint(source_hashes)
            except Exception as exc:
                failed = True
                record = {"status": "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "ERROR",
                          "cfg": cfg, "batch_size": a.current_batch,
                          "system": current_system or runtime.last_system,
                          "window": current_window, "arrival_rate_per_second": cell["arrival_rate_per_second"],
                          "error": str(exc),
                          "traceback": traceback.format_exc()}
                window_file.write(json.dumps(record) + "\n")
                window_file.flush()
                print(f"cell failed: {record['status']} {exc}", flush=True)
                # OOM may leave allocator/model state uncertain. Preserve records and stop.
                break
    verify_fingerprint(source_hashes)
    assets_unchanged = fingerprint([Path(path) for path in m["asset_hashes"]]) == m["asset_hashes"]
    if not assets_unchanged:
        failed = True
        print("assets changed during run", file=sys.stderr)
    status = "FAILED" if failed else ("PARTIAL" if unsupported else "COMPLETE")
    write_json(output / "run.json", {"schema": SCHEMA, "status": status,
        "provenance_valid": assets_unchanged,
        "manifest_sha256": digest(output / "manifest.json"), "windows_sha256": digest(window_path),
        "quality_sha256": digest(quality_path)})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
