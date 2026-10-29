#!/usr/bin/env python3
"""Compare eager and legacy KV-cache generation on real WMT14 prompts.

The probe leaves the upstream implementation untouched.  It wraps
``_sample_level`` on one loaded model so that each run captures logits before
CFG combines the conditional and null branches.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer


DEFAULT_INDICES = [767, 1273, 1484, 1281, 866, 1933, 200, 1775, 393]
SEED = 20261027


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncm-root", type=Path, default=Path("upstream/ncm"))
    parser.add_argument("--config", type=Path, default=Path("artifacts/wmt14/config.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/wmt14"))
    parser.add_argument("--data", type=Path, default=Path("data/wmt14/sources.pickle"))
    parser.add_argument(
        "--hf-tokenizer", type=Path, default=Path("artifacts/qwen3-0.6b-tokenizer")
    )
    parser.add_argument(
        "--indices",
        default=",".join(map(str, DEFAULT_INDICES)),
        help="comma-separated WMT14 row indices",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_rev(path: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def branch_metrics(eager: torch.Tensor, cached: torch.Tensor) -> dict[str, Any]:
    eager64 = eager.double()
    delta64 = cached.double() - eager64
    abs_delta = delta64.abs()
    eager_norm_sq = eager64.square().sum().item()
    delta_norm_sq = delta64.square().sum().item()
    argmax_eager = eager.argmax(dim=-1)
    argmax_cached = cached.argmax(dim=-1)
    flips = argmax_eager.ne(argmax_cached)
    return {
        "elements": eager.numel(),
        "positions": flips.numel(),
        "different_elements": eager.ne(cached).sum().item(),
        "mean_abs": abs_delta.mean().item(),
        "rmse": math.sqrt(delta_norm_sq / eager.numel()),
        "max_abs": abs_delta.max().item(),
        "relative_l2": math.sqrt(delta_norm_sq / eager_norm_sq),
        "argmax_flips": flips.sum().item(),
        "argmax_flip_rate": flips.double().mean().item(),
        "_delta_norm_sq": delta_norm_sq,
        "_eager_norm_sq": eager_norm_sq,
        "_abs_sum": abs_delta.sum().item(),
    }


def public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    elements = sum(row["elements"] for row in metrics)
    positions = sum(row["positions"] for row in metrics)
    delta_norm_sq = sum(row["_delta_norm_sq"] for row in metrics)
    eager_norm_sq = sum(row["_eager_norm_sq"] for row in metrics)
    return {
        "elements": elements,
        "positions": positions,
        "different_elements": sum(row["different_elements"] for row in metrics),
        "mean_abs": sum(row["_abs_sum"] for row in metrics) / elements,
        "rmse": math.sqrt(delta_norm_sq / elements),
        "max_abs": max(row["max_abs"] for row in metrics),
        "relative_l2": math.sqrt(delta_norm_sq / eager_norm_sq),
        "argmax_flips": sum(row["argmax_flips"] for row in metrics),
        "argmax_flip_rate": sum(row["argmax_flips"] for row in metrics) / positions,
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    return float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            result[order[position]] = average_rank
        start = end
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def main() -> None:
    args = parse_args()
    args.ncm_root = args.ncm_root.resolve()
    args.config = args.config.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.data = args.data.resolve()
    args.hf_tokenizer = args.hf_tokenizer.resolve()
    args.output = args.output.resolve()
    indices = [int(value) for value in args.indices.split(",") if value]

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.ncm_root))
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg = OmegaConf.load(args.config)
    cfg.tokenizer.pre_tokenizer.model_id = str(args.hf_tokenizer)
    cfg.tokenizer.vqvae.checkpoint_path = str(args.checkpoint_dir / cfg.checkpoints.tokenizer)

    print("loading WMT14 tokenizer checkpoint", flush=True)
    tokenizer = VQVAETokenizer.from_config(cfg.tokenizer)
    tokenizer.vqvae = tokenizer.vqvae.cpu().eval()
    print("loading WMT14 generator checkpoint", flush=True)
    net = NextConceptTransformer(vqvae_local=tokenizer.vqvae, config=cfg.model).cpu()
    generator_path = args.checkpoint_dir / cfg.checkpoints.generator
    state = torch.load(generator_path, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"], strict=True)
    del state
    gc.collect()
    net.eval()

    with args.data.open("rb") as handle:
        sources = pickle.load(handle)
    source_tokenizer = AutoTokenizer.from_pretrained(
        str(args.hf_tokenizer), local_files_only=True
    )
    pad = tokenizer.pre_tokenizer.pad_token_id
    eos = tokenizer.pre_tokenizer.eos_token_id
    prefix_length = int(cfg.model.prefix_len)

    prompt_rows = []
    prefix_rows = []
    for index in indices:
        text = sources[index]
        ids = source_tokenizer(text)["input_ids"]
        effective_ids = ids[:prefix_length]
        padded = (effective_ids + [pad] * prefix_length)[:prefix_length]
        prompt_rows.append(
            {
                "index": index,
                "text": text,
                "token_length": len(ids),
                "effective_token_length": len(effective_ids),
                "padding_tokens": prefix_length - len(effective_ids),
            }
        )
        prefix_rows.append(padded)
    prefix_tensor = torch.tensor(prefix_rows, dtype=torch.long)
    batch_size = len(indices)

    def run(use_kv_cache: bool) -> tuple[torch.Tensor, list[torch.Tensor], float]:
        captures: list[torch.Tensor] = []
        original_sample_level = net._sample_level

        def capture(logits: torch.Tensor, temperature: float, top_k: int, top_p: float):
            captures.append(logits.detach().float().cpu().clone())
            return original_sample_level(logits, temperature, top_k, top_p)

        net._sample_level = capture
        torch.manual_seed(SEED)
        started = time.perf_counter()
        try:
            output = net.generate(
                None,
                prefix_tensor,
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                use_kv_cache=use_kv_cache,
                cfg_scale=1.0,
                cfg_null_eos=eos,
            ).cpu()
        finally:
            net._sample_level = original_sample_level
        elapsed = time.perf_counter() - started
        if len(captures) != len(net.levels):
            raise RuntimeError(f"captured {len(captures)} levels, expected {len(net.levels)}")
        return output, captures, elapsed

    print(f"running eager CFG=1 batch={batch_size}", flush=True)
    eager_codes, eager_logits, eager_seconds = run(False)
    print(f"running KV-cache CFG=1 batch={batch_size}", flush=True)
    cached_codes, cached_logits, cached_seconds = run(True)

    levels = list(map(int, net.levels))
    per_level = []
    per_prompt_metrics: list[dict[str, Any]] = [
        {"cond": [], "null": [], "guided": []} for _ in indices
    ]
    for level_index, level_size in enumerate(levels):
        eager = eager_logits[level_index]
        cached = cached_logits[level_index]
        if eager.shape[0] != 2 * batch_size:
            raise RuntimeError(f"unexpected CFG batch at level {level_index}: {eager.shape}")
        cond = branch_metrics(eager[:batch_size], cached[:batch_size])
        null = branch_metrics(eager[batch_size:], cached[batch_size:])
        guidance_weight = level_index / max(len(levels) - 1, 1)
        eager_guided = (1.0 + guidance_weight) * eager[:batch_size] - guidance_weight * eager[batch_size:]
        cached_guided = (1.0 + guidance_weight) * cached[:batch_size] - guidance_weight * cached[batch_size:]
        guided = branch_metrics(eager_guided, cached_guided)
        per_level.append(
            {
                "level_index": level_index,
                "level_size": level_size,
                "guidance_weight": guidance_weight,
                "cond": public_metrics(cond),
                "null": public_metrics(null),
                "guided": public_metrics(guided),
            }
        )
        for prompt_index in range(batch_size):
            for name, left, right in (
                ("cond", eager[prompt_index], cached[prompt_index]),
                ("null", eager[batch_size + prompt_index], cached[batch_size + prompt_index]),
                ("guided", eager_guided[prompt_index], cached_guided[prompt_index]),
            ):
                per_prompt_metrics[prompt_index][name].append(branch_metrics(left, right))

    offsets = [0]
    for level_size in levels:
        offsets.append(offsets[-1] + level_size)
    prompt_results = []
    for prompt_index, row in enumerate(prompt_rows):
        code_levels = []
        for level_index in range(len(levels)):
            left = eager_codes[prompt_index, offsets[level_index] : offsets[level_index + 1]]
            right = cached_codes[prompt_index, offsets[level_index] : offsets[level_index + 1]]
            mismatch = left.ne(right)
            code_levels.append(
                {
                    "level_index": level_index,
                    "level_size": levels[level_index],
                    "mismatched_codes": mismatch.sum().item(),
                    "mismatch_rate": mismatch.double().mean().item(),
                }
            )
        first_code_mismatch = next(
            (item["level_index"] for item in code_levels if item["mismatched_codes"]), None
        )
        branch_summary = {
            name: aggregate(per_prompt_metrics[prompt_index][name])
            for name in ("cond", "null", "guided")
        }
        first_exact_difference = {
            name: next(
                (
                    level_index
                    for level_index, item in enumerate(per_prompt_metrics[prompt_index][name])
                    if item["different_elements"]
                ),
                None,
            )
            for name in ("cond", "null", "guided")
        }
        first_argmax_flip = {
            name: next(
                (
                    level_index
                    for level_index, item in enumerate(per_prompt_metrics[prompt_index][name])
                    if item["argmax_flips"]
                ),
                None,
            )
            for name in ("cond", "null", "guided")
        }
        prompt_results.append(
            {
                **row,
                "first_exact_logit_difference": first_exact_difference,
                "first_argmax_flip": first_argmax_flip,
                "first_code_mismatch": first_code_mismatch,
                "code_levels": code_levels,
                "aggregate": branch_summary,
                "level1": {
                    name: public_metrics(per_prompt_metrics[prompt_index][name][1])
                    for name in ("cond", "null", "guided")
                },
            }
        )

    global_summary = {
        name: aggregate(
            [metric for prompt in per_prompt_metrics for metric in prompt[name]]
        )
        for name in ("cond", "null", "guided")
    }
    lengths = [float(row["effective_token_length"]) for row in prompt_results]
    length_relation = {}
    for scope, getter in (
        ("level1", lambda row, name: row["level1"][name]),
        ("all_levels", lambda row, name: row["aggregate"][name]),
    ):
        length_relation[scope] = {}
        for branch in ("cond", "null", "guided"):
            length_relation[scope][branch] = {}
            for metric_name in ("rmse", "relative_l2", "argmax_flip_rate"):
                values = [float(getter(row, branch)[metric_name]) for row in prompt_results]
                length_relation[scope][branch][metric_name] = {
                    "pearson_r": pearson(lengths, values),
                    "spearman_rho": spearman(lengths, values),
                }
    code_mismatch_counts = [
        float(sum(level["mismatched_codes"] for level in row["code_levels"]))
        for row in prompt_results
    ]
    code_mismatch_rates = [value / offsets[-1] for value in code_mismatch_counts]
    length_relation["codes"] = {
        "mismatch_count": {
            "pearson_r": pearson(lengths, code_mismatch_counts),
            "spearman_rho": spearman(lengths, code_mismatch_counts),
        },
        "mismatch_rate": {
            "pearson_r": pearson(lengths, code_mismatch_rates),
            "spearman_rho": spearman(lengths, code_mismatch_rates),
        },
    }

    result = {
        "schema_version": 1,
        "experiment": "WMT14 net.generate eager vs legacy KV cache, CFG=1",
        "seed": SEED,
        "settings": {
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.0,
            "cfg_scale": 1.0,
            "batch_size": batch_size,
            "prefix_length": prefix_length,
            "levels": levels,
            "threads": args.threads,
            "device": "cpu",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "ncm_commit": git_rev(args.ncm_root),
            "files": {
                "generator": {"path": str(generator_path), "sha256": sha256(generator_path)},
                "tokenizer": {
                    "path": str(args.checkpoint_dir / cfg.checkpoints.tokenizer),
                    "sha256": sha256(args.checkpoint_dir / cfg.checkpoints.tokenizer),
                },
                "config": {"path": str(args.config), "sha256": sha256(args.config)},
                "sources": {"path": str(args.data), "sha256": sha256(args.data)},
            },
        },
        "runtime_seconds": {"eager": eager_seconds, "kv_cache": cached_seconds},
        "codes_all_equal": torch.equal(eager_codes, cached_codes),
        "total_mismatched_codes": eager_codes.ne(cached_codes).sum().item(),
        "global": global_summary,
        "per_level": per_level,
        "prompts": prompt_results,
        "length_relation": length_relation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
