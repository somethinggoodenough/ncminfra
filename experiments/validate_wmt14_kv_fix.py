"""Paired native generate validation, with isolated in-memory mask ablations.

Run small before full. No forcing of concept histories or sampling shortcuts.
Bitwise means byte equality (including signed zero), not torch.allclose.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import pickle
import platform
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/ncm"))
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

SAMPLE = [767, 1273, 1484, 1281, 866, 1933, 200, 1775]
ATOL, RTOL, SEED = 1e-4, 1e-3, 20261027
PREFILL_FIX = "            if _pbias is not None:\n                pfx_mask = pfx_mask + _pbias\n"
LEVEL_FIX = "                if _pbias is not None:\n                    lvl_mask = lvl_mask + F.pad(_pbias, (0, L_cur))\n"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()


def stats(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    # Integer views distinguish signed zero as well as different float values.
    bits = a.contiguous().view(torch.int32).ne(b.contiguous().view(torch.int32))
    diff = (a - b).abs()
    outside = diff > (ATOL + RTOL * a.abs())
    flips = a.argmax(-1).ne(b.argmax(-1))
    return {
        "elements": a.numel(), "bitwise_different_elements": bits.sum().item(),
        "numerically_different_elements": a.ne(b).sum().item(),
        "outside_tolerance": outside.sum().item(), "finite": finite,
        "max_abs": diff.max().item(),
        "squared_error": diff.double().square().sum().item(),
        "reference_squared_norm": a.double().square().sum().item(),
        "positions": flips.numel(), "argmax_flips": flips.sum().item(),
    }


def merge_stats(rows):
    if not rows:
        return {}
    sums = ("elements", "bitwise_different_elements", "numerically_different_elements",
            "outside_tolerance", "squared_error", "reference_squared_norm",
            "positions", "argmax_flips")
    out = {k: sum(r[k] for r in rows) for k in sums}
    out.update(finite=all(r["finite"] for r in rows),
               max_abs=max(r["max_abs"] for r in rows))
    return out


def classify(logits, latent, decoder, codes_equal, ids_equal, text_equal):
    all_stats = [*logits, latent, decoder]
    finite = all(r["finite"] for r in all_stats)
    exact = finite and all(r["bitwise_different_elements"] == 0 for r in all_stats)
    close = finite and all(r["outside_tolerance"] == 0 for r in all_stats)
    discrete = codes_equal and ids_equal and text_equal
    if exact and discrete:
        return "BITWISE"
    if close and discrete:
        return "TIER_1"
    return "FAIL"


def make_variants(net):
    original = net.generate
    source = textwrap.dedent(inspect.getsource(NextConceptTransformer.generate))
    # Require exactly the two intended patch sites, protecting the shared L0 mask.
    assert source.count(PREFILL_FIX) == source.count(LEVEL_FIX) == 1
    variants = {"both": original}
    for name, remove in (("legacy", [PREFILL_FIX, LEVEL_FIX]),
                         ("prefill_only", [LEVEL_FIX]),
                         ("level_only", [PREFILL_FIX])):
        body = source
        for block in remove:
            body = body.replace(block, "", 1)
        namespace = {}
        exec(compile(body, f"<ablation:{name}>", "exec"),
             NextConceptTransformer.generate.__wrapped__.__globals__, namespace)
        variants[name] = types.MethodType(namespace["generate"], net)
    return variants, source


def generate(net, method, prefix, cfg_scale, cache, eos, capture=True):
    logits = []
    sample = net._sample_level

    def tap(value, temperature, top_k, top_p):
        logits.append(value.detach().clone())
        return sample(value, temperature, top_k, top_p)

    torch.manual_seed(SEED)
    if capture:
        net._sample_level = tap
    start = time.perf_counter()
    try:
        codes = method(None, prefix, temperature=1.0, top_k=1, top_p=0.0,
                       use_kv_cache=cache, cfg_scale=cfg_scale, cfg_null_eos=eos)
    finally:
        if capture:
            del net._sample_level
    assert all(not b.attn.caching and b.attn.cached_k is None
               and b.attn.cached_v is None for b in net.transformer.blocks)
    assert net._cfg is None
    if capture:
        assert len(logits) == len(net.levels)
    return codes, logits, torch.get_rng_state().clone(), time.perf_counter() - start


@torch.no_grad()
def compare(tk, net, eager, cached, indices, cfg_scale, lengths, decode_bs=2):
    ec, el, erng, et = eager
    cc, cl, crng, ct = cached
    b = len(indices)
    levels, samples = [], []
    branches = ["cond", "null", "guided"] if cfg_scale else ["cond"]
    first_bits = [{name: None for name in branches} for _ in indices]
    first_flips = [{name: None for name in branches} for _ in indices]
    for level, (a, z) in enumerate(zip(el, cl)):
        assert a.dtype == z.dtype == torch.float32
        pairs = {"cond": (a[:b], z[:b])}
        if cfg_scale:
            t = cfg_scale * level / (len(net.levels) - 1)
            pairs.update(null=(a[b:], z[b:]),
                         guided=((1+t)*a[:b]-t*a[b:], (1+t)*z[:b]-t*z[b:]))
        row = {"level": level, "length": int(net.levels[level]), "branches": {}}
        for name, (left, right) in pairs.items():
            row["branches"][name] = stats(left, right)
            for i in range(b):
                if first_bits[i][name] is None and not torch.equal(
                    left[i].contiguous().view(torch.int32),
                    right[i].contiguous().view(torch.int32)
                ):
                    first_bits[i][name] = level
                if first_flips[i][name] is None and not torch.equal(
                    left[i].argmax(-1), right[i].argmax(-1)
                ):
                    first_flips[i][name] = level
        row["eager_sha256"] = tensor_hash(a)
        row["cache_sha256"] = tensor_hash(z)
        levels.append(row)

    # Decode both trajectories independently, even when the concept IDs match.
    _, ez = tk.vqvae.vq.idx_to_zhats(ec, return_zlevels=True)
    _, cz = tk.vqvae.vq.idx_to_zhats(cc, return_zlevels=True)
    latent = stats(ez[-1], cz[-1])
    token_rows, decoder_stats, decoder_hashes = [], [], []
    for start in range(0, b, decode_bs):
        a = tk.vqvae.decoder(ez[-1][start:start+decode_bs])
        z = tk.vqvae.decoder(cz[-1][start:start+decode_bs])
        decoder_stats.append(stats(a, z))
        decoder_hashes.append({"eager": tensor_hash(a), "cache": tensor_hash(z)})
        token_rows.extend(zip(a.argmax(-1).tolist(), z.argmax(-1).tolist()))
        del a, z
    decoder = merge_stats(decoder_stats)
    eos, pad = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id

    def text(ids):
        ids = ids[:ids.index(eos)] if eos in ids else ids
        return tk.pre_tokenizer.decode([v for v in ids if v not in (eos, pad)],
                                       skip_special_tokens=True)

    offset = [0]
    for length in net.levels:
        offset.append(offset[-1] + length)
    for i, index in enumerate(indices):
        counts = [int(ec[i, offset[l]:offset[l+1]].ne(cc[i, offset[l]:offset[l+1]]).sum())
                  for l in range(len(net.levels))]
        ei, ci = token_rows[i]
        etext, ctext = text(ei), text(ci)
        samples.append({
            "index": index, "token_length": lengths[index],
            "effective_length": min(lengths[index], net.prefix_len),
            "first_logit_bit_difference": first_bits[i],
            "first_branch_argmax_difference": first_flips[i],
            "first_code_difference": next((l for l, n in enumerate(counts) if n), None),
            "code_mismatches_by_level": counts,
            "eager_codes": ec[i].tolist(), "cache_codes": cc[i].tolist(),
            "eager_ids": ei, "cache_ids": ci,
            "id_mismatches": sum(x != y for x, y in zip(ei, ci)),
            "eager_text": etext, "cache_text": ctext,
            "text_equal": etext == ctext,
        })
    codes_equal = torch.equal(ec, cc)
    ids_equal = all(row["id_mismatches"] == 0 for row in samples)
    texts_equal = all(row["text_equal"] for row in samples)
    all_logits = [value for row in levels for value in row["branches"].values()]
    result = {
        "classification": classify(all_logits, latent, decoder, codes_equal, ids_equal, texts_equal),
        "rng_equal": torch.equal(erng, crng),
        "runtime_seconds": {"eager": et, "cache": ct},
        "levels": levels, "latent": latent, "decoder": decoder,
        "decoder_hashes": decoder_hashes, "samples": samples,
    }
    assert result["rng_equal"], "paired runs consumed different RNG streams"
    return result


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["small", "full"], required=True)
    ap.add_argument("--cfg", default="1")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # New directory is mandatory: never mix partial or previous measurements.
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cfg = OmegaConf.load(ROOT / "artifacts/wmt14/config.yaml")
    cfg.tokenizer.pre_tokenizer.model_id = str(ROOT / "artifacts/qwen3-0.6b-tokenizer")
    cfg.tokenizer.vqvae.checkpoint_path = str(ROOT / "artifacts/wmt14/tokenizer.pt")
    print("loading checkpoints", flush=True)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae.eval()
    net = NextConceptTransformer(tk.vqvae, cfg.model).eval()
    sd = torch.load(ROOT / "artifacts/wmt14/generator.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    del sd
    gc.collect()
    variants, source = make_variants(net)
    with (ROOT / "data/wmt14/sources.pickle").open("rb") as f:
        sources = pickle.load(f)
    assert len(sources) == 3003
    tokens = [tk.pre_tokenizer.encode(t) for t in sources]
    lengths = [len(t) for t in tokens]
    pad, eos, p = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id, net.prefix_len
    prefixes = torch.tensor([(t[:p] + [pad]*p)[:p] for t in tokens], dtype=torch.long)
    indices = SAMPLE if args.mode == "small" else list(range(len(sources)))
    cfgs = [float(v) for v in args.cfg.split(",")]
    variants_to_test = ["legacy", "prefill_only", "level_only", "both"] if args.mode == "small" else ["both"]
    manifest = {
        "command": sys.argv, "mode": args.mode, "indices": indices,
        "cfgs": cfgs, "variants": variants_to_test, "seed": SEED,
        "batch_size": args.batch_size, "decoder_microbatch_size": 2,
        "dtype": "float32", "device": "cpu", "threads": args.threads,
        "prefix_length": p, "eos": eos, "pad": pad, "levels": list(net.levels),
        "temperature": 1., "top_k": 1, "top_p": 0.,
        "comparison": {"atol": ATOL, "rtol": RTOL, "reference": "eager",
                       "bitwise": "float32 byte equality plus all discrete outputs exact",
                       "tier_1": "finite logits/latent/decoder allclose; codes/IDs/text exact"},
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
        "upstream_sha": subprocess.check_output(["git", "-C", str(ROOT/"upstream/ncm"), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_diff": subprocess.check_output(["git", "-C", str(ROOT/"upstream/ncm"), "diff"], text=True),
        "generate_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "files": {str(path.relative_to(ROOT)): file_hash(path) for path in [
            Path(__file__).resolve(), ROOT/"upstream/ncm/models/next_concept.py",
            ROOT/"upstream/ncm/models/common.py", ROOT/"artifacts/wmt14/generator.pt",
            ROOT/"artifacts/wmt14/tokenizer.pt", ROOT/"artifacts/wmt14/config.yaml",
            ROOT/"artifacts/qwen3-0.6b-tokenizer/tokenizer.json", ROOT/"data/wmt14/sources.pickle"]},
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
    }
    save_json(manifest_path, manifest)
    # Validate hook transparency on the original eight prompts, for each requested CFG.
    summaries = []
    for scale in cfgs:
        records_path = output / f"cfg{scale:g}-batches.jsonl"
        summary = {"cfg": scale, "expected_samples": len(indices), "samples": 0,
                   "batches": 0, "variants": {name: {"BITWISE": 0, "TIER_1": 0, "FAIL": 0,
                       "code_mismatches": 0, "id_mismatches": 0, "text_mismatches": 0,
                       "first_failed_indices": [], "logits": {}, "latent": {}, "decoder": {}}
                       for name in variants_to_test}}
        start = time.perf_counter()
        with records_path.open("x") as records:
            for offset in range(0, len(indices), args.batch_size):
                batch = indices[offset:offset+args.batch_size]
                x = prefixes[batch]
                eager = generate(net, variants["both"], x, scale, False, eos)
                if args.mode == "small" and offset == 0:
                    bare = generate(net, variants["both"], x, scale, False, eos, capture=False)
                    assert torch.equal(eager[0], bare[0]) and torch.equal(eager[2], bare[2])
                    summary["eager_probe_transparent"] = True
                for name in variants_to_test:
                    cached = generate(net, variants[name], x, scale, True, eos)
                    if args.mode == "small" and offset == 0:
                        bare = generate(net, variants[name], x, scale, True, eos, capture=False)
                        assert torch.equal(cached[0], bare[0]) and torch.equal(cached[2], bare[2])
                        summary["variants"][name]["probe_transparent"] = True
                    row = compare(tk, net, eager, cached, batch, scale, lengths)
                    row.update(variant=name, cfg=scale, batch=summary["batches"], indices=batch)
                    records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    records.flush()
                    target = summary["variants"][name]
                    target[row["classification"]] += 1
                    for sample in row["samples"]:
                        target["code_mismatches"] += sum(sample["code_mismatches_by_level"])
                        target["id_mismatches"] += sample["id_mismatches"]
                        target["text_mismatches"] += not sample["text_equal"]
                        if sample["first_code_difference"] is not None and len(target["first_failed_indices"]) < 20:
                            target["first_failed_indices"].append(sample["index"])
                    for layer in row["levels"]:
                        for branch, value in layer["branches"].items():
                            key = f"level{layer['level']}/{branch}"
                            old = target["logits"].get(key)
                            target["logits"][key] = merge_stats([old, value] if old else [value])
                    for key in ("latent", "decoder"):
                        old = target[key]
                        target[key] = merge_stats([old, row[key]] if old else [row[key]])
                    print(f"cfg={scale:g} {offset+len(batch)}/{len(indices)} {name}: {row['classification']} "
                          f"codes={target['code_mismatches']} ids={target['id_mismatches']} "
                          f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
                    del cached, row
                del eager
                summary["batches"] += 1
                summary["samples"] += len(batch)
                summary["elapsed_seconds"] = time.perf_counter() - start
                save_json(output / f"cfg{scale:g}-progress.json", summary)
        # Independently verify exact coverage from the durable batch records.
        seen = {name: [] for name in variants_to_test}
        with records_path.open() as records:
            for line in records:
                row = json.loads(line)
                seen[row["variant"]].extend(row["indices"])
        assert all(values == indices for values in seen.values())
        summary["coverage_verified"] = True
        summary["records_sha256"] = file_hash(records_path)
        save_json(output / f"cfg{scale:g}-summary.json", summary)
        summaries.append(summary)
    save_json(output / "summary.json", summaries)


if __name__ == "__main__":
    main()
