# WMT14 legacy KV-cache divergence probe

Date: 2026-10-27  
Status: reproduced

## Result

With CFG=1, every right-padded prompt diverged between
`net.generate(use_kv_cache=False)` and `net.generate(use_kv_cache=True)` at
**level index 1**, whose target length is 4. Level index 0 (target length 1)
was identical. The first raw-logit difference, first top-1 difference, and
first returned-code difference were all at level 1 for all eight padded
prompts.

The no-padding control behaved differently: its conditional branch was
bitwise identical through level 3. The null branch still diverged at level 1,
the guided top-1 first changed at level 3, and the returned code first changed
at level 3. Its conditional branch only began to differ at level 4, after the
earlier guided-code divergence had changed the next-level input.

Across all nine prompts and all 11 levels, the null branch was affected more
than the conditional branch:

| branch | RMSE | relative L2 | raw top-1 flip rate |
|---|---:|---:|---:|
| conditional | 5.2380 | 39.74% | 73.52% |
| null | 6.1857 | 46.70% | 77.46% |
| CFG-combined | 7.6331 | 57.29% | 77.07% |

The null branch therefore had 18.1% higher RMSE, 6.96 percentage points
higher relative-L2 error, and 3.94 percentage points more raw top-1 flips than
the conditional branch over the complete generation trajectory.

## Per-prompt observations

`tokens` is the effective prompt length after truncation to 96; `pad` is the
number of right-padding tokens. Level numbers are zero-based.

| WMT14 row | tokens | pad | first cond logit diff | first null logit diff | first guided top-1 diff | first code diff | changed codes / 313 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 767 | 3 | 93 | 1 | 1 | 1 | 1 | 228 |
| 1273 | 12 | 84 | 1 | 1 | 1 | 1 | 202 |
| 1484 | 24 | 72 | 1 | 1 | 1 | 1 | 225 |
| 1281 | 33 | 63 | 1 | 1 | 1 | 1 | 241 |
| 866 | 44 | 52 | 1 | 1 | 1 | 1 | 275 |
| 1933 | 56 | 40 | 1 | 1 | 1 | 1 | 248 |
| 200 | 73 | 23 | 1 | 1 | 1 | 1 | 265 |
| 1775 | 92 | 4 | 1 | 1 | 1 | 1 | 300 |
| 393 | 96 | 0 | 4 | 1 | 3 | 3 | 187 |

Prompt length has two different answers depending on where the difference is
measured:

- At the first affected level, shorter prompts produced a clearly larger
  direct conditional-logit error. Conditional level-1 RMSE versus effective
  length had Pearson `r=-0.888` and Spearman `rho=-0.883`; relative L2 had
  `r=-0.880` and `rho=-0.950`. The 96-token/no-pad control had exactly zero
  conditional difference at level 1.
- After all 11 levels, discrete code differences were not monotonic with
  length. Changed-code count versus length had only weak positive correlation
  (`r=0.302`, `rho=0.300`), because early top-1 changes cascade through later
  levels. Thus the supported conclusion is **shorter means larger direct
  padding-induced logit error**, not “shorter always means more final code
  changes.”

## Cause in the inspected implementation

The eager path adds `_pbias` to its prompt and per-level attention masks. The
legacy KV path re-encodes the prompt with an all-zero `pfx_mask`, then uses an
all-zero `lvl_mask`; neither includes the padding bias. Consequently, padded
prompt keys are visible only on the cached path. Under CFG, the null prefix is
one EOS token plus 95 PAD tokens, so it is structurally the most exposed
branch.

This also explains why level 0 is identical: the shared level-0 computation
uses the correctly padded mask before the implementation enters its KV-cache
prefill path.

## Method and reproducibility

- NCM source commit: `c70274273f489abc73832b2b98584a8483b6b203`
- WMT14 levels: `[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]`
- CFG=1, temperature=1, top-k=1, top-p=0, seed=20261027
- Nine real WMT14 prompts in one batch; prefix length 96
- PyTorch 2.6.0, FP32, CPU on Apple M2; this is a correctness probe, not a
  CUDA performance measurement
- Both runs used the same loaded model instance and reset the RNG to the same
  seed. The probe captured raw conditional/null logits immediately before CFG
  combination without changing model code.
- A second identical run produced the same result payload (excluding timing),
  SHA-256 `106240b609c6c959bd2d1cd2c6d992da786ccc8561f89f9bbd19eaf988809135`.

Raw measurements are in
`reports/raw/2026-10-27-wmt14-kv-cache-probe.json`; the reproducible probe is
`experiments/wmt14_kv_cache_probe.py`.
