# STAGE 3 — GraSTI / ToyNet / VIB AUDIT

Date: 2026-08-17 · Baseline: Stage 1+2 frozen · Seed 42 · Read-only (zero repo code edits — all probing test-local) · No epochs, no accuracy.

## 1. Paper vs authors' code vs our code

| quantity | paper | authors' released code | our code | verdict |
|---|---|---|---|---|
| ToyNet input per ROI | Eq. 13 rewrites W as **dynamic** W ∈ R^{T×N×N}; VIB conditions on it | **static PCC row only** [90]; `get_mu_std_logits(edge_weight)` — `dyn_weight` is only a None/not-None switch, values never enter | static row [90] + dynamic rows [T·90] concatenated → **[360]** | authors MISMATCH paper (no dynamic); ours uses dynamic (Eq.13-aligned) + static (superset) — AMBIGUOUS whether static belongs, documented |
| FC_mean / FC_var | not specified at this granularity | `Linear(hidden→1)` then `view(1,-1)` then `FC_*_trans: Linear(90→800)` | `Linear(hidden→800)` directly | see §4: authors' path BROKEN in execution |
| latent_dim | not specified | 800 (via `*_trans`) | 800 | MATCH (intent) |
| decoder | maps latent → edge decisions | `Linear(800→90)`, no activation | identical | MATCH |
| mu/std aggregation | not specified | stack 90 ROI latents, mean per subject | identical | MATCH |
| stochastic z | reparameterization (Eq. 11 trick) | `z = mu + eps·std` | identical | MATCH |

## 2. Input trace (two real subjects)

static [2,90,90] ✓ · dynamic [2,3,90,90] ✓ · `row = concat(static_row[90], dyn[T·90])` → **[360] = 90·(1+T)** ✓ (both subjects).

## 4. Authors' dimension path — EXECUTED

Authors' ToyNet instantiated verbatim (`input_dim=90`, matching their own `GraSTIACL.py`), fed exactly what their `get_mu_std_logits` passes (`edge_weight_reshaped[i,j,:]` → shape [90]):

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1 and 90x800)
```

Mechanism: on a 1-D [90] input, `encode` → [hidden], `FC_mean(hidden→1)` → [1], `view(1,-1)` → [1,1], then `FC_mean_trans` expects last dim 90. **The released path cannot execute a single ROI forward** — meaning the released code, as-is, cannot be the code that produced the paper's results. Our current `FC_mean/FC_var: Linear(hidden→800)` executes cleanly on the same logical input (mu [800], std [800], logits [90]).

## 5. Shape contract (B=2, T=3)

per-ROI input [360] · mu_row/std_row **[800]** · edge_logits_row **[90]** · per subject **[8100]** · two subjects **[16200]** · returned mu/std **[1600] = B·800** · training reshape → KLD input **[2,800]** — no batch/ROI collapse. All asserted ✓.

## 6. Edge-order alignment — ALL positions

Test-local probe (forward replaced by identity-on-static-half): resulting `edge_logits` equals `W_static.flatten()` row-major **bitwise for all 16,200 positions**, and `edge_index[:, j·90+k] == [j,k]` re-verified for all 8,100. Logit ordering == edge ordering. ✓

## 7-8. Information tests (fixed params + seed)

Perturb static only: Δlogits 5.78e-02, Δmu 1.41e-03, Δstd 1.01e-05 → **static used**. ✓
Perturb dynamic only: Δlogits 5.90e-02, Δmu 8.58e-03, Δstd 5.81e-05 → **dynamic used — the temporal branch is numerically real**. ✓

## 9. Subject isolation

Perturbing subject 1's static+dynamic leaves subject 0's mu/std/edge_logits allclose-unchanged. ✓

## 10-11. VIB + decoder

mu finite, std finite and strictly positive (softplus(·−5)); fixed seed → bit-reproducible; different seed → differs (stochastic z retained — matches both paper's reparameterization and both codebases). Decoder = bare `Linear(800→90)`, no sigmoid/softmax — outputs genuine logits (observed range escapes [0,1]); Stage-2's Eq. 12 owns the squashing. ✓

## 12. Aggregation + KLD

ROI latent [800] × 90 → mean per subject → [800]; concat → [B·800]; training reshape [B,800]; KLD finite scalar. Aggregation rule identical to authors'; paper silent; unchanged. ✓

## 13-14. Gradients + numerical safety

All five layer groups receive finite nonzero gradients; the dynamic-input columns (90:360) of the first encode layer receive nonzero gradient — the temporal route trains. Two real subjects: everything finite, std>0, no sanitization used in-test (fail-loud). ✓

## 15. M_ij

Untouched. ViewLearner confirmed returning `edge_logits, mu, std, edge_prod` together. Stage 4 material.

## Final summary

```
STATIC_PCC_USED_NUMERICALLY        = YES
DYNAMIC_PCC_USED_NUMERICALLY       = YES
PER_ROI_INPUT_DIM                  = 360  (= 90*(1+T), T=3)
LATENT_DIM                         = 800
EDGE_LOGITS_PER_ROI                = 90
EDGE_LOGITS_PER_SUBJECT            = 8100
EDGE_LOGIT_ORDER_MATCHES_EDGE_INDEX = YES (bitwise, all positions)
MU_STD_SHAPES_CORRECT              = YES ([B*800] -> KLD [B,800])
STD_STRICTLY_POSITIVE              = YES
SUBJECT_ISOLATION                  = PASS
AUTHORS_ORIGINAL_DIMENSION_PATH    = BROKEN (RuntimeError executed, 1x1 vs 90x800)
OUR_CURRENT_DIMENSION_REPAIR       = CODE_ONLY_CORRECTION
                                     (dimension fix unavoidable; dynamic-input
                                      inclusion is Eq.13-aligned; static
                                      inclusion is a documented superset choice
                                      the paper neither requires nor forbids)
KLD_SHAPE                          = [B, 800]
ALL_STAGE3_TESTS                   = PASS (30/30)
```

No confirmed error in our code → **no edits made**. Proposed minimal fix: none required.
