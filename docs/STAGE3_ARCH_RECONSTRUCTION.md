# STAGE 3 (REOPENED) — PAPER-INTENT RECONSTRUCTION OF ToyNet

Date: 2026-08-17 · Read-only (zero production edits) · Seed 42 · No training, no accuracy.
Supersedes the fidelity claim implied by the earlier 30/30 (which certified INTERNAL consistency only).

## 1. Parameter counts — the decisive evidence

| Model | Exact count | Paper states |
|---|---|---|
| Authors' ToyNet, two-stage (90→400→400; 400→1, 90→800 ×2; 800→90) | **415,292** | "~415k shared" |
| Main model (TAEncoder emb32 + authors' ToyNet + proj head) | **422,076** | "~422k main" |
| View module (TAEncoder emb32 + authors' ToyNet + edge MLP) | **424,189** | "~420k auxiliary" |
| **Our current ToyNet** (360→400→400; 400→800 ×2; 800→90) | **1,018,490** | — |

The authors' layer definitions reproduce the paper's three published parameter
counts to within 0.02-1%. This is quantitative proof that **the paper's own
parameter accounting was computed from the two-stage 90→800 architecture** —
the released layer shapes ARE the paper's architecture; only the released
`forward()` wiring is broken. Our current ToyNet is **2.45×** the paper's
size. Not "parameter-count proof" of every detail — but the two-stage
architecture is the only one quantitatively consistent with the paper.

## 2. Reconstructed intended tensor flow — executed

Batched two-stage flow (one subject): `encode([90,90]) → [90,400]` →
`FC_mean → [90,1] → squeeze [90]` → `FC_mean_trans → mu [800]` (likewise
std via `softplus(·−5)`) → `z [800]` → `decode(z) → [90]`. **EXECUTES.**

Unresolved remainder: `decode(z)` yields **90 outputs per subject**, but the
edge contract needs **8100** (their own released loop concatenates 90 logits
per ROI × 90 ROIs). A subject-level z cannot produce per-ROI logits with this
decoder. **Edge-logit production under the reconstruction = AMBIGUOUS** —
unresolvable from layer shapes alone.

## 3. Dynamic input — paper-intent evidence (Eq. 13)

| Implementation | Evidence | Status |
|---|---|---|
| A. dynamic windows only | Eq. 13 literally rewrites W as dynamic 𝒲 ∈ R^{T×N×N} and conditions p(A\|𝒲); Eq. 4's prose calls W "the edge weight matrix representing temporal information" | **PAPER_SUPPORTED** |
| B. static + dynamic concatenation (current, [360]) | dynamic part supported as above; a static-PCC VIB input is nowhere mentioned in Eq. 13's chain | **PAPER_AMBIGUOUS** (superset; static half unsupported-but-unforbidden) |
| C. shared per-window processing ([90] per window) + aggregation | not specified — but uniquely compatible with BOTH Eq. 13 (dynamic) AND `input_dim=90` (the parameter-count-matching architecture) | **PAPER_AMBIGUOUS** (the only variant satisfying both constraints; noted) |

## 4. 40-TR window fidelity

Current `dyn_weight`: Notebook 2 — `N_WINDOWS=3`, `WINDOW_SECONDS=96`,
`window_volumes = round(96/TR)`, non-overlapping from index 0, remainder
discarded, fixed T=3 (longer scans' extra full segments unused).
TR=1.5→64 vol, TR=2.0→48, **TR=2.4→40 (paper-exact)**, TR=3.0→32.
Per-subject TR table is external (subject_tr.csv absent locally); notebook
records 0/956 subjects short, min spare 2 volumes.
**Classification: ABIDE_ADAPTATION** (user-decided, documented) — only
TR=2.4s sites match the paper's literal 40 TR.

## 5. KL aggregation diagnostic (two real subjects, ROI-level retained)

| Subject | KL(mean_ROI(mu,std)) — current | mean_ROI(KL) — alternative | ratio |
|---|---|---|---|
| 0 | 3604.856 | 3606.251 | 1.000 |
| 1 | 3601.004 | 3603.527 | 1.001 |

Numerically near-identical here. Current rule == authors' released rule
(average ROI mu/std before returning). Paper's Eq. 14 double-sums over i,j
but never specifies the neural latent parameterization.
**MATCH AUTHORS · PAPER AMBIGUOUS · no change made or warranted by this measurement.**

## 6. Dead/landmine paths

- `weight_init()`: 0 call sites (dead); if ever invoked it RAISES
  `TypeError: 'Linear' object is not iterable` — landmine, but dead.
- `num_sample`: no production call site passes it (default 1 everywhere);
  the `num_sample>1` branch would softmax the logits — landmine, dead.
- bare `edge_logits.squeeze()` statement: confirmed no-op.
- `softplus(var−5)`: std ≈ 0.00672 at init — small-init design, documented.

## 7. Decision table

| quantity | paper | authors' layer intent | authors' executable code | current code | evidence |
|---|---|---|---|---|---|
| ToyNet input | dynamic 𝒲 (Eq. 13) | [90] rows (static or per-window) | static rows only; dyn = None-switch | [360] static+dyn concat | STRONG (param count: 90-input) |
| dynamic processing | required by Eq. 13 | ambiguous (option C fits) | ABSENT | present (concat) | STRONG (code trace) |
| mean path | unspecified | 400→1, then 90→800 | crashes (1×1 vs 90×800) | 400→800 direct | STRONG (executed) |
| variance path | unspecified | same two-stage | crashes | 400→800 direct | STRONG (executed) |
| latent dim | via counts: 800 | 800 | 800 | 800 | STRONG |
| decoder | unspecified | 800→90 | 800→90 (per-ROI loop) | 800→90 | MATCH |
| parameter count | 422k/420k/415k | **415,292 / 422,076 / 424,189** | same layers | 1,018,490 | **DECISIVE** |
| ROI aggregation | unspecified | mean over 90 | mean over 90 | mean over 90 | MATCH |
| KL aggregation | Eq. 14 ambiguous | avg-first | avg-first | avg-first | MATCH AUTHORS |
| edge-logit production | 8100 contract | UNRESOLVABLE under two-stage | per-ROI loop (broken fwd) | per-ROI loop (working) | AMBIGUOUS |
| 40-TR windows | 40 TR exact | — | — | 96-second adaptation | ABIDE_ADAPTATION |

## 8. Final verdict

```
CURRENT_TOYNET_INTERNAL_CONSISTENCY          = PASS
CURRENT_TOYNET_PAPER_FIDELITY                = FAIL
    (parameter count 2.45x the paper's own published accounting, which the
     two-stage 90->800 architecture reproduces to 0.02-1%)

AUTHORS_RELEASED_FORWARD                     = BROKEN
AUTHORS_LAYER_PARAM_COUNT                    = 415,292
CURRENT_TOYNET_PARAM_COUNT                   = 1,018,490
PAPER_MAIN_PARAM_COUNT                       = ~422k  (reconstructed: 422,076)
PAPER_SHARED_PARAM_COUNT                     = ~415k  (reconstructed: 415,292)

PARAM_COUNT_SUPPORTS_TWO_STAGE_90_TO_800     = YES

CURRENT_STATIC_DYNAMIC_CONCAT_PAPER_SUPPORTED = AMBIGUOUS
    (dynamic half supported by Eq. 13; static half unmentioned; note: only
     option C -- shared per-window [90] processing -- satisfies Eq. 13 AND
     the parameter-count-matching input_dim=90 simultaneously)

CURRENT_DYNAMIC_WINDOWS_PAPER_EXACT          = NO   (96s adaptation; only TR=2.4 sites = 40 TR)

CURRENT_KL_AGGREGATION_MATCHES_AUTHORS       = YES
CURRENT_KL_AGGREGATION_PAPER_STATUS          = AMBIGUOUS

SAFE_TO_FREEZE_STAGE3                        = NO
SAFE_TO_EDIT_GRASTI_PY                       = YES
    (a confirmed, quantified paper-fidelity error exists: 2.45x parameter
     count; the paper-consistent target architecture is now precisely known --
     but per instruction, NO edit until this reconstruction is reviewed)
```
