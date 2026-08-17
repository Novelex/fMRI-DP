# STAGE 3 — PAPER-INTENT RECONSTRUCTION OF ToyNet (FINAL RESOLUTION)

Date: 2026-08-17 · Read-only stage: zero production edits · Seed 42 · No training, no accuracy.
Supersedes the earlier version of this document, whose fidelity claims exceeded the evidence
(corrected per final Stage-3 resolution; see "Two constraints" below).

## The two simultaneously true constraints

**1. Parameter-count constraint.** The authors' released two-stage layer definitions give
ToyNet = **415,292** parameters and a reconstructed main model ≈ **422,076** (view module
424,189), closely reproducing the paper's reported main ≈ 422k / auxiliary ≈ 420k /
shared ≈ 415k. The published parameter accounting therefore **strongly reflects these
released layer dimensions**.

**2. Edge-output constraint.** The authors' own `get_mu_std_logits` loop processes 90 ROI
rows × 90 decoder logits to generate **8100 logits per subject** (the full 90×90 adjacency).
But the natural executable reconstruction of the two-stage 415k layers —
`[90,90] → [90,400] → 400→1 → collect [90] → 90→800 → decode 800→90` — produces only
**90 logits per subject**, not 8100.

**Therefore: the parameter-count match does NOT uniquely reconstruct the successful paper
forward architecture.** The released layer dimensions are strongly supported by the published
parameter accounting, but their intended executable tensor flow cannot be uniquely
reconstructed, because the natural two-stage implementation produces 90 outputs while the
released per-ROI augmentation path requires 8100. The public source contains an
**irreducible provenance/implementation inconsistency**. (The released `forward()` itself is
BROKEN in execution: `RuntimeError, 1x1 vs 90x800`, proven by running it.)

## Current production ToyNet — preserved unchanged

Status (all previously proven, 30/30):
internally consistent · produces 8100 correctly ordered logits (bitwise-aligned to
`edge_index` across all positions) · static PCC numerically active · dynamic PCC numerically
active · subject-isolated · finite mu/std (std strictly positive) · finite gradients
including the dynamic route.

But: input 360 (= static [90] + 3 dynamic rows [270]) and **1,018,490 parameters** (2.45×
the paper's accounting). Its correct classification is therefore:

**`EXECUTABLE_ABIDE_CORRECTED_VIB` — not `EXACT_PAPER_TOYNET`.**

## Dynamic input status

Paper: Eq. 13 conditions the node IB on dynamic 𝒲 ∈ R^{T×N×N} → **dynamic PCC is required
by the paper.** But the paper does not specify concatenating static [90] + all dynamic T×90
into [360]. The current concatenation is **PAPER_AMBIGUOUS / ABIDE_IMPLEMENTATION_CHOICE**,
not paper-exact.

## Window status

Paper: non-overlapping 40-TR windows. Current: 3 non-overlapping 96-second windows
(round(96/TR) volumes; only TR=2.4s sites coincide with 40 TR). Kept unchanged — it is the
already-documented variable-TR ABIDE adaptation. **`CURRENT_DYNAMIC_WINDOWS =
ABIDE_ADAPTATION`**, not paper-exact.

## KL

Kept unchanged. Current rule (average ROI mu/std before KL) matches the authors' released
rule; the paper's Eq. 14 does not specify the neural latent parameterization. Measured
current-vs-per-ROI-KL ratios on two real subjects: **1.000** and **1.001**.

## Supporting measurements (unchanged from the audit)

- Authors' released `forward()` executed on its own loop's input: RAISES (broken).
- Dead paths: `weight_init()` never called (raises if invoked); `num_sample>1` branch dead
  (would softmax logits); bare `.squeeze()` no-op; `softplus(var−5)` small-init documented.
- Full decision table and 30/30 internal-consistency results: `docs/STAGE3_VIB_AUDIT.md`.

## CAPACITY LIMITATION

```
CURRENT_TOYNET_PARAM_COUNT      = 1,018,490
AUTHORS_LAYER_PARAM_COUNT       = 415,292
CURRENT_TOYNET_VS_AUTHORS_RATIO = 2.452x
```

The current ToyNet has substantially greater absolute capacity than the
paper-accounted layer configuration; paper-tuned optimization/regularization
hyperparameters may therefore not transfer quantitatively.

CORRECTED (cross-stage parameter-usage audit, measured on two real subjects,
seed 42, exact Phase-1/Phase-2 flows): both wrappers instantiate equally
enlarged ToyNets, but the current training path bypasses `model.net`
(`dyn_weight=None` in both phases: zero gradient in both, zero parameter
delta after both optimizer steps) while `view.net` drives the learned
augmentation (Phase-1 grad norm 1.187, delta 0.5045 after the real step).
Therefore INSTANTIATED parameter symmetry does NOT imply effective
optimization-capacity symmetry. Measured effective optimized capacity:

```
STRUCTURAL (instantiated):
  model.encoder 4,672 · model.net 1,018,490 · model.proj_head 2,112
  view.encoder  4,672 · view.net  1,018,490 · view.mlp_edge_model 4,225

EFFECTIVE (receiving real optimizer updates in the current path):
  Theta (model_optimizer): encoder + proj_head = 6,784   (model.net: 0 updates)
  Phi   (view_optimizer):  view.net = 1,018,490          (view.encoder + mlp: 0 updates
                                                          in the dyn branch)
```

The consequences of this measured asymmetry are deferred to the
training/min-max audit. No underfitting/overfitting inference is drawn from
these counts alone.

## FINAL STAGE-3 BLOCK

```
CURRENT_TOYNET_INTERNAL_CONSISTENCY =
    PASS

AUTHORS_RELEASED_FORWARD =
    BROKEN

AUTHORS_LAYER_PARAM_COUNT =
    415292

CURRENT_TOYNET_PARAM_COUNT =
    1018490

PAPER_PARAMETER_ACCOUNTING_MATCHES_RELEASED_LAYER_DEFINITIONS =
    YES

NATURAL_415K_RECONSTRUCTION_OUTPUTS_PER_SUBJECT =
    90

REQUIRED_EDGE_LOGITS_PER_SUBJECT =
    8100

RELEASED_LAYER_DEFINITIONS_NATURALLY_SATISFY_8100_EDGE_CONTRACT =
    NO

PAPER_INTENDED_EXECUTABLE_TOYNET =
    UNRESOLVED_FROM_PUBLIC_SOURCES

PUBLIC_SOURCE_PROVENANCE_LIMIT =
    YES

DYNAMIC_PCC_PAPER_SUPPORTED =
    YES

CURRENT_STATIC_DYNAMIC_360_CONCAT =
    ABIDE_IMPLEMENTATION_CHOICE

CURRENT_DYNAMIC_WINDOWS =
    ABIDE_ADAPTATION

CURRENT_KL_AGGREGATION_MATCHES_AUTHORS =
    YES

CURRENT_KL_AGGREGATION_PAPER_STATUS =
    AMBIGUOUS

CURRENT_TOYNET_STATUS =
    EXECUTABLE_ABIDE_CORRECTED_VIB

SAFE_TO_REPLACE_CURRENT_TOYNET_WITH_415K_RECONSTRUCTION =
    NO

SAFE_TO_FREEZE_STAGE3_AS_EXACT_PAPER_REPRODUCTION =
    NO

SAFE_TO_CLOSE_STAGE3_AS_DOCUMENTED_ABIDE_ADAPTATION =
    YES
```
