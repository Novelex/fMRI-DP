# STAGE 8D — TRAINED REPRESENTATION-HEALTH TRAJECTORY (RESULTS)

Date: 2026-08-19 · Repo commit at submission `25b0c23` · 8 arms × 30 epochs, all COMPLETED
· `alff_new_z` (N=956), frozen Stage-8C hyperparameters, reg_lambda 0.2 baseline · No tuning,
no CV, no model change.

## Execution

`STAGE8D_ARRAY_COMPLETE = YES` — all 8 arms COMPLETED with ExitCode 0:0, 30/30 epochs,
checkpoints at {0,1,3,5,10,20,30}, ~45 min each. Config-divergence check passed: arm01
(consistent) vs arm04 (legacy), same seed 42, epoch-1 CL 3.4692 vs 3.6123, |diff| 0.143 →
`PHASE_STATE_MODE_THREADING_CONCERN = NO`.

## The central result

**The contrastive loss never leaves its uninformative fixed point.** With batch size 32,
`log(B−1) = log(31) = 3.4340` is the value L_CL takes when every similarity is equal. Measured
L_CL across all 30 epochs and all 8 arms: **3.43 – 3.46**. It does not descend.

**Uniformity collapses to ~0 by epoch 1 and stays there.** Uniformity
`log E exp(−2‖zi−zj‖²)` → 0 means all pairwise distances → 0.

| arm | e0 | e1 | e5 | e10 | e20 | e30 |
|---|---|---|---|---|---|---|
| arm01 consistent s42 | −0.530 | −0.0004 | −0.023 | −0.005 | −0.0006 | −0.0025 |
| arm04 legacy s42 | −0.530 | −0.152 | −0.235 | −1.146 | −0.004 | −0.0006 |
| arm06 legacy s2024 | −0.364 | −0.955 | −0.000 | −0.476 | −0.009 | −0.232 |

**Positive and negative cosines move together.** arm01 at e30: pos 0.7927, mean-neg 0.7919,
margin **+0.0008**. The Stage-8C "positive cosine improves" signal was global collapse, not
discrimination. Margins across consistent arms stay at 0.0004–0.006 for all 30 epochs.

**Subject effective rank declines** from ~8 to 2.8–3.6 (consistent) and 1.25–8.81 (legacy,
far more erratic).

## Is it caused by `consistent`? NO.

| | rank_z e30 | margin e30 | uniformity e30 |
|---|---|---|---|
| consistent (3 seeds) | 2.77 / 3.07 / 3.62 | 0.0008 / 0.0021 / 0.0016 | −0.0025 / −0.0018 / −0.0021 |
| legacy (3 seeds) | 1.25 / 8.68 / 8.81 | 0.0002 / 0.0351 / 0.0374 | −0.0006 / −0.136 / −0.232 |

Both modes degenerate. Legacy is not healthier — it is **noisier**, with wildly non-monotonic
trajectories (arm06 positive cosine: 0.31 → 0.10 → 0.97 → 0.40 → 0.94 → 0.70 → 0.71). The
Stage-8C correction did not create the decline; it made an already-degenerate process
*reproducible*.

## Φ optimizes correctly — the failure is on the representation side

J_Φ falls monotonically in **every** arm: 9.4 → −1.09 (consistent) / −1.20 (legacy).
KLD 3600 → 57–114. Gate learns modestly and stably: std 0.291 → 0.302, mean ≈ 0.51.
REG behaves as designed: arm08 (reg=1.0) drives gate mean to 0.587 and REG to 0.413.
Clip fires 2–14 % of Phase-1 steps (higher under consistent, as Stage 8C predicted).

So Φ is doing exactly what the audited objective asks. **The contrastive branch is what fails.**

## Layer localization (Section 20 — identify only, do NOT fix)

arm01, epoch0 → epoch30, subject effective rank:

| stage | e0 | e30 | change |
|---|---|---|---|
| R0 (input) | 2.97 | 2.97 | +0.00 |
| GCN1 | 5.52 | 4.96 | −0.56 |
| **GCN2** | 5.48 | **2.73** | **−2.75** |
| X_atte | 11.57 | 8.51 | −3.07 |
| TAE fusion | 5.49 | 2.76 | −2.73 |
| pooling | 5.49 | 2.76 | −2.73 |
| projection | 8.18 | 2.77 | −5.41 |

`ADDITIONAL_COMPRESSION_FIRST_APPEARS_AT = GCN2`. The input is untouched by training (by
construction); GCN1 barely moves; the second GCN layer is where trained compression first
appears and it propagates unchanged through fusion and pooling.

## Supporting results

**CPU baselines** (956 subjects, 5 folds × 3 seeds, train-only scaling and C selection):
FC upper-triangle **0.663** balanced accuracy, FC+ALFF **0.657**, ALFF alone **0.591** —
consistent with the historical 64–68 % classical range.

**Stage-6 train-mode-BN recheck = PARTIAL.** Under train statistics the Stage-6 collapse is
milder but real: GCN1 eRank 5.00→5.63, GCN2 3.00→3.54, ROI cosine at GCN2 0.822→0.573,
subject rank 5.29→8.25. Stage-6's findings are therefore not pure eval-BN artifacts, but they
were overstated in magnitude.

## FINAL BLOCK

```
STAGE8D_ARRAY_COMPLETE = YES  (8/8 arms, 30/30 epochs, ExitCode 0:0)
PHASE_STATE_MODE_THREADING_CONCERN = NO
REPRESENTATION_HEALTH_CLASSIFICATION = PERSISTENT_DEGENERATION
DEGENERATION_SPECIFIC_TO_CONSISTENT = NO  (present in matched legacy seeds too)
CONTRASTIVE_LOSS_LEAVES_FIXED_POINT = NO  (L_CL stays at log(B-1)=3.434 for 30 epochs)
GATE_LEARNS = YES but weakly (std 0.291 -> 0.302, mean ~0.51, stable)
PHI_OBJECTIVE_OPTIMIZES = YES  (J_Phi 9.4 -> -1.09; KLD 3600 -> ~60-114)
ADDITIONAL_COMPRESSION_FIRST_APPEARS_AT = GCN2
STAGE6_FINDINGS_SURVIVE_TRAIN_MODE_BN = PARTIAL
NEED_STRUCTURAL_INVESTIGATION = YES
SAFE_TO_BEGIN_FULL_NESTED_CV = NO
NO_STAGE8E_AUTOMATICALLY = TRUE  (none opened; nothing modified)
```

Nothing was fixed, tuned or redesigned. Stage 8D collected trained evidence only.

---

# Amendment after Stage 8E verification

Nothing above has been deleted. Stage 8E re-derived the Stage-8D numbers from the raw
`metrics.json` files and found two errors in this document's *interpretation*. Both
are corrected here; the original text is left in place so the history is auditable.

## Amendment 1 — the "all arms pinned at log(32)" claim is withdrawn

This document stated that all eight arms sat at CL = 3.43–3.46, i.e. pinned at the
InfoNCE null log(32) = 3.4657.

**That is false.** Re-reading *every* recorded epoch rather than only the checkpoint
epochs {0, 1, 3, 5, 10, 20, 30}:

* only **38.3 %** of recorded checkpoints fall inside 3.43–3.46;
* the observed range across all arms and all epochs is **2.089 – 3.980**.

The claim was a sampling error on my part — I generalised from the checkpointed
epochs to the whole trajectory. The corrected statement is: CL *returns to* the
neighbourhood of log(32) at the checkpointed epochs, with excursions of roughly
±1.4 nats in between.

## Amendment 2 — the measurement surface was eval-mode only

Every representation metric in this document (CL, uniformity, alignment, effective
rank, margin, top-1) was produced by an **eval-mode** probe. The Phase-2 objective is
optimised in **train** mode, and Stage 8E measured that the two differ radically:

| arm | epoch | TRAIN CL | EVAL CL |
|---|---|---|---|
| arm04 legacy seed42 | 30 | **0.133** | 3.436 |
| arm05 legacy seed7 | 30 | **−0.103** | ≈3.44 |

The train-mode values were cross-checked against the Phase-2 `model_loss` recorded
during training, so they are the objective actually descended.

**Consequence for this document's conclusions.** Stage 8D's metrics correctly
diagnose the *downstream geometry* — the representation a linear probe or classifier
would see. They do **not** establish that the training objective failed, and no
sentence in this document may be read as showing the optimiser was stuck at the
null. The optimiser descends its objective to ≈0 and below; what fails is the
**transfer from the train-mode forward surface (batch statistics) to the eval-mode
surface (running statistics)**.

Stage 8E therefore reports every metric in both forward states, and identifies
**λ asymmetry between the two contrastive views** as the primary supported mechanism
(see `docs/STAGE8E_COLLAPSE_MECHANISM_AUDIT.md` §7, an intervention at frozen weights
that moves posRank from the exact null 16.53/32 to 4.38/32 and top-1 from 0.031 to
0.531 without changing a single weight).

## What is NOT amended

* The Stage-8D arm matrix, seeds, commit provenance, checkpoints and raw
  `metrics.json` files are unchanged and remain the evidence base.
* The finding that the degeneration **persists to epoch 30 in every arm** stands —
  it is measured on the eval surface, which is the surface that matters downstream.
* The finding that the failure is **not specific to `--phase_state_mode consistent`**
  stands.
