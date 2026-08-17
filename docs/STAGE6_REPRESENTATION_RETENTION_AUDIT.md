# STAGE 6 — REPRESENTATION RETENTION / INFORMATION-SURVIVAL AUDIT

Date: 2026-08-17 · Stages 1-5 FROZEN (verified: HEAD == origin/main == `469cbab`, clean
tree, gamma-shape guard present, STAGE5_STATUS=FROZEN) · Read-only: ZERO production edits
· No epochs, no accuracy · Seed 42, seed-42 INITIALIZED weights (architecture forward-path
audit, same convention as Stages 2-5) · Primary profile: alff_paper + signed + paper_intent
+ standard pooling + mij alff · Extractor verified BITWISE against the production forward
(R6/R7 == encoder, R8 == GInfoMinMax proj_head).

## Checkpoint map (section 2)

R0 = batch.x raw ALFF [N,3] (= R1: x feeds convs[0] directly, no pre-GCN transform) ·
R2 = relu(bn0(conv0)) · R3 = bn1(conv1) (= R4 = X_topo: 2-layer stack, drop_ratio 0) ·
R5 = trans_conv(x, batch) raw X_atte · R6 = X_update (Eq. 19) · R7 = global_add_pool ·
R8 = proj_head(R7) [32->32->32].

## 3. Shape / finite / scale (B = 1, 2, 16 real subjects)

ALL finite at every B — no NaN/Inf anywhere, no nan_to_num used. Representative (B=16):

| cp | shape | min | max | mean | std | L2mean | L2std | frac==0 |
|---|---|---|---|---|---|---|---|---|
| R0 | [1440,3] | 0 | 1.0 | 0.306 | 0.186 | 0.542 | 0.302 | 0.0037 |
| R2 | [1440,32] | 0 | 0.284 | 0.046 | 0.059 | 0.410 | 0.104 | 0.468 (ReLU) |
| R3 | [1440,32] | −0.244 | 0.193 | 0.007 | 0.074 | 0.408 | 0.104 | 0 |
| R5 | [1440,32] | 0 | 2.483 | 0.407 | 0.649 | 4.331 | 0.038 | 0.602 (ReLU) |
| R6 | [1440,32] | −0.244 | 0.193 | 0.007 | 0.074 | 0.408 | 0.104 | 0 |
| R7 | [16,32] | −16.66 | 13.13 | 0.65 | 6.53 | 36.51 | 6.75 | 0 |
| R8 | [16,32] | −10.80 | 10.97 | 0.56 | 4.56 | 25.55 | 4.86 | 0 |

Scale note: ‖X_atte‖ ≈ 4.33 per node vs ‖X_topo‖ ≈ 0.41 — a 10.6× branch-scale gap
(the Stage-5 Eq.19 materiality, re-confirmed here at the checkpoint level).

## 4 / 12 / 13. Within-subject node diversity, oversmoothing, effective rank (100 subjects)

Pairwise cosine over each subject's 90 ROI vectors (4005 pairs × 100 subjects):

| cp | mean | med | p5 | p95 | >.90 | >.99 | >.999 | featVar | eRank | tolRank |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 0.972 | 0.990 | 0.880 | 1.000 | 0.935 | 0.505 | 0.154 | 0.02744 | 1.48 | 2.8 |
| R2 | 0.987 | 0.9997 | 0.986 | 1.000 | 0.984 | 0.935 | 0.689 | 0.00020 | 1.21 | 2.1 |
| R3 | 0.971 | 1.0000 | 0.995 | 1.000 | 0.981 | 0.964 | 0.875 | 0.00019 | 1.08 | 1.5 |
| R5 | 0.991 | 0.997 | 0.963 | 1.000 | 0.989 | 0.794 | 0.322 | 0.00533 | 1.80 | 4.1 |
| R6 | 0.971 | 1.0000 | 0.995 | 1.000 | 0.981 | 0.964 | 0.875 | 0.00019 | 1.08 | 1.5 |

(eRank = entropy effective rank exp(H(σ/Σσ)); tolRank = #σ > 1%·σmax, secondary.)

Reading: **the input itself is already near-collinear** — R0's 90 ROI vectors sit in the
positive octant of a 3-space ([0,1] shared min-max, three positively correlated ALFF
bands): mean cosine 0.972, eRank 1.48 of a possible 3. GCN layer 1 then amplifies:
per-feature variance drops 137× (0.0274→0.0002), pairs > 0.99 jump 50%→93%. Layer 2
reaches eRank 1.08 with 87.5% of ROI pairs above cosine 0.999 — near rank-1 node geometry.
Mechanism (measured, not assumed): the graph is COMPLETE (90×90 with self-loops), so one
signed-safe normalized aggregation is already a global weighted average per node — on a
complete weighted graph, strong single-layer smoothing is the expected behavior, amplified
by the already-collinear input. The attention branch (R5) retains the MOST diversity of
any learned representation (eRank 1.80, tolRank 4.1) — the softmax mixes but its
input MLP + residual keep more directions alive than complete-graph convolution.

## 6. Stage-to-stage retention (same-dim pairs)

R2→R3: node cosine mean 0.308, rel change 1.18 (layer 2 rotates strongly — BN affine +
signed aggregation). R3→R6: cosine 1.0000, rel 0.0013 (paper_intent γ=1: attention
contributes 0.13% BY DESIGN). R5→R6: cosine −0.065 (attention direction unrelated to
fused output at λ=1e-4). R0→R2 (3→32 dims): compared via the diversity table — the
big diversity drop happens exactly here.

## 5 / 13 / 14. Between-subject diversity — ALL 956 subjects (feasible: 78 s CPU)

| | cosine mean | >.90 | >.99 | >.999 | eucl mean | centered eRank | zero-var dims | exact dup | near-dup pairs |
|---|---|---|---|---|---|---|---|---|---|
| R7 | 0.9911 | 0.9992 | 0.712 | 0.258 | 10.53 | **2.20** (tol 4) | 0/32 | 0 | 269 |
| R8 | 0.9952 | 1.0000 | 0.850 | 0.339 | 7.11 | **2.46** (tol 6) | 0/32 | 0 | 356 |

Top-2 centered singular values carry 95.9% (R7) / 91.9% (R8) of the spectrum. Subjects are
distinguished almost exclusively by position in a ~2-dimensional subspace — dominated by
overall embedding NORM (Euclidean spread is healthy: mean 10.5) while DIRECTION is nearly
common to the whole cohort (cosine 0.991). No exact duplicates, no dead dimensions;
near-duplicate pairs are 269/456,490 = 0.06%.

## 7. ALFF sensitivity (PCC/weights/seed fixed; only ALFF perturbed) — PASS

| perturbation | R2 rel | R3 rel | R7 rel | R8 rel | R7 cos |
|---|---|---|---|---|---|
| A. +0.01·randn (clamped [0,1]) | 0.0035 | 0.0028 | 0.0028 | 0.0026 | 1.0000 |
| B. permute 90 ROI rows | 0.0371 | 0.0119 | 0.0108 | 0.0103 | 1.0000 |
| C. replace with subject B's ALFF | 0.165 | 0.139 | 0.138 | 0.107 | 0.9912 |
| D. strong: uniform random [0,1] | 1.168 | 1.099 | 1.100 | 1.136 | 0.9222 |

Graded, proportionate response — representations DO respond to meaningful ALFF change.
Notable: ROI-permutation (B) barely moves the graph embedding (1.1%) — sum pooling makes
R7 nearly invariant to WHICH ROI carries the ALFF; the cohort embedding largely encodes
each subject's overall ALFF distribution, not its spatial arrangement.

## 8. Static PCC sensitivity (ALFF/dyn fixed; diagnostic copies) — the KEY finding

| perturbation | R2 rel | R3 rel | R7 rel | R7 cos | R8 rel |
|---|---|---|---|---|---|
| A. +0.01 off-diag | 0.0017 | 0.0012 | 0.0002 | 1.0000 | 0.0002 |
| B. identity graph (no off-diag) | 0.592 | 0.601 | **0.0234** | 0.9997 | 0.0210 |
| C. shuffle ALL off-diag weights | 0.116 | 0.111 | **0.0072** | 1.0000 | 0.0063 |
| D. zero off-diag, keep diag | 0.592 | 0.601 | 0.0234 | 0.9997 | 0.0210 |

(Dataset PCC diagonal == 1 → B and D are numerically identical probes.)

The topology branch genuinely uses connectivity at the NODE level (deleting all edges
changes node representations by 60%). But after sum pooling the graph embedding is
**nearly connectivity-blind**: randomly rewiring every edge weight changes R7 by 0.7%;
deleting every off-diagonal edge changes it by 2.3%. Mechanism: normalized aggregation
REDISTRIBUTES mass among near-identical node vectors, and global_add_pool then sums the
redistribution back — with near-collinear nodes, W changes where the mass sits, not the
total that pooling reports. This structurally explains the frozen campaign fact that
classical PCC baselines (64-68%) beat every deep arm (~55%): the pooled deep pipeline
cannot see most of the connectivity signal the classical baselines consume directly.

## 9. Dynamic PCC boundary — PASS both directions

Original path: dyn_weight has no code path into the encoder forward → R2/R3/R4 bitwise
unchanged. Augmented path: perturbing dyn changes the gate (|Δ| mean 0.033, max 0.28)
→ W_aug changes (|Δ| mean 0.012) → augmented representations change (R2 rel 0.014).
Exactly the Stage-2/Stage-3 contract.

## 10. Original vs augmented (64 spread subjects, real training-path gate)

R6 cos(orig,aug) mean 0.092 · R7 cos 0.087 (rel L2 6.0) · R8 cos 0.502. A: 0/64 bitwise
identical ✓ · B: not universally random (all cos > 0, structured) ✓ · C: finite ✓ ·
D (descriptive): own-augmented closer to own-original for 34/64 subjects, mean margin
0.0016 — essentially chance AT INIT. Driver: paper_intent's λ-regime asymmetry — original
view (γ=1) has λ=1e-4 (pure topology) while augmented views (mean gate ≈ 0.5) get λ ≈ 0.5,
and with ‖X_atte‖ ≈ 10.6×‖X_topo‖ the augmented view is attention-DOMINATED. The
orig/aug pair crosses representation regimes. Recorded as a Stage-7 input, NOT audited
against the contrastive loss here.

## 11. paper_intent vs paper_printed (identical weights — bitwise-verified; γ=1, eval)

Attention contribution ‖X_update−X_topo‖/‖X_topo‖: intent 0.0013 vs printed **12.78**.
Between conventions: R6 cosine 0.012, R7 cosine 0.007 (rel 13.2), R8 cosine 0.55
(rel 15.1). The Eq.18 parameter-order ambiguity selects between two nearly ORTHOGONAL
representation families for the original view. Material, documented.

## 15. Subject isolation (eval) — PASS

A alone vs A batched with a RADICALLY altered B (random x, random signed W): R2/R3/R6/R7
bitwise equal; R5/R8 allclose at 1e-6 (float32 batched-matmul reassociation only).

## 16. Train vs eval decomposition (fixed weights, B=4, no steps)

Train seeded → bitwise reproducible; different seed → rel 0.29 (stochastic sources).
Train-vs-eval rel 2.40 total; forcing BatchNorm to eval removes the largest share
(→ 0.65); additionally disabling TransConv dropout leaves Beta-λ-only at 0.85 (the
components interact non-additively through dropout's 1/(1−p) scaling — reported as
measured, not as a linear decomposition). Train-mode BN batch-stat dependence is
EXPECTED BatchNorm behavior, documented separately — NOT graph leakage (eval isolation
is bitwise-PASS above).

## 17. Projection head boundary

R7→R8: cosine concentration rises (0.9911→0.9952) but centered eRank RISES 2.20→2.46,
no dead dims, no duplicates introduced. **The collapse happens BEFORE the projection
head** (input geometry + complete-graph GCN + sum pooling); the head neither causes nor
repairs it.

## 18. Descriptive label diagnostic (no classifier, no accuracy, no tuning)

R7: within-ASD 10.67, within-NC 10.41, between 10.53 — between/within ratio 0.9992.
R8: ratio 0.9993. At seed-42 initialization the embedding geometry carries essentially
zero label-related structure; the ~2-D cone (total-ALFF scale) is label-agnostic.

## 19. First-collapse locations

```
FIRST_MAJOR_INFORMATION_COMPRESSION = R0  (input geometry: eRank 1.48/3, positive-octant
                                           collinearity from [0,1] shared min-max ALFF;
                                           first ARCHITECTURAL amplification at R2)
FIRST_MAJOR_NODE_OVERSMOOTHING      = R2  (GCN layer 1 on a COMPLETE weighted graph:
                                           featVar ÷137, eRank 1.48->1.21; layer 2 -> 1.08)
FIRST_MAJOR_SUBJECT_COLLAPSE        = R7  (first graph-level checkpoint: centered eRank
                                           2.20/32, direction cosine 0.991; driven by R0
                                           geometry + smoothing + sum pooling, not by R7's
                                           own operation alone)
```

Not labeled a bug: every mechanism above is the documented, expected behavior of this
architecture on this data (complete graph + near-collinear nonneg features + sum pooling)
— no implementation defect was found in the forward path.

## 20. Proposed minimal follow-up experiments (PROPOSALS ONLY — nothing changed)

1. Diagnostic-only ALFF re-centering (per-subject mean-center per band) to test how much
   of the R0 collinearity — and everything downstream — is the [0,1] positive-octant
   scaling alone. (Dataset stays frozen; a probe script, not a mode.)
2. Diagnostic-only readout probe (e.g. concat/max/attention readout instead of sum) to
   test whether graph-level connectivity signal survives ANY readout, quantifying how
   much of section 8's blindness is pooling vs node collapse.
3. Stage 7 must audit the contrastive objective under the measured orig/aug λ-regime
   asymmetry (section 10) — the positive pair crosses representation regimes at init.

## 21. FINAL BLOCK

```
R0_RAW_ALFF_DIVERSITY               = PASS (finite, per-subject/ROI distinct; near-collinear
                                      geometry documented: eRank 1.48/3, mean ROI cos 0.972)
GCN1_RETENTION                      = CONCERN (featVar ÷137, ROI-pairs>0.99: 50%->93%)
GCN_DEEP_RETENTION                  = CONCERN (eRank 1.08; 87.5% ROI pairs cos>0.999)
X_TOPO_RETENTION                    = CONCERN (= R3)
X_ATTE_DIVERSITY                    = PASS (most diverse learned repr: eRank 1.80, tolRank 4.1)
EQ19_FUSION_RETENTION               = PASS (adds 0.13% at gamma=1 under paper_intent --
                                      MINIMAL_BY_DESIGN; introduces no additional collapse)
POOLED_SUBJECT_DIVERSITY            = CONCERN (cohort centered eRank 2.20/32; cosine 0.991;
                                      subjects separated mostly by norm in a ~2-D subspace)
PROJECTION_HEAD_DIVERSITY           = PASS (eRank 2.20->2.46; no dead dims; collapse is
                                      upstream of the head)
ALFF_SENSITIVITY                    = PASS (graded response; spatial arrangement nearly
                                      pooled away: ROI-permutation moves R7 only 1.1%)
STATIC_PCC_SENSITIVITY              = PASS at node level (60% change on edge deletion) /
                                      CONCERN at pooled level (R7 moves 0.7% under full
                                      rewiring, 2.3% under full edge deletion)
DYNAMIC_PCC_ORIGINAL_INDEPENDENCE   = PASS (bitwise)
DYNAMIC_PCC_AUGMENTED_SENSITIVITY   = PASS (gate/W_aug/repr all respond)
ORIG_AUG_NOT_IDENTICAL              = PASS (0/64 identical; finite; structured; lambda-regime
                                      asymmetry documented for Stage 7)
SUBJECT_ISOLATION                   = PASS (eval bitwise/1e-6)
EFFECTIVE_RANK_COLLAPSE             = YES -- node level from R0 (input) amplified at R2/R3;
                                      subject level visible at R7 (2.20/32)
FIRST_MAJOR_INFORMATION_COMPRESSION = R0 (input geometry), first architectural
                                      amplification R2
STAGE6_CONFIRMED_ARCHITECTURE_BUG   = NO (no implementation defect; documented STRUCTURAL
                                      mechanisms: complete-graph smoothing + positive-octant
                                      input + sum-pooling connectivity blindness)
SAFE_TO_FREEZE_STAGE6               = YES
```
