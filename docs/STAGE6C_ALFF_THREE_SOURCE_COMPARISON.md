# STAGE 6C — THREE-SOURCE ALFF COMPARISON (ROI-first / voxel-first / alff_new)

Date: 2026-08-18 · Diagnostics only: no production edits, no training, no Stage 7 ·
Frozen Stage-6/6B methodology (seed-42 weights, signed PCC, sum-pooled GCN2 battery) ·
Common cohort: **954 subjects**, identical IDs/order in all three sources.

## 0. Sources and alignment (verified before analysis)

| source | file | keys | shape | route |
|---|---|---|---|---|
| M1 | ALFF_func_proc/method1/alff_roi_first.npz | file_ids, alff, tr | [954,90,3] | 4D → detrend → ROI-average voxels → FFT → 3-band ALFF |
| M2 | ALFF_func_proc/method2/alff_voxel_first.npz | file_ids, alff, tr | [954,90,3] | 4D → detrend → FFT per voxel → 3-band ALFF → ROI-average |
| new | alff_new/non_combat/alff_new.npz | file_ids, alff, malff, dx_group, ok | [956,90,3] | official rois_aal.1D → this project's ROI-timeseries ALFF script (A-GCL-INSPIRED; **correction, Stage 6D**: the official A-GCL repository documents VOXELWISE DPABI `y_alff_falff` on the 4D BOLD volume — this ROI-first computation is NOT "exact A-GCL ALFF") |

M1 IDs == M2 IDs (same order) ✓. alff_new ∖ M1 = exactly {CMU_b_0050669, Leuven_1_0050706}
(the two zero-ROI func_preproc subjects — excluded, NOT imputed, no .1D substitution) ✓.
alff_new reordered to the M1 954-order for every comparison. All three finite, NaN-free.

## A. Raw diagnostics (954-aligned; ROI cosine/eRank over 100-subject spread)

| src | min | max | mean | ROI cos | cos>.99 | eRank | band means s5/s4/cl |
|---|---|---|---|---|---|---|---|
| M1 | 0.461 | 62.1 | 5.48 | 0.9894 | 0.68 | 1.41 | 6.23/4.99/5.22 |
| M2 | 0.540 | 126.2 | 13.28 | **0.9985** | **0.98** | **1.19** | 13.92/12.88/13.06 |
| new | 0.002 | 51.9 | 4.45 | 0.9896 | 0.68 | 1.41 | 5.05/4.06/4.24 |

Voxel-first inflates amplitude ~2.4× (average of |FFT| ≥ |FFT| of average) and is the most
ROI-homogeneous raw source — voxel-level ALFF averaging smooths regional distinctions.

## B. Cross-method agreement (954 subjects)

| pair | per-subject r (mean/med/range) | band r s5/s4/cl | 270-cell across-subject r | rel L2 |
|---|---|---|---|---|
| M1 vs M2 | 0.757 / 0.772 / [0.32, 0.95] | .766/.733/.749 | 0.721 | 1.410 |
| M1 vs new | 0.727 / 0.728 / [0.42, 0.91] | .706/.711/.705 | **0.906** | 0.312 |
| M2 vs new | 0.574 / 0.583 / [0.06, 0.85] | .561/.552/.553 | 0.687 | 0.670 |

The operator (ROI-first vs voxel-first) is a MATERIAL lever (r≈0.76 on identical 4D input).
M1 and alff_new — both ROI-first, different formulas/inputs — agree strongly in the
across-subject structure (r 0.906) that drives between-subject discrimination.

## C-D. Representation-health battery (12 variants; cohort-wide sensitivities over 24
probe subjects; rel = ‖Δ‖/‖base‖ at sum-pooled R7; rewire = SYMMETRIC off-diag permutation)

| variant | R0cos | R0eR | G1eR | G2eR | subjR | PCC-rewire | PCC-delete | ALFF-rep | ‖base‖ |
|---|---|---|---|---|---|---|---|---|---|
| m1:raw | 0.989 | 1.41 | 1.17 | 1.06 | 1.82 | 0.019 | 0.18 | 0.42 | 700.7 |
| m1:mm01 | 0.964 | 1.54 | 1.23 | 1.08 | 2.34 | 0.022 | 0.19 | 0.36 | 31.6 |
| m1:cent | 0.017 | 1.86 | 4.64 | 2.29 | 5.16 | 0.53 | 8.20 | 0.80 | 23.1 |
| **m1:z** | 0.016 | 1.87 | **4.82** | 2.32 | **5.64** | 0.52 | 8.53 | 0.61 | 10.2 |
| m2:raw | 0.999 | 1.19 | 1.09 | 1.03 | 1.60 | 0.019 | 0.18 | 0.33 | 1700.8 |
| m2:mm01 | 0.982 | 1.35 | 1.15 | 1.05 | 2.27 | 0.021 | 0.18 | 0.36 | 40.2 |
| m2:cent | 0.007 | 1.63 | 4.21 | 2.19 | 3.71 | 0.49 | 8.56 | 1.38 | 34.7 |
| m2:z | 0.007 | 1.63 | 4.28 | 2.20 | 4.60 | 0.48 | 8.66 | 0.63 | 9.4 |
| new:raw | 0.990 | 1.41 | 1.17 | 1.06 | 1.81 | 0.015 | 0.17 | 0.44 | 575.4 |
| new:mm01 | 0.972 | 1.48 | 1.21 | 1.07 | 2.20 | 0.016 | 0.18 | 0.36 | 32.6 |
| new:cent | 0.022 | 1.76 | 4.53 | 2.28 | 5.15 | 0.52 | 8.22 | 0.73 | 21.4 |
| **new:z** | 0.021 | 1.76 | 4.66 | 2.29 | **5.42** | 0.50 | 8.53 | 0.56 | 9.4 |

All finite. Rewire/delete cosines: ~1.000 for raw/mm01 (blind), 0.92-0.96 for cent/z
(responsive). Cohort-wide distributions, not single-subject.

## E. M_ij = σ(v_i·v_j) (50 subjects × 8100 pairs; leading variants)

| variant | min | med | max | mean | %<0.5 | %>0.5 | symmetric | finite |
|---|---|---|---|---|---|---|---|---|
| m1:z | 0.000 | 0.507 | 1.000 | 0.502 | 48.7% | 51.3% | 6e-08 | ✓ |
| m2:z | 0.000 | 0.502 | 1.000 | 0.501 | 49.6% | 50.4% | 6e-08 | ✓ |
| new:z | 0.000 | 0.509 | 1.000 | 0.502 | 48.4% | 51.6% | 0 | ✓ |
| new:cent | 0.000 | 0.537 | 1.000 | 0.512 | 48.4% | 51.6% | 0 | ✓ (saturated tails: p25 0.013 / p75 0.989) |
| new:mm01 | 0.500 | 0.544 | 0.943 | 0.560 | **0.0%** | **100%** | 6e-08 | ✓ (Stage-4 one-sidedness replicated) |

z-variants give a balanced two-sided M with full range — repairing the Stage-4
one-sidedness that [0,1] features force. Centered-raw units saturate the sigmoid.

## FINAL ANSWERS

```
ROI_FIRST_VS_VOXEL_FIRST_DIFFERENCE   = MATERIAL
    (r 0.757 per subject on identical 4D input; rel L2 1.41; and voxel-first is
     consistently the weakest source at FIXED normalization: subjR 4.60 vs 5.64,
     R0 eRank 1.63 vs 1.87 after z)

OLD_ALFF_NEW_MATCHES_NEW_ROI_FIRST    = PARTIAL
    (per-subject r 0.727 — values differ; across-subject structure r 0.906 — strong;
     downstream health statistically equivalent: new:z 5.42 vs m1:z 5.64 with
     near-identical sensitivities)

POSITIVE_NONCENTERED_COLLAPSE_REPLICATES = YES  (all three sources, raw AND [0,1])

CENTERING_RESCUES_METHOD1             = YES
CENTERING_RESCUES_METHOD2             = YES  (weaker: subjR 3.71 vs 5.16)
ZSCORE_RESCUES_METHOD1                = YES  (best overall)
ZSCORE_RESCUES_METHOD2                = YES  (weaker: 4.60 vs 5.64)

ALFF_OPERATOR_PRIMARY_CAUSE           = NO
    (every operator collapses when positive and rescues when centered — the operator
     does not cause the collapse; it is a SECONDARY quality factor favoring ROI-first)

NORMALIZATION_PRIMARY_CAUSE           = YES  (consistent with Stage 6B, now replicated
    across three independent ALFF computations)

BEST_CURRENT_FEATURE_CANDIDATE        = per-subject per-band Z-SCORED ROI-FIRST ALFF
    Primary: m1:z (controlled provenance, TR-validated, best metrics; 954 subjects).
    Equivalent alternative: new:z (full 956 coverage; statistically indistinguishable
    health). Voxel-first (m2) not recommended. Decision NOT based on a single metric:
    m1:z leads or ties on R0/GCN1/GCN2/subject rank, sensitivity balance, and M_ij
    behavior simultaneously.
```

No Dataset.py change, no frozen-cache change, no training. STOP for review.
