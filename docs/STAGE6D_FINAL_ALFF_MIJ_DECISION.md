# STAGE 6D — FINAL ALFF + Mij DECISION AUDIT

Date: 2026-08-18 · Diagnostics only: NO production edits, NO Stage 7, NO full training ·
Common 954 cohort (M1/M2 IDs identical; alff_new minus exactly CMU_b_0050669 +
Leuven_1_0050706, excluded not imputed) · Multi-seed battery: seeds 0-4, 100
deterministic PCC-perturbation probe subjects, percentile reporting.

## Source-level corrections adopted (override old assumptions)

- A-GCL paper: 3-band ALFF, ONE shared min/max → [0,1]. A-GCL OFFICIAL REPO: fMRIPrep +
  voxelwise DPABI `y_alff_falff` on the 4D BOLD — therefore our ROI-first alff_new is
  NOT "exact A-GCL ALFF" (Stage-6C doc corrected). A-GCL has NO GraSTI Mij/BCE target.
- GraSTI paper: DPARSF, AAL90, slow5/slow4/classical, Eq.4 Mij = σ(v_i·v_j) with v =
  3-band ALFF. Exact operator/normalization NOT disclosed.
- GraSTI release: loads precomputed norm_matrix; its ViewLearner CE target uses LEARNED
  node-embedding dot products, NOT raw ALFF → PAPER Mij ≠ AUTHORS-RELEASE Mij.
- Stage-6C forward collapse is CE-independent (forward-only) — Mij is a possible SECOND
  (training-time) issue, audited here.

```
GRASTI_ALFF_OPERATOR      = PUBLICLY_UNRESOLVED
GRASTI_ALFF_NORMALIZATION = PUBLICLY_UNRESOLVED
GRASTI_PAPER_MIJ          = RAW NODE-FEATURE DOT PRODUCT (Eq. 4)
GRASTI_RELEASE_MIJ        = LEARNED NODE-EMBEDDING DOT PRODUCT
A_GCL_MIJ                 = NOT USED
ALFF_PAPER_MODE_STATUS    = A-GCL-LIKE shared-[0,1] NORMALIZATION CONTROL on our ALFF
                            source -- NOT "exact A-GCL preprocessing"
```

## B. Normalization set + malff algebra (proven)

Added: JOINT_CENTER (one scalar mean over 270) and JOINT_Z (scalar mean+std).
malff = alff / c_s with c_s = per-subject mean(alff), exact to 5e-15. Mathematically
duplicate variants NOT re-run: shared01(malff)≡shared01(alff); per-band-z(malff)≡
per-band-z(alff); joint_z(malff)≡joint_z(alff); centered(malff) = centered(alff)/c_s.
No log/rank/quantile transforms added.

## C. Band-information audit (954 subjects)

Per-band z forces every per-subject band-std ratio to EXACTLY 1 and every band mean to 0.
Those quantities carry real cross-subject variation (raw s5/s4 std-ratio ≈ 1.47 ± 0.29
for M1; ratios preserved bit-exactly by shared01 / joint_center / joint_z). Joint
variants keep band means (M1 j_cent: +0.75 / −0.49 / −0.26) and all scale ratios.

**DOES_PER_BAND_Z_ERASE_INFORMATION_THAT_JOINT_Z_PRESERVES = YES**
(band means + between-band scale ratios — whether that information is USEFUL is section G).

## D. Multi-seed representation health (seeds 0-4; median [p5/p25/p75/p95] over 500 probe-runs)

| config | R0cos | R0eR | G1eR | G2eR | subjR (±sd) | PCC sym-rewire | PCC delete | ALFF replace |
|---|---|---|---|---|---|---|---|---|
| **m1:z** | 0.016 | 1.87 | **4.67** | 2.27 | **5.44±0.41** | 0.536 [.33/.44/.62/.71] | 7.39 [4.3/5.6/10.3/17.1] | 0.44 |
| m1:jz | 0.173 | 2.01 | 3.38 | 1.64 | 4.40±0.52 | 0.138 [.02/.08/.24/.48] | 1.94 [0.6/1.2/3.0/5.2] | 0.51 |
| m2:z | 0.007 | 1.63 | 4.11 | 2.17 | 4.40±0.32 | 0.502 | 8.28 | 0.43 |
| m2:jz | 0.133 | 1.80 | 3.58 | 1.73 | 4.60±0.43 | 0.182 | 2.95 | 0.66 |
| new:z | 0.021 | 1.76 | 4.50 | 2.25 | 5.22±0.38 | 0.508 | 8.39 | 0.42 |
| new:jz | 0.138 | 1.90 | 3.53 | 1.69 | 4.65±0.54 | 0.155 | 2.46 | 0.54 |
| new:mm01 | 0.972 | 1.48 | 1.20 | 1.06 | 2.41±0.32 | 0.015 | 0.11 | 0.19 |
| m1:cent | 0.017 | 1.86 | 4.51 | 2.25 | 5.05±0.30 | 0.544 | 7.20 | 0.58 |

Perturbation cosines: z/cent ≈ 0.94 (responsive); mm01 ≈ 1.000 (blind); jz delete-cos
0.79-0.83 but rewire-cos 0.99 (mass-sensitive, rearrangement-weak — the preserved band
offsets keep a positive common component). Stable across seeds; all finite.

## E. Mij semantics (MEASURED; 50 subjects × 8100 pairs)

| target | %>0.5 | extremes (<.01 / >.99) | entropy | r(PCC) | r(\|PCC\|) |
|---|---|---|---|---|---|
| M_PAPER shared01 | **100.0%** | 0 / 0 (range [0.5, 0.95]) | 0.678 | +0.00…+0.07 | +0.03…+0.09 |
| M_PAPER pb_cent | 51.6% | **19.5% / 21.4%** (saturated) | 0.26-0.27 | +0.06…+0.08 | +0.06…+0.08 |
| M_PAPER pb_z | 51.5% | 4.3% / 3.4% | 0.48-0.49 | +0.06…+0.08 | +0.07…+0.09 |
| M_PAPER j_z | 57-59% | 3.4% / 3.7% | 0.49-0.50 | +0.07…+0.09 | +0.08…+0.10 |
| M_AUTHORS (any feat) | **97.6-98.5%** | 0 / 0 (range ≈ [σ(−1), σ(1)]) | 0.58-0.61 | **+0.39…+0.58** | +0.28…+0.46 |

Authors-release M was NOT assumed two-sided — measured: it is ONE-SIDED (near-collinear
embeddings → cosine ≈ 1 → M ≈ 0.73), bounded, but tracks PCC far better than feature-dot M.
Diagonal 0.731 = σ(1) exactly (normalized embeddings). All symmetric ≤1e-7, finite.
M_NONE: control only — no BCE constraint; NOT GraSTI.

## F. CE one-step deterministic test (exact executable CE; identical noise; one Adam step)

| target | features | CE | grad pushes gate UP | Δ mean gate |
|---|---|---|---|---|
| M_PAPER | shared01 | 1.009 | **56.5%** | **+0.0119** |
| M_PAPER | pb_z | 1.006 | 50.7% | +0.0023 |
| M_PAPER | j_z | 1.009 | 55.4% | +0.0116 |
| M_PAPER | pb_cent | 1.007 | 51.1% | +0.0031 |
| M_AUTHORS | any of the four | 1.014 | **69-73%** | **+0.0161** |

```
DOES_SHARED01_PAPER_MIJ_PUSH_GATE_UNIFORMLY_UPWARD = YES (majority-direction, +0.012/step,
    structural: every target > 0.5)
DOES_AUTHORS_MIJ_REMOVE_ONE_SIDED_TARGET_PROBLEM   = NO  (measured: STRONGER upward push)
IS_ZSCORE_ONLY_LOOKING_GOOD_BECAUSE_IT_REPAIRS_PAPER_MIJ = NO (Stage-6C collapse is
    forward-only/CE-independent; z fixes geometry AND the CE target -- two separate,
    separately measured benefits)
```

## G. Development-signal audit (fixed dev folds, fixed LinearSVC(C=1), fold-internal scaler;
DEVELOPMENT diagnostics only, outer test untouched)

| features (M1) | DX balAcc | DX AUC | SITE balAcc (chance ≈ 0.053) |
|---|---|---|---|
| 3 band means | 0.513 | 0.514 | 0.073 |
| 3 band stds | 0.483 | 0.479 | 0.073 |
| means+stds | 0.508 | 0.508 | 0.133 |
| flat shared01 | 0.558 | 0.574 | 0.526 |
| flat pb_cent | 0.542 | 0.571 | 0.470 |
| flat pb_z | 0.547 | 0.568 | 0.514 |
| flat j_z | 0.554 | 0.573 | 0.515 |

The very quantities per-band z erases (band means/scales) probe at CHANCE for diagnosis
and near-chance for site. Diagnosis signal lives in the spatial pattern and survives
every normalization roughly equally.

## H. GraSTI integration (finalists; 64 subjects; real training-path gate)

| finalist | X_topo eR | X_atte eR | fused eR | R7 subjR(64) | authors-release topo eR | orig-aug cos / relL2 / own>stranger |
|---|---|---|---|---|---|---|
| m1:pb_z | 2.38 | **8.01** | 2.44 | 5.31 | 2.51 | 0.286 / 16.6 / 40/64 (+0.008) |
| m1:j_z | 1.73 | 7.26 | 1.75 | 3.93 | 1.89 | 0.369 / 4.9 / **55/64 (+0.032)** |
| new:pb_z | 2.35 | 7.51 | 2.42 | 5.02 | 2.47 | 0.239 / 18.3 / 34/64 (+0.007) |

z features raise the attention branch to node-rank ~8 (vs 1.8 under [0,1]). The
paper_intent λ-regime asymmetry still dominates orig-vs-aug (Stage-7 material, unchanged).

## J. FINAL DECISION TABLE

```
BEST_SOURCE_FOR_PAPER_FIDELITY   = M2 (voxel-first: A-GCL official is voxelwise DPABI;
                                   GraSTI used DPARSF whose mALFF chain is voxelwise;
                                   GraSTI's exact operator remains PUBLICLY_UNRESOLVED)
BEST_SOURCE_FOR_ABIDE_PERFORMANCE= M1 (multi-seed best at fixed normalization; NEW
                                   statistically equivalent, with full-956 coverage;
                                   M1 costs 2 subjects: 954)
BEST_NORMALIZATION_FOR_REPRESENTATION = per-band z (subjR 5.44 vs joint-z 4.40;
                                   rewire sensitivity 0.54 vs 0.14 -- multi-seed)
BEST_NORMALIZATION_FOR_BAND_INFORMATION = joint_z (preserves band means + scale ratios)
PER_BAND_Z_REMOVES_USEFUL_DIAGNOSIS_SIGNAL = NO (removed quantities probe at chance, G)
JOINT_Z_RESCUES_GEOMETRY         = YES (partially: 4.4-4.65, below per-band z)
JOINT_Z_PRESERVES_MORE_BAND_INFORMATION = YES
PAPER_MIJ_HEALTHY_WITH_SHARED01  = FAIL     (100% > 0.5; ~zero PCC correlation)
PAPER_MIJ_HEALTHY_WITH_Z         = PASS     (48.5/51.5; ~4% saturated tails noted)
AUTHORS_MIJ_HEALTHY_WITH_SHARED01= FAIL     (98.5% > 0.5, measured)
AUTHORS_MIJ_HEALTHY_WITH_Z       = FAIL     (97.9% > 0.5, measured; bounded and
                                   PCC-correlated, but CE pushes 69.6% of gates up)
Mij_PRIMARY_TRAINING_RISK        = YES under the frozen shared01 configuration
                                   (measured one-sided CE pressure on every gate);
                                   NEUTRALIZED by per-band-z features (50.7% balanced)
FINAL_CORRECTED_FEATURE          = M1 (controlled ROI-first) + PER-SUBJECT PER-BAND
                                   Z-SCORE. Not predecided: pb_z wins representation
                                   (multi-seed), Mij balance, and CE neutrality; its
                                   only cost (band means/scales) probes at chance for
                                   diagnosis. Runner-up documented: j_z (band-info
                                   preservation + best orig-aug identity), and new:z
                                   when full-956 coverage is required.
FINAL_MIJ_SEMANTICS              = PAPER (Eq. 4 fixed non-learnable feature-dot target;
                                   balanced under z; authors-release target measured
                                   one-sided AND is a moving learned target)
READY_TO_INTEGRATE_PRODUCTION    = YES (as a certified recommendation -- integration
                                   itself awaits explicit approval; Dataset.py and
                                   caches untouched)
```

STOP. No Dataset.py edits, no cache replacement, no Stage 7, no full epochs.
