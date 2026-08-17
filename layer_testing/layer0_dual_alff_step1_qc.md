# Step 1 QC — Dual ALFF Recompute (4 pilot subjects)

Per the approved plan: this is QC only, not a result. Correlations below are reported for
transparency but must not be interpreted as the real Method A vs. B comparison (that's Step 5,
>=100 subjects). This run includes four safety guards added after review: per-subject
affine/shape assertion, AAL-label validation, finite-vs-zero voxel-validity distinction
(fail loudly on non-finite, silently exclude exact-zero), and CSV-vs-NIfTI-header TR
cross-check.

Pilot subjects: `Caltech_0051456`, `NYU_0050996`, `UM_1_0050278` (swapped in for
`UM_1_0050279`, which is not part of this project's actual 956-subject cohort), `Yale_0050612`.

## Safety guard results

| Guard | Result |
|---|---|
| Atlas labels 1-90 present, all non-empty | Pass — full unique label list printed (0-116), 90/90 validated |
| Per-subject shape + affine assertion vs. atlas | Pass, all 4 — affine max abs diff = **0.00e+00** (exact, not just within tolerance) |
| CSV TR vs. NIfTI header TR | Pass, all 4 — identical to 6 decimal places (2.000000s both sides), no mismatch |
| Nyquist >= highest band edge (0.08Hz) | Pass, all 4 (TR=2.0s -> Nyquist=0.25Hz) |
| Every band has >=1 FFT bin | Pass, all 4 (enforced in code, would raise otherwise) |
| Non-finite voxels inside any AAL ROI | 0 found, all 4 subjects (would have raised and halted) |
| Zero-coverage ROIs (0 valid voxels) | 0, all 4 subjects (would have raised and halted) |

## Per-subject detail

| Subject | TR (CSV/header) | T | Min ROI coverage % | Commute check (max abs diff) | A raw (min/max/mean) | B raw (min/max/mean) | A vs B corr (QC only) |
|---|---|---|---|---|---|---|---|
| Caltech_0051456 | 2.000000 / 2.000000 | 146 | 12.10% | 1.07e-13 | 1.03 / 14.50 / 3.88 | 3.80 / 18.80 / 9.52 | 0.7904 |
| NYU_0050996 | 2.000000 / 2.000000 | 176 | 39.46% | 2.27e-13 | 2.46 / 14.13 / 6.35 | 7.37 / 25.25 / 16.12 | 0.7888 |
| UM_1_0050278 | 2.000000 / 2.000000 | 296 | 69.82% | 6.82e-13 | 3.77 / 46.35 / 9.79 | 13.25 / 66.54 / 23.53 | 0.8323 |
| Yale_0050612 | 2.000000 / 2.000000 | 196 | 69.82% | 2.56e-13 | 1.70 / 14.58 / 5.11 | 4.03 / 20.22 / 9.13 | 0.8968 |

Min-coverage numbers here (finite-vs-zero-corrected) are consistent with the earlier,
independent atlas-coverage check (e.g. Caltech's worst region was found at 12.1% there too) —
cross-validates rather than contradicts it.

## Interpretable finding (not the Step 5 result, but real)

Method B (voxel-first) is **systematically larger in magnitude** than Method A (roi-first) in
every subject — not noise. Averaging voxels *before* the FFT cancels phase-incoherent,
voxel-level noise (destructive interference); averaging *after* `|FFT|` can't cancel anything,
since sign/phase information is already discarded at that point. Structurally,
`mean(|FFT(x))| >= |FFT(mean(x))|`. Relevant for Step 5/6: the two methods will differ not
just in correlation but in absolute scale, which matters once normalization is layered back in
(Step 6).

Full per-ROI voxel counts + coverage %: `layer_testing/layer0_dual_alff_step1_qc_raw.json`.
Raw Method A/B arrays: `layer_testing/layer0_dual_alff_step1_raw.npz`.

## Verdict

**Step 1 PASSES, with all four added safety guards clean.** Safe to proceed to Step 4 (scale
to >=100 subjects).
