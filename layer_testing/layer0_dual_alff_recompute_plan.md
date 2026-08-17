# Dual ALFF Recomputation — ROI-first vs. Voxel-first, Controlled

Extends Layer 0. Goal: isolate "order of operations" (average-then-FFT vs. FFT-then-average)
as the **only** variable, by holding everything else fixed — same source data, same atlas,
same FFT code. Cleaner than the existing `alff_new` (ROI-first) vs. old `norm_matrix`
(voxel-first) comparison, which confounds order-of-operations with three other differences:
different atlas/grid, different tooling (CPAC `.1D` vs. DPARSFA), different final
normalization (raw vs. z-scored mALFF). This design directly targets that: same C-PAC
`nofilt_noglobal` 4D, same AAL90 voxel sets, same TR, same FFT/bands, same Python ALFF
implementation — the only thing that changes is average-before-FFT vs. FFT-before-average.

Revision note: this plan went through a review pass; four corrections and an approved 7-step
sequence are folded in below (superseding the single-pass version this file originally had).

## Correction 1 — detrend at the same point in the pipeline

Original draft detrended *after* averaging for ROI-first and *before* averaging for
voxel-first — a second, hidden variable on top of order-of-operations. Fixed: detrend every
voxel first, from one shared detrended voxel matrix, then branch:

```
func_preproc 4D
  -> detrend EACH voxel first
  -> same detrended voxel matrix
       |-------------------------|
       v                         v
  average voxels             FFT voxels
       v                         v
  FFT -> ROI ALFF            ALFF voxel -> average -> ROI ALFF
     (Method A)                  (Method B)
```

Now the only non-commuting operation left is `|F(mean(x))|` vs. `mean(|F(x)|)` — the cleanest
possible version of this experiment. Separately verify `detrend(mean(voxels))` ≈
`mean(detrend(voxels))` to floating tolerance, as a check that this simplification is valid
(linear detrending should commute with averaging, but confirm rather than assume).

## Correction 2 — identical voxel sets in both methods

For each subject and ROI, define one fixed voxel index set = `AAL90 ROI ∩ valid functional
coverage`, and use those exact same indices for both Method A and Method B. Method A must not
average over 142 voxels while Method B averages a different 137 — that would reintroduce a
second confound. Since the atlas/functional grid affine and all-90-ROI coverage are already
confirmed matching (Layer 0), no resampling is needed — just make sure both methods read from
the same index set. Report the voxel count per ROI for all 4 pilot subjects as part of Step 1.

## Correction 3 — raw vs. raw only, no normalization yet

Save `roi_first_raw [subjects,90,3]` and `voxel_first_raw [subjects,90,3]` and compare those
directly. Do not apply z-score, mALFF, min-max, or ComBat inside this comparison — that mixes
the "which operator" question with the "which normalization" question. Once the raw-vs-raw
question is answered, derive identical transforms from *both* raw arrays afterward (raw /
joint-[0,1] / per-band-centered / per-band-z-score), so operator effects and normalization
effects stay separable rather than entangled.

## Correction 4 — don't overclaim from matching the historical r≈0.66

Original draft: "if the controlled comparison also lands ~0.66, that's evidence
order-of-operations is *the dominant driver*" — too strong. Pearson correlations from two
different experiments aren't an additive causal decomposition; two different mechanisms could
independently produce similarly-sized disagreement by coincidence. Corrected framing:

- Controlled r ≈ historical r (≈0.66): "consistent with order-of-operations being a major
  contributor" — suggestive, not proof.
- Controlled r much *higher* than historical r (e.g. controlled ≈0.95 vs. historical ≈0.66):
  this is the actually strong inference — it bounds what order-of-operations *can't* explain.
  If isolating it alone barely moves the needle (0.95 = high agreement) but the full historical
  comparison shows a much bigger gap, something else (atlas/tooling/normalization) must be
  responsible for most of that extra disagreement. A process-of-elimination argument, not a
  magnitude-matching one — this is the more informative outcome to watch for, not the
  matching-magnitudes case.

## Additional checks folded in

- TR must be read correctly per-subject (`subject_tr.csv`) and used to build the frequency
  vector — verify per-subject, not assumed constant.
- Bands stay exactly `0.010-0.027`, `0.027-0.073`, `0.010-0.080` Hz (already verified against
  both `alff_new` and the A-GCL paper text).
- On the full-sample run, report more than just the 270 (ROI,band) Pearson correlations:
  mean/median/range, **plus** per-subject correlation, cosine similarity, absolute L2
  difference, relative L2 difference, **plus** ROI-first/voxel-first ratio by band. Then run
  Stage-6 geometry tests (this repo's existing representation-survival diagnostics) on both
  raw arrays.
- No silent truncation: if any subject is dropped (missing TR, failed FFT, etc.), log it
  explicitly rather than silently shrinking the sample.

## Final approved plan (7 steps)

1. **4 subjects** (`Caltech_0051456`, `NYU_0050996`, `UM_1_0050279`, `Yale_0050612`) — QC
   only. Verify: same atlas/func affine, all 90 ROI voxel sets present and voxel-count-matched
   between methods, subject-specific TR read correctly, voxelwise detrending correct, finite
   data, no NaN. **Do not interpret correlations from 4 subjects** — this step is QC, not a
   result.
2. From the same detrended voxels, compute:
   - Method A: mean voxels -> FFT -> ALFF
   - Method B: FFT voxels -> ALFF -> mean voxels
   Save raw A and raw B.
3. Confirm Method A against `alff_new` as a sanity comparison only — high correlation expected,
   exact equality not required (different atlas/ROI-numbering scheme, per the note above).
4. Scale to >=100 subjects, preferably the full 956.
5. Compare raw ROI-first vs. raw voxel-first (full battery of metrics from "Additional checks"
   above, not just mean Pearson r).
6. Only afterward, derive identical transforms from each raw array: `[0,1]`, centered,
   z-score.
7. Run Stage-6 representation tests on both, and determine: operator problem, normalization
   problem, or both.

## Deliverables

- `layer_testing/dual_alff_recompute.py` — implements Method A/B per Corrections 1-2, reusing
  `compute_alff()`'s exact BANDS/nfft logic.
- `layer_testing/layer0_dual_alff_step1_qc.md` — Step 1's 4-subject QC results.
- `layer_testing/layer0_dual_alff_results.md` — full-sample comparison (Steps 4-7), once Step
  1 passes.

Status: approved, proceeding with Step 1.
