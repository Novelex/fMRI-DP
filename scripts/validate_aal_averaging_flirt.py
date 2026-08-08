#!/usr/bin/env python3
"""
Validate AAL region-averaging using the historical C-PAC procedure:
resample func_preproc INTO the (unchanged) atlas grid via FSL FLIRT
(identity transform, trilinear interpolation), then average voxels
within each atlas label at every timepoint. No masking, filtering,
detrending, or standardization -- matches historical 3dROIstats-style
mean extraction.

Compares against the official rois_aal (filt_noglobal) for the same
subject.

Usage:
    python scripts/validate_aal_averaging_flirt.py
"""

import nibabel as nib
import numpy as np

FILE_ID = "Caltech_0051456"
ATLAS_PATH = "data/software/atlases/aal_mask_pad.nii.gz"
FUNC_IN_AAL_PATH = f"data/raw/_validation/{FILE_ID}_func_preproc_in_aal.nii.gz"
ROIS_AAL_PATH = f"data/raw/rois_aal/{FILE_ID}_rois_aal.1D"

N_REGIONS = 116
N_REGIONS_90_CUTOFF = 9001  # AAL-90 = labels < 9001 (cerebellum/vermis are >= 9001)


def main() -> int:
    func_img = nib.load(FUNC_IN_AAL_PATH)
    func_data = func_img.get_fdata()

    atlas_img = nib.load(ATLAS_PATH)
    atlas_data = np.round(atlas_img.get_fdata()).astype(int)

    assert func_data.shape[:3] == atlas_data.shape, (
        f"Grid mismatch: func {func_data.shape[:3]} vs atlas {atlas_data.shape}"
    )

    labels = sorted(int(l) for l in np.unique(atlas_data) if l != 0)
    print(f"Atlas labels found: {len(labels)} (expected {N_REGIONS})")

    n_timepoints = func_data.shape[3]
    ours = np.zeros((n_timepoints, len(labels)))
    missing_labels = []

    for i, label in enumerate(labels):
        roi_voxels = atlas_data == label
        n_voxels = roi_voxels.sum()
        if n_voxels == 0:
            missing_labels.append(label)
            continue
        ours[:, i] = func_data[roi_voxels, :].mean(axis=0)

    official = np.loadtxt(ROIS_AAL_PATH)

    print(f"\nShapes: ours={ours.shape}, official={official.shape}")
    assert ours.shape == official.shape

    correlations = np.array(
        [np.corrcoef(ours[:, i], official[:, i])[0, 1] for i in range(len(labels))]
    )
    abs_diff = np.abs(ours - official)

    labels_arr = np.array(labels)
    aal90_idx = np.where(labels_arr < N_REGIONS_90_CUTOFF)[0]

    print(f"\nAll {len(labels)} ROI correlations: "
          f"min={np.nanmin(correlations):.4f}, "
          f"median={np.nanmedian(correlations):.4f}, "
          f"max={np.nanmax(correlations):.4f}")

    print(f"\nAAL-90 (labels < {N_REGIONS_90_CUTOFF}), n={len(aal90_idx)}: "
          f"min={np.nanmin(correlations[aal90_idx]):.4f}, "
          f"median={np.nanmedian(correlations[aal90_idx]):.4f}, "
          f"max={np.nanmax(correlations[aal90_idx]):.4f}")

    worst_order = np.argsort(np.nan_to_num(correlations, nan=-1))
    print("\nFive lowest-correlating labels:")
    for idx in worst_order[:5]:
        print(f"  label={labels[idx]}, corr={correlations[idx]:.6f}")

    print(f"\nMean abs diff: {abs_diff.mean():.6f}")
    print(f"Max abs diff: {abs_diff.max():.6f}")

    n_matched = int((correlations >= 0.99).sum())
    print(f"\nColumns with correlation >= 0.99: {n_matched}/{len(labels)}")

    print(f"\nMissing labels (zero voxels): {missing_labels if missing_labels else 'none'}")

    aal90_median = np.nanmedian(correlations[aal90_idx])
    aal90_min = np.nanmin(correlations[aal90_idx])

    if not missing_labels and n_matched == N_REGIONS and aal90_median > 0.999 and aal90_min >= 0.99:
        print("\nVALIDATION PASS")
        return 0
    else:
        print("\nVALIDATION: see stats above -- not a clean pass")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
