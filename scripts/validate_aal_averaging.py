#!/usr/bin/env python3
"""
Validate our own AAL-atlas region-averaging against C-PAC's official rois_aal.

For one subject: take the raw func_preproc (filt_noglobal), resample the
official C-PAC AAL atlas (aal_mask_pad.nii.gz, from FCP-INDI/C-PAC_templates)
onto its grid, mask with func_mask, average each of the 116 AAL regions per
timepoint, and compare against that subject's official rois_aal (filt_noglobal).

If this doesn't match closely, the atlas/masking/averaging code is wrong and
nothing built on top of rois_aal-derived features can be trusted.

Usage:
    python scripts/validate_aal_averaging.py
"""

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

FILE_ID = "Caltech_0051456"
ATLAS_PATH = "data/software/atlases/aal_mask_pad.nii.gz"
FUNC_PATH = f"data/raw/_validation/{FILE_ID}_func_preproc_filt_noglobal.nii.gz"
MASK_PATH = f"data/raw/func_mask/{FILE_ID}_func_mask.nii.gz"
ROIS_AAL_PATH = f"data/raw/rois_aal/{FILE_ID}_rois_aal.1D"

N_REGIONS = 116


def main() -> int:
    func_img = nib.load(FUNC_PATH)
    func_data = func_img.get_fdata()
    mask_data = nib.load(MASK_PATH).get_fdata()

    atlas_img = nib.load(ATLAS_PATH)
    atlas_resampled = resample_from_to(atlas_img, (func_img.shape[:3], func_img.affine), order=0)
    atlas_data = np.round(atlas_resampled.get_fdata()).astype(int)

    labels = sorted(int(l) for l in np.unique(atlas_data) if l != 0)
    print(f"Resampled atlas: {len(labels)} non-background labels (expected {N_REGIONS})")
    assert len(labels) == N_REGIONS, f"Expected {N_REGIONS} labels, got {len(labels)}"

    n_timepoints = func_data.shape[3]
    our_timeseries = np.zeros((n_timepoints, N_REGIONS))

    brain_mask = mask_data > 0

    for i, label in enumerate(labels):
        roi_mask = (atlas_data == label) & brain_mask
        n_voxels = roi_mask.sum()
        if n_voxels == 0:
            print(f"  WARNING: label {label} has 0 voxels inside func_mask")
            continue
        our_timeseries[:, i] = func_data[roi_mask, :].mean(axis=0)

    official_timeseries = np.loadtxt(ROIS_AAL_PATH)
    print(f"Our timeseries shape: {our_timeseries.shape}, official: {official_timeseries.shape}")
    assert our_timeseries.shape == official_timeseries.shape

    correlations = np.array(
        [
            np.corrcoef(our_timeseries[:, i], official_timeseries[:, i])[0, 1]
            for i in range(N_REGIONS)
        ]
    )

    print(f"\nPer-region correlation: min={correlations.min():.4f}, "
          f"mean={correlations.mean():.4f}, median={np.median(correlations):.4f}")
    print(f"Regions with correlation < 0.99: {(correlations < 0.99).sum()} / {N_REGIONS}")

    if correlations.min() >= 0.99:
        print("\nVALIDATION PASS: our AAL averaging matches the official rois_aal.")
        return 0
    else:
        worst = np.argsort(correlations)[:5]
        print("\nVALIDATION FAIL. Worst regions (index, label, correlation):")
        for idx in worst:
            print(f"  index={idx}, label={labels[idx]}, corr={correlations[idx]:.4f}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
