"""
Layer 0 -- Dual ALFF recomputation (ROI-first vs. voxel-first), controlled.

See layer0_dual_alff_recompute_plan.md for the full design and the four
corrections this implementation follows, plus the four safety guards added
after review (affine check per subject, AAL-label validation, finite-vs-zero
voxel-validity distinction, TR header cross-check). Step 1 only: 4-subject QC
pass, no correlation interpretation (that's Steps 4-7, a separate run once
this passes).

All paths point at this project's own data root -- nothing outside this repo
and its designated scratch data directory is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.signal import detrend

DATA_ROOT = Path("/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data")
ATLAS_PATH = DATA_ROOT / "software/DPABI/Templates/AAL_61x73x61_YCG.nii"
FUNC_DIR = DATA_ROOT / "raw/func_preproc"
TR_TABLE_PATH = DATA_ROOT / "raw/phenotypic_filtered_v2.csv"

# Same band definitions verified earlier this session against alff_new and
# the A-GCL paper text (Sec 2.1): slow-5, slow-4, classical.
BANDS = [(0.010, 0.027), (0.027, 0.073), (0.010, 0.080)]
N_ROIS = 90
TR_MISMATCH_TOL_SEC = 0.01  # CSV vs NIfTI-header TR must agree within this

# UM_1_0050279 (an earlier pilot candidate) is not part of this project's
# actual 956-subject cohort (phenotypic_filtered_v2.csv doesn't contain it).
# UM_1_0050278 is used instead -- confirmed intentional, not a typo.
PILOT_SUBJECTS = ["Caltech_0051456", "NYU_0050996", "UM_1_0050278", "Yale_0050612"]


def load_atlas(atlas_path: Path):
    atlas_img = nib.load(str(atlas_path))
    atlas_data = np.round(atlas_img.get_fdata()).astype(int)

    # Safety guard 2: validate AAL labels once, up front, not assumed.
    unique_labels = np.unique(atlas_data)
    print(f"  atlas unique labels (incl. background 0): {unique_labels.tolist()}")
    missing = [i for i in range(1, N_ROIS + 1) if i not in unique_labels]
    if missing:
        raise ValueError(f"Atlas missing AAL90 labels: {missing}")
    for i in range(1, N_ROIS + 1):
        if int((atlas_data == i).sum()) == 0:
            raise ValueError(f"AAL ROI {i} has zero voxels")

    roi_masks = {label: (atlas_data == label) for label in range(1, N_ROIS + 1)}
    return atlas_img, roi_masks


def get_tr_seconds(subject_tr: pd.DataFrame, file_id: str) -> float:
    row = subject_tr.loc[subject_tr["FILE_ID"] == file_id]
    if len(row) != 1:
        raise ValueError(f"{file_id}: expected exactly 1 row in TR table, found {len(row)}")
    return float(row["TR_seconds"].iloc[0])


def alff_from_timeseries(ts: np.ndarray, tr: float) -> np.ndarray:
    """ts: [T, K] (K = num ROIs for Method A, or num voxels for Method B's
    per-voxel FFT). Returns [K, 3]. Assumes ts is ALREADY detrended --
    Correction 1: detrending happens once, upstream, on the shared voxel
    matrix, not separately per method."""
    n = ts.shape[0]
    nfft = 2 ** int(np.ceil(np.log2(n)))
    amp = 2 * np.abs(np.fft.rfft(ts, n=nfft, axis=0)) / n
    freqs = np.fft.rfftfreq(nfft, d=tr)
    alff = np.zeros((ts.shape[1], 3))
    for b, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs <= hi)
        if not m.any():
            raise ValueError(f"Band {BANDS[b]} contains 0 FFT bins (T={n}, tr={tr}) -- "
                              f"cannot compute ALFF for this band")
        alff[:, b] = amp[m].mean(axis=0)
    return alff


def process_subject(file_id: str, atlas_img, roi_masks: dict[int, np.ndarray],
                     subject_tr: pd.DataFrame) -> dict:
    func_path = FUNC_DIR / f"{file_id}_func_preproc.nii.gz"
    if not func_path.exists():
        raise FileNotFoundError(f"{file_id}: {func_path} not found")

    tr_csv = get_tr_seconds(subject_tr, file_id)
    func_img = nib.load(str(func_path))

    # Safety guard 1: explicit shape + affine check against the atlas, per
    # subject -- not just assumed from the earlier 4-subject atlas QC.
    if func_img.shape[:3] != atlas_img.shape:
        raise ValueError(f"{file_id}: spatial shape mismatch: func={func_img.shape[:3]}, "
                          f"atlas={atlas_img.shape}")
    affine_max_diff = float(np.max(np.abs(func_img.affine - atlas_img.affine)))
    if not np.allclose(func_img.affine, atlas_img.affine, atol=1e-5):
        raise ValueError(f"{file_id}: functional and AAL atlas affines do not match "
                          f"(max abs diff {affine_max_diff:.2e})")

    # Safety guard 4: cross-check CSV TR against the NIfTI header. CSV stays
    # authoritative (it's what alff_new was computed from), but a material
    # disagreement is a hard stop, not a silent pick.
    tr_header = float(func_img.header.get_zooms()[3])
    if abs(tr_csv - tr_header) > TR_MISMATCH_TOL_SEC:
        raise ValueError(f"{file_id}: TR mismatch -- CSV={tr_csv:.6f}s, "
                          f"NIfTI header={tr_header:.6f}s (tol={TR_MISMATCH_TOL_SEC}s)")
    tr = tr_csv

    func_data = func_img.get_fdata()  # [61,73,61,T]
    T = func_data.shape[3]

    nyquist = 1.0 / (2.0 * tr)
    if nyquist < BANDS[-1][1]:
        raise ValueError(f"{file_id}: Nyquist frequency {nyquist:.4f}Hz below the highest "
                          f"band edge {BANDS[-1][1]}Hz (tr={tr})")

    # Safety guard 3: finite-vs-zero are different things. Exactly-zero is
    # expected (background/skull-strip mask) and silently excluded. Non-finite
    # (NaN/Inf) inside an AAL ROI is NOT expected -- fail loudly rather than
    # silently letting it into detrend().
    union_roi_mask = np.zeros(atlas_img.shape, dtype=bool)
    for mask in roi_masks.values():
        union_roi_mask |= mask
    nonfinite_in_rois = ~np.all(np.isfinite(func_data[union_roi_mask]), axis=-1)
    if nonfinite_in_rois.any():
        raise ValueError(f"{file_id}: {int(nonfinite_in_rois.sum())} non-finite voxel(s) "
                          f"found inside AAL ROI territory -- investigate before proceeding")

    finite_voxel = np.all(np.isfinite(func_data), axis=3)
    nonzero_voxel = np.any(func_data != 0, axis=3)
    voxel_valid = finite_voxel & nonzero_voxel  # [61,73,61]

    # Correction 2: one fixed voxel set per ROI = AAL90 ROI intersect valid
    # functional coverage. Used identically by both Method A and Method B.
    roi_voxel_counts = {}
    roi_time_series_valid = {}  # label -> [n_valid_voxels, T]
    for label, mask in roi_masks.items():
        full_mask = mask & voxel_valid
        n_total = int(mask.sum())
        n_valid = int(full_mask.sum())
        roi_voxel_counts[label] = {
            "atlas_voxels": n_total,
            "valid_voxels": n_valid,
            "coverage_pct": 100.0 * n_valid / n_total if n_total else 0.0,
        }
        roi_time_series_valid[label] = func_data[full_mask]  # [n_valid_voxels, T]

    zero_coverage_rois = [l for l, c in roi_voxel_counts.items() if c["valid_voxels"] == 0]
    if zero_coverage_rois:
        raise ValueError(f"{file_id}: ROI(s) with zero valid voxels: {zero_coverage_rois}")

    # Correction 1: detrend each ROI's voxel matrix (each voxel independently
    # -- detrending is a per-voxel-timeseries operation, so this is
    # equivalent to detrending one big shared matrix, just looped per ROI)
    # ONCE, before branching into Method A / Method B.
    detrended_by_roi = {}
    for label, vox_ts in roi_time_series_valid.items():
        ts_T = vox_ts.T  # [T, n_valid_voxels]
        detrended_by_roi[label] = detrend(ts_T, axis=0)

    # Commutativity check (Correction 1's stated verification):
    # detrend(mean(voxels)) ~= mean(detrend(voxels))  per ROI, per timepoint
    commute_max_abs_diff = 0.0
    for label, det_vox in detrended_by_roi.items():
        mean_of_detrended = det_vox.mean(axis=1)  # [T]
        raw_roi_ts = roi_time_series_valid[label].mean(axis=0)  # [T], raw mean before detrend
        detrend_of_mean = detrend(raw_roi_ts)
        diff = np.max(np.abs(mean_of_detrended - detrend_of_mean))
        commute_max_abs_diff = max(commute_max_abs_diff, float(diff))

    # Method A: mean voxels -> FFT -> ALFF (ROI-first)
    method_a = np.zeros((N_ROIS, 3))
    # Method B: FFT voxels -> ALFF -> mean voxels (voxel-first)
    method_b = np.zeros((N_ROIS, 3))

    for label in range(1, N_ROIS + 1):
        det_vox = detrended_by_roi[label]  # [T, n_valid_voxels]
        roi_mean_ts = det_vox.mean(axis=1, keepdims=True)  # [T, 1]
        method_a[label - 1] = alff_from_timeseries(roi_mean_ts, tr)[0]

        voxelwise_alff = alff_from_timeseries(det_vox, tr)  # [n_valid_voxels, 3]
        method_b[label - 1] = voxelwise_alff.mean(axis=0)

    return {
        "file_id": file_id,
        "tr_csv": tr_csv,
        "tr_header": tr_header,
        "affine_max_diff": affine_max_diff,
        "T": T,
        "roi_voxel_counts": roi_voxel_counts,
        "commute_max_abs_diff": commute_max_abs_diff,
        "method_a_raw": method_a,
        "method_b_raw": method_b,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Dual ALFF recompute (ROI-first vs voxel-first).")
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Space-separated FILE_IDs to process. Default: the 4 Step-1 "
                              "pilot subjects (Caltech_0051456, NYU_0050996, UM_1_0050278, "
                              "Yale_0050612).")
    parser.add_argument("--chunk-index", type=int, default=None,
                         help="Process every --num-chunks'th subject starting at this index "
                              "(0-based), from the full sorted 956-subject list -- strided, "
                              "not contiguous, so same-site subjects (similar scan length) "
                              "don't cluster into one uneven chunk. Requires --num-chunks.")
    parser.add_argument("--num-chunks", type=int, default=None,
                         help="Total number of chunks for --chunk-index striding.")
    parser.add_argument("--output-suffix", default="",
                         help="Suffix appended to output filenames, e.g. '_slurm_pilot'. "
                              "Auto-set to '_chunkNNN' when --chunk-index is used and this "
                              "is left blank.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading TR table: {TR_TABLE_PATH}")
    subject_tr = pd.read_csv(TR_TABLE_PATH)
    print(f"  {len(subject_tr)} subjects")

    output_suffix = args.output_suffix
    if args.chunk_index is not None:
        if args.num_chunks is None:
            raise ValueError("--chunk-index requires --num-chunks")
        all_ids = sorted(subject_tr["FILE_ID"].tolist())
        subjects = all_ids[args.chunk_index::args.num_chunks]
        if not output_suffix:
            output_suffix = f"_chunk{args.chunk_index:03d}"
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = PILOT_SUBJECTS

    print(f"Loading atlas: {ATLAS_PATH}")
    atlas_img, roi_masks = load_atlas(ATLAS_PATH)
    print(f"  {len(roi_masks)} ROI masks loaded, all validated non-empty")

    print(f"Processing {len(subjects)} subject(s): {subjects[:5]}{'...' if len(subjects) > 5 else ''}")

    results = []
    failures = []
    for file_id in subjects:
        print(f"\n=== {file_id} ===")
        try:
            r = process_subject(file_id, atlas_img, roi_masks, subject_tr)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append({"file_id": file_id, "error": str(exc)})
            continue
        results.append(r)

        a, b = r["method_a_raw"], r["method_b_raw"]
        vc = r["roi_voxel_counts"]
        coverages = [c["coverage_pct"] for c in vc.values()]
        zero_cov = sum(1 for c in coverages if c == 0.0)

        print(f"  func shape={FUNC_DIR}/{file_id}_func_preproc.nii.gz spatial dims match atlas: "
              f"True (asserted)")
        print(f"  affine max abs diff: {r['affine_max_diff']:.2e}")
        print(f"  TR: CSV={r['tr_csv']:.6f}s, NIfTI header={r['tr_header']:.6f}s")
        print(f"  T={r['T']}")
        print(f"  min ROI coverage: {min(coverages):.2f}%  |  zero-coverage ROIs: {zero_cov}")
        print(f"  commute check (detrend/mean order) max abs diff: {r['commute_max_abs_diff']:.2e}")
        print(f"  Method A finite: {np.isfinite(a).all()}  |  Method B finite: {np.isfinite(b).all()}")
        print(f"  Method A raw ALFF: min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")
        print(f"  Method B raw ALFF: min={b.min():.4f} max={b.max():.4f} mean={b.mean():.4f}")

        r_ab = np.corrcoef(a.flatten(), b.flatten())[0, 1]
        print(f"  Method A vs B flattened correlation (QC signal only, NOT the Step 5 result): {r_ab:.4f}")
        r["a_vs_b_corr_qc_only"] = float(r_ab)
        r["min_coverage_pct"] = float(min(coverages))
        r["zero_coverage_rois"] = zero_cov

    print(f"\n{len(results)}/{len(subjects)} succeeded, {len(failures)} failed.")
    if failures:
        print("FAILURES (not silently dropped -- logged here and in the QC json):")
        for f in failures:
            print(f"  {f['file_id']}: {f['error']}")

    out_path = Path(__file__).parent / f"layer0_dual_alff_step1_raw{output_suffix}.npz"
    if results:
        np.savez(
            out_path,
            file_ids=np.array([r["file_id"] for r in results]),
            method_a_raw=np.stack([r["method_a_raw"] for r in results]),
            method_b_raw=np.stack([r["method_b_raw"] for r in results]),
            tr_csv=np.array([r["tr_csv"] for r in results]),
            tr_header=np.array([r["tr_header"] for r in results]),
            commute_max_abs_diffs=np.array([r["commute_max_abs_diff"] for r in results]),
            failed_file_ids=np.array([f["file_id"] for f in failures]),
        )
        print(f"\nSaved raw arrays: {out_path}")
    else:
        print("\nNo subjects succeeded -- not writing an empty output file.")

    qc_summary = {
        "succeeded": {
            r["file_id"]: {
                "tr_csv": r["tr_csv"],
                "tr_header": r["tr_header"],
                "affine_max_diff": r["affine_max_diff"],
                "T": r["T"],
                "min_coverage_pct": r["min_coverage_pct"],
                "zero_coverage_rois": r["zero_coverage_rois"],
                "commute_max_abs_diff": r["commute_max_abs_diff"],
                "method_a_finite": bool(np.isfinite(r["method_a_raw"]).all()),
                "method_b_finite": bool(np.isfinite(r["method_b_raw"]).all()),
                "a_vs_b_corr_qc_only": r["a_vs_b_corr_qc_only"],
                "roi_voxel_counts": r["roi_voxel_counts"],
            }
            for r in results
        },
        "failed": failures,
    }
    qc_path = Path(__file__).parent / f"layer0_dual_alff_step1_qc_raw{output_suffix}.json"
    with open(qc_path, "w") as f:
        json.dump(qc_summary, f, indent=2)
    print(f"Saved QC summary: {qc_path}")


if __name__ == "__main__":
    main()
