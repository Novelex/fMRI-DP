"""Raw data loading + per-outer-fold ComBat harmonization for the nested-CV
deep pipeline runs.

Mirrors datasets/Dataset.py's ADNIDataset._load_group (same file-matching-by-
subject-ID safety, same nan_to_num/meshgrid edge_index construction) but loads
every subject's raw x/edge_weight/dyn_weight into memory up front, so a given
outer fold can harmonize (or not) before building WindowedData objects --
something the disk-cached, pre-built ADNIDataset can't do per-fold.

ComBat mechanics (fit-on-train-only, apply-to-test) and the missing-site
padding workaround are identical to ml_combat_combinations/run_baseline_combat.py.
"""
import os
import os.path as osp

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from neuroHarmonize import harmonizationLearn, harmonizationApply

from datasets.Dataset import WindowedData

DATA_ROOT = "/users/3171356m/muhammad/GraSTIACL/data/GraSTIACL_ABIDE_979/raw"
ALFF_NEW_NPZ = "/users/3171356m/muhammad/GraSTIACL/alff_new/non_combat/alff_new.npz"
PHENOTYPE_CSV = "/users/3171356m/muhammad/GraSTIACL/data/raw/phenotypic_filtered_v2.csv"
SEED = 123


def _subject_ids(folder, suffix):
    return {f.rsplit(suffix, 1)[0]: f for f in os.listdir(folder)}


def _load_group(adj_dir, nf_dir, dw_dir, y):
    adj_files = _subject_ids(adj_dir, "_adj.mat")
    nf_files = _subject_ids(nf_dir, "_nf.mat")
    dw_files = _subject_ids(dw_dir, "_dw.mat")
    subject_ids = sorted(set(adj_files) & set(nf_files) & set(dw_files))
    assert subject_ids == sorted(set(adj_files)) == sorted(set(nf_files)) == sorted(set(dw_files)), (
        f"Subject sets differ across {adj_dir}, {nf_dir}, {dw_dir}"
    )

    x_list, ew_list, dw_list = [], [], []
    for sid in subject_ids:
        nf = sio.loadmat(osp.join(nf_dir, nf_files[sid]))
        x = np.nan_to_num(nf['norm_matrix']).astype(np.float64)
        x_list.append(x)

        adj = sio.loadmat(osp.join(adj_dir, adj_files[sid]))
        pcc_matrix = np.nan_to_num(adj['cropped_matrix']).astype(np.float64)
        ew_list.append(pcc_matrix)

        dw = sio.loadmat(osp.join(dw_dir, dw_files[sid]))
        dw_array = dw['correlation_matrices']
        windows = np.stack([np.nan_to_num(dw_array[j, 0]) for j in range(dw_array.shape[0])]).astype(np.float64)
        dw_list.append(windows)

    y_arr = np.full(len(subject_ids), y, dtype=np.int64)
    return subject_ids, x_list, ew_list, dw_list, y_arr


def load_alff_paper_features(subject_ids):
    """Stage-2 integration: the SAME alff_paper recipe as ADNIDataset
    (datasets/Dataset.py), applied to the nested pipeline's subject order.
    Raw ROI-first 3-band ALFF from alff_new.npz matched by FILE_ID, then ONE
    shared per-subject [0,1] min-max over the whole (90,3). Strict loading:
    no nan_to_num, fail loudly (identical asserts to the Stage-1 path)."""
    d = np.load(ALFF_NEW_NPZ, allow_pickle=True)
    assert len(d['file_ids']) == 956
    assert len(set(d['file_ids'].tolist())) == len(d['file_ids'])
    assert d['ok'].all()
    raw_map = {}
    for i, sid in enumerate(d['file_ids']):
        arr = d['alff'][i].astype(np.float64)
        assert arr.shape == (90, 3), f"{sid}: raw ALFF shape {arr.shape}"
        assert np.isfinite(arr).all(), f"{sid}: non-finite raw ALFF"
        raw_map[sid] = arr
    npz_ids = set(d['file_ids'].tolist())
    want = set(subject_ids)
    assert npz_ids == want, (f"alff_paper(nested): subject-set mismatch -- "
                             f"npz-only={sorted(npz_ids - want)[:5]}, "
                             f"loader-only={sorted(want - npz_ids)[:5]}")
    out = []
    for sid in subject_ids:
        a = raw_map[sid]
        lo, hi = a.min(), a.max()
        assert hi > lo, f"{sid}: degenerate ALFF (max==min)"
        x = (a - lo) / (hi - lo)
        assert np.isclose(x.min(), 0.0) and np.isclose(x.max(), 1.0), sid
        out.append(x)
    return np.stack(out)


def load_all_subjects():
    """Returns subject_ids, x_all [N,90,3] (90 ROIs x 3 node features per the
    raw .mat's norm_matrix shape), edge_weight_all [N,90,90], dyn_weight_all
    [N,T,90,90], y_all [N], covars (DataFrame with SITE/AGE/SEX, row order
    matching subject_ids)."""
    ids_asd, x_asd, ew_asd, dw_asd, y_asd = _load_group(
        osp.join(DATA_ROOT, "ASD_ADJ"), osp.join(DATA_ROOT, "ASD_NF"), osp.join(DATA_ROOT, "ASD_DW"), y=1,
    )
    ids_nc, x_nc, ew_nc, dw_nc, y_nc = _load_group(
        osp.join(DATA_ROOT, "NC_ADJ"), osp.join(DATA_ROOT, "NC_NF"), osp.join(DATA_ROOT, "NC_DW"), y=0,
    )

    subject_ids = ids_asd + ids_nc
    x_all = np.stack(x_asd + x_nc)
    ew_all = np.stack(ew_asd + ew_nc)
    dw_all = np.stack(dw_asd + dw_nc)
    y_all = np.concatenate([y_asd, y_nc])

    pheno = pd.read_csv(PHENOTYPE_CSV).set_index("FILE_ID")
    missing = [sid for sid in subject_ids if sid not in pheno.index]
    assert not missing, f"{len(missing)} subjects missing from phenotype csv: {missing[:5]}"
    covars = pheno.loc[subject_ids, ["SITE_ID", "AGE_AT_SCAN", "SEX"]].reset_index(drop=True)
    covars = covars.rename(columns={"SITE_ID": "SITE"})
    covars["AGE_AT_SCAN"] = covars["AGE_AT_SCAN"].astype(np.float64)
    covars["SEX"] = covars["SEX"].astype(np.float64)

    return subject_ids, x_all, ew_all, dw_all, y_all, covars


def harmonization_apply_safe(feats, covars, model):
    """neuroHarmonize's harmonizationApply sizes its site one-hot encoding from
    whichever SITE labels are present in *this* call, not the fixed set
    learned during training, so an outer fold's test split missing one site
    crashes with a matmul shape mismatch. Padding with one dummy row per
    missing site (discarded after transform) fixes the shape without
    affecting real rows, since adjust_data_final processes each site's rows
    independently. Identical to ml_combat_combinations's helper."""
    site_labels = model["SITE_labels"]
    present = set(covars["SITE"].unique())
    missing_sites = [s for s in site_labels if s not in present]
    if not missing_sites:
        return harmonizationApply(feats, covars, model)

    n_pad = len(missing_sites)
    pad_covars = covars.iloc[:1].copy()
    pad_covars = pd.concat([pad_covars] * n_pad, ignore_index=True)
    pad_covars["SITE"] = missing_sites
    covars_padded = pd.concat([covars, pad_covars], ignore_index=True)
    feats_padded = np.vstack([feats, np.zeros((n_pad, feats.shape[1]))])

    result_padded = harmonizationApply(feats_padded, covars_padded, model)
    return result_padded[: len(covars)]


def _harmonize_matrix_feature(mat_all, train_idx, test_idx, covars):
    """mat_all: [N, 90, 90] (or [N, T, 90, 90] pre-flattened to [N, T*90*90]
    by the caller for dyn_weight). Harmonizes the upper-tri entries only (diag
    is dropped for the fit/apply, then re-inserted unchanged on reconstruction
    -- self-correlation is a constant, not a real between-ROI signal, matching
    the convention already used by ml_combat_gPCC/dPCC). Fit on train_idx only,
    apply to test_idx only -- never refit on test. Returns the full [N,90,90]
    array with train rows harmonized-via-fit and test rows harmonized-via-apply.
    """
    n_roi = mat_all.shape[-1]
    iu = np.triu_indices(n_roi, k=1)
    flat_all = mat_all[:, iu[0], iu[1]]  # [N, n_pairs]

    covars_train = covars.iloc[train_idx].reset_index(drop=True)
    covars_test = covars.iloc[test_idx].reset_index(drop=True)

    model, flat_train_h = harmonizationLearn(flat_all[train_idx], covars_train, seed=SEED)
    flat_test_h = harmonization_apply_safe(flat_all[test_idx], covars_test, model)

    out = mat_all.copy()
    for local_i, subj_i in enumerate(train_idx):
        out[subj_i, iu[0], iu[1]] = flat_train_h[local_i]
        out[subj_i, iu[1], iu[0]] = flat_train_h[local_i]
    for local_i, subj_i in enumerate(test_idx):
        out[subj_i, iu[0], iu[1]] = flat_test_h[local_i]
        out[subj_i, iu[1], iu[0]] = flat_test_h[local_i]
    return out


def harmonize_fold(x_all, ew_all, dw_all, covars, train_idx, test_idx, combat_on):
    """Returns (x_h, ew_h, dw_h) for ALL subjects (train harmonized via fit,
    test harmonized via apply-only) if combat_on, else the raw arrays
    unchanged. x_all: [N,90,3] (per-ROI node features, no upper-tri concept).
    ew_all: [N,90,90] (static/global PCC). dw_all: [N,T,90,90] (windowed PCC).
    """
    if not combat_on:
        return x_all, ew_all, dw_all

    covars_train = covars.iloc[train_idx].reset_index(drop=True)
    covars_test = covars.iloc[test_idx].reset_index(drop=True)

    # ALFF/node features: [N,90,3] -- flatten to [N,270] for ComBat (all
    # values are real per-ROI-per-feature signal, no upper-tri concept here),
    # then reshape back.
    n_subj, n_roi, n_feat = x_all.shape
    x_flat_all = x_all.reshape(n_subj, n_roi * n_feat)
    x_model, x_train_h = harmonizationLearn(x_flat_all[train_idx], covars_train, seed=SEED)
    x_test_h = harmonization_apply_safe(x_flat_all[test_idx], covars_test, x_model)
    x_flat_h = x_flat_all.copy()
    x_flat_h[train_idx] = x_train_h
    x_flat_h[test_idx] = x_test_h
    x_h = x_flat_h.reshape(n_subj, n_roi, n_feat)

    # Static/global PCC: upper-tri only, mirrored back, diagonal untouched.
    ew_h = _harmonize_matrix_feature(ew_all, train_idx, test_idx, covars)

    # Windowed/dynamic PCC: flatten all T windows' upper-tri values into one
    # ComBat fit (ComBat corrects each feature column independently, so one
    # call across T*n_pairs columns is equivalent to fitting each window
    # separately) then reshape back into [N,T,90,90].
    n_subj, T, n_roi, _ = dw_all.shape
    iu = np.triu_indices(n_roi, k=1)
    dw_flat_all = dw_all[:, :, iu[0], iu[1]].reshape(n_subj, T * len(iu[0]))  # [N, T*n_pairs]

    dw_model, dw_train_h = harmonizationLearn(dw_flat_all[train_idx], covars_train, seed=SEED)
    dw_test_h = harmonization_apply_safe(dw_flat_all[test_idx], covars_test, dw_model)

    dw_h = dw_all.copy()
    n_pairs = len(iu[0])
    for local_i, subj_i in enumerate(train_idx):
        for t in range(T):
            vals = dw_train_h[local_i, t * n_pairs:(t + 1) * n_pairs]
            dw_h[subj_i, t, iu[0], iu[1]] = vals
            dw_h[subj_i, t, iu[1], iu[0]] = vals
    for local_i, subj_i in enumerate(test_idx):
        for t in range(T):
            vals = dw_test_h[local_i, t * n_pairs:(t + 1) * n_pairs]
            dw_h[subj_i, t, iu[0], iu[1]] = vals
            dw_h[subj_i, t, iu[1], iu[0]] = vals

    return x_h, ew_h, dw_h


def compute_alff_pcc_scale_stats(x_all, ew_all, train_idx):
    """Population-wide min/max for alff_pcc mode (Dataset.py's approach)
    computed on train_idx ONLY, never test_idx -- unlike the single-run
    ADNIDataset pipeline (which fits population-wide across every subject,
    train+val+test alike, a real transductive leak for anything claiming to
    be a held-out generalization estimate), nested CV's whole reason to exist
    is producing a number that's actually clean of exactly this kind of
    leakage, so the fold's test subjects must never influence these stats.
    Returns (alff_min, alff_max, pcc_min, pcc_max), each a python float.
    """
    train_alff = x_all[train_idx]
    train_pcc = ew_all[train_idx]
    return (float(train_alff.min()), float(train_alff.max()),
            float(train_pcc.min()), float(train_pcc.max()))


def build_windowed_data_list(indices, x_all, ew_all, dw_all, y_all, node_feature_mode='alff',
                              alff_min=None, alff_max=None, pcc_min=None, pcc_max=None):
    n_roi = x_all.shape[1]  # x_all is [N, 90, 3] -- ROI count, not feature count
    row, col = np.meshgrid(np.arange(n_roi), np.arange(n_roi), indexing='ij')
    edge_index = torch.tensor(np.stack([row.flatten(), col.flatten()]), dtype=torch.long)

    # 'alff_paper' arrives here with x_all ALREADY substituted+normalised by
    # load_alff_paper_features (run_nested_cv.py does it right after loading,
    # BEFORE any ComBat/fold transformation) -- so it is treated like 'alff':
    # x used as-is, no further scaling.
    assert node_feature_mode in ('alff', 'alff_pcc', 'alff_paper')
    if node_feature_mode == 'alff_pcc':
        assert None not in (alff_min, alff_max, pcc_min, pcc_max), (
            "alff_pcc mode requires scale stats from compute_alff_pcc_scale_stats "
            "(fit on train_idx only), not recomputed per-split here."
        )

    data_list = []
    for i in indices:
        if node_feature_mode == 'alff_pcc':
            # Signed [-1,1] rescale, matching pcc_scaled and datasets/Dataset.py's
            # identical fix -- [0,1] made every dot product v_i.v_j >= 0, so
            # mij_source='alff''s M_ij (Eq. 4) could never read below 0.5.
            alff_scaled = 2 * (x_all[i] - alff_min) / (alff_max - alff_min + 1e-12) - 1
            pcc_scaled = 2 * (ew_all[i] - pcc_min) / (pcc_max - pcc_min + 1e-12) - 1
            x = torch.tensor(np.concatenate([alff_scaled, pcc_scaled], axis=1), dtype=torch.float)
        else:
            x = torch.tensor(x_all[i], dtype=torch.float)
        # edge_weight always raw PCC, unaffected by node_feature_mode -- same
        # convention as datasets/Dataset.py.
        edge_weight = torch.tensor(ew_all[i].flatten(), dtype=torch.float)
        # y must be shape [1,1] per subject, matching the real ADNIDataset
        # pipeline's actual per-sample shape after set_tu_dataset_y_shape's
        # unsqueeze(1) (verified directly: dataset[0].y.shape == [1,1]), so
        # that after PyG batches N subjects, batch.y comes out [N,1].
        y = torch.tensor([[int(y_all[i])]])

        data = WindowedData(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        data.dyn_weight = torch.tensor(dw_all[i], dtype=torch.float)
        data_list.append(data)
    return data_list
