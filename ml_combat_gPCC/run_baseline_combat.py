"""Classical ML baseline on Global PCC features, with ComBat site harmonization.

Same 7 algorithms and nested 5-fold stratified CV as ml/run_baseline.py, but
restricted to the global_pcc feature set, with a ComBat harmonization step
(site=batch, age+sex=covariates, diagnosis excluded -- "No Target" scheme)
fit fresh inside every training split so no test subject's data ever
contributes to the harmonization model.

For the sklearn/XGBoost algorithms, ComBat is the first step of the
cross-validated Pipeline, so GridSearchCV refits it per inner fold exactly
like the StandardScaler already is (same leak-free pattern as
ml/run_baseline.py's run_sklearn_nested_cv). For the MLP, ComBat is fit once
per outer-train fold (mirroring how that function already handles its
scaler).

Usage:
    python run_baseline_combat.py --algorithm elasticnet
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from neuroHarmonize import harmonizationLearn, harmonizationApply
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

DATA_ROOT = "/users/3171356m/muhammad/GraSTIACL/data/GraSTIACL_ABIDE_979/raw"
PHENOTYPE_CSV = "/users/3171356m/muhammad/GraSTIACL/data/raw/phenotypic_filtered_v2.csv"
SEED = 123
N_ROI = 90
N_COVAR_COLS = 3  # SITE, AGE_AT_SCAN, SEX appended after the feature columns


# --------------------------------------------------------------------------- #
# Data loading -- global_pcc only, file-matching-by-ID (never os.listdir()
# position -- see ISSUES.md), plus SITE/AGE/SEX covariates joined by FILE_ID.
# --------------------------------------------------------------------------- #

def _subject_ids(folder, suffix):
    return {f.rsplit(suffix, 1)[0]: f for f in os.listdir(folder)}


def _upper_tri(mat):
    iu = np.triu_indices(N_ROI, k=1)
    return mat[iu]


def _load_group(adj_dir, nf_dir, dw_dir, y):
    adj_files = _subject_ids(adj_dir, "_adj.mat")
    nf_files = _subject_ids(nf_dir, "_nf.mat")
    dw_files = _subject_ids(dw_dir, "_dw.mat")
    subject_ids = sorted(set(adj_files) & set(nf_files) & set(dw_files))
    assert subject_ids == sorted(set(adj_files)) == sorted(set(nf_files)) == sorted(set(dw_files)), (
        f"Subject sets differ across {adj_dir}, {nf_dir}, {dw_dir}"
    )

    rows = []
    for sid in subject_ids:
        mat = sio.loadmat(osp.join(adj_dir, adj_files[sid]))
        pcc = np.nan_to_num(mat["cropped_matrix"])
        rows.append(_upper_tri(pcc))
    X = np.stack(rows)
    y_arr = np.full(len(subject_ids), y, dtype=np.int64)
    return X, y_arr, subject_ids


def load_features_and_covars():
    X_asd, y_asd, ids_asd = _load_group(
        osp.join(DATA_ROOT, "ASD_ADJ"), osp.join(DATA_ROOT, "ASD_NF"), osp.join(DATA_ROOT, "ASD_DW"), y=1,
    )
    X_nc, y_nc, ids_nc = _load_group(
        osp.join(DATA_ROOT, "NC_ADJ"), osp.join(DATA_ROOT, "NC_NF"), osp.join(DATA_ROOT, "NC_DW"), y=0,
    )
    X = np.concatenate([X_asd, X_nc], axis=0)
    y = np.concatenate([y_asd, y_nc], axis=0)
    subject_ids = ids_asd + ids_nc

    pheno = pd.read_csv(PHENOTYPE_CSV).set_index("FILE_ID")
    missing = [sid for sid in subject_ids if sid not in pheno.index]
    assert not missing, f"{len(missing)} subjects missing from phenotype csv: {missing[:5]}"
    covars = pheno.loc[subject_ids, ["SITE_ID", "AGE_AT_SCAN", "SEX"]].reset_index(drop=True)
    covars = covars.rename(columns={"SITE_ID": "SITE"})
    return X, y, covars


# --------------------------------------------------------------------------- #
# ComBat as an sklearn-compatible Pipeline step. Covariates travel alongside
# the feature columns (appended on the right) so that row alignment survives
# CV slicing automatically; fit/transform split them back apart internally
# and only the harmonized features are passed downstream.
# --------------------------------------------------------------------------- #

class ComBatHarmonizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=None):
        self.n_features = n_features

    def _split(self, X):
        n_feat = self.n_features
        feats = X[:, :n_feat].astype(np.float64)
        covars = pd.DataFrame(X[:, n_feat:], columns=["SITE", "AGE_AT_SCAN", "SEX"])
        covars["AGE_AT_SCAN"] = covars["AGE_AT_SCAN"].astype(np.float64)
        covars["SEX"] = covars["SEX"].astype(np.float64)
        return feats, covars

    def fit(self, X, y=None):
        feats, covars = self._split(X)
        self.model_, _ = harmonizationLearn(feats, covars, seed=SEED)
        return self

    def transform(self, X):
        feats, covars = self._split(X)
        return harmonization_apply_safe(feats, covars, self.model_)


def _pack_with_covars(X, covars_df):
    return np.hstack([X.astype(np.float64), covars_df[["SITE", "AGE_AT_SCAN", "SEX"]].values])


def harmonization_apply_safe(feats, covars, model):
    """neuroHarmonize's harmonizationApply builds its site one-hot encoding
    from whichever SITE labels are present in *this* call, not from the
    fixed set learned during training (it uses np.unique() on the passed
    batch column, sized by however many distinct sites happen to be
    present). If a CV fold happens to contain zero subjects from one of
    the training sites, the design matrix comes out a column short and
    harmonizationApply crashes with a matmul shape mismatch.

    Workaround: pad with one dummy row per missing site so every trained
    site is represented, transform, then drop the padding rows. adjust_data_final
    processes each site's rows independently (see neuroCombat.adjust_data_final),
    so the padding rows cannot influence the real rows' output -- this is a
    shape fix, not a statistical one.
    """
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


# --------------------------------------------------------------------------- #
# Algorithm registry -- identical to ml/run_baseline.py.
# --------------------------------------------------------------------------- #

def build_algorithm(name):
    if name == "elasticnet":
        est = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, random_state=SEED)
        grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__l1_ratio": [0.2, 0.5, 0.8]}
    elif name == "linear_svm":
        est = LinearSVC(dual=False, max_iter=10000, random_state=SEED)
        grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    elif name == "rbf_svm":
        est = SVC(kernel="rbf", random_state=SEED)
        grid = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}
    elif name == "random_forest":
        est = RandomForestClassifier(random_state=SEED)
        grid = {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 5, 10]}
    elif name == "gradient_boosting":
        est = XGBClassifier(eval_metric="logloss", random_state=SEED)
        grid = {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 5], "clf__learning_rate": [0.01, 0.1]}
    elif name == "knn":
        est = KNeighborsClassifier()
        grid = {"clf__n_neighbors": [3, 5, 9, 15]}
    else:
        raise ValueError(f"Unknown algorithm {name}")
    return est, grid


# --------------------------------------------------------------------------- #
# Metrics -- identical to ml/run_baseline.py.
# --------------------------------------------------------------------------- #

def sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn + 1e-6)


def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp + 1e-6)


def score_fold(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
    }


# --------------------------------------------------------------------------- #
# PyTorch MLP -- ComBat fit once per outer-train fold, mirroring how this
# function's StandardScaler is already handled in ml/run_baseline.py (fit
# once on the outer-train fold, not re-nested per inner fold).
# --------------------------------------------------------------------------- #

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_mlp(X_train, y_train, hidden_dim, weight_decay, device, epochs=150):
    model = MLP(X_train.shape[1], hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    return model


def _mlp_predict_score(model, X, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(Xt).cpu().numpy()
    return logits, (logits > 0).astype(np.int64)


def run_mlp_nested_cv(X, y, covars):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_grid = [64, 128]
    wd_grid = [1e-4, 1e-3, 1e-2]

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_outer, y_train_outer = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        covars_train_outer = covars.iloc[train_idx].reset_index(drop=True)
        covars_test = covars.iloc[test_idx].reset_index(drop=True)

        combat_model, X_train_outer_h = harmonizationLearn(
            X_train_outer.astype(np.float64), covars_train_outer, seed=SEED,
        )
        X_test_h = harmonization_apply_safe(X_test.astype(np.float64), covars_test, combat_model)

        scaler = StandardScaler().fit(X_train_outer_h)
        X_train_outer_s = scaler.transform(X_train_outer_h)
        X_test_s = scaler.transform(X_test_h)

        # inner CV: pick (hidden_dim, weight_decay) by mean inner-fold accuracy
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        best_acc, best_params = -1, (hidden_grid[0], wd_grid[0])
        for hidden_dim in hidden_grid:
            for wd in wd_grid:
                inner_accs = []
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_s, y_train_outer):
                    model = _fit_mlp(X_train_outer_s[inner_train_idx], y_train_outer[inner_train_idx],
                                      hidden_dim, wd, device)
                    _, pred = _mlp_predict_score(model, X_train_outer_s[inner_val_idx], device)
                    inner_accs.append(accuracy_score(y_train_outer[inner_val_idx], pred))
                mean_acc = float(np.mean(inner_accs))
                if mean_acc > best_acc:
                    best_acc, best_params = mean_acc, (hidden_dim, wd)

        # refit on the FULL outer-train set using the chosen hyperparameters
        final_model = _fit_mlp(X_train_outer_s, y_train_outer, best_params[0], best_params[1], device)
        y_score, y_pred = _mlp_predict_score(final_model, X_test_s, device)
        fold_scores.append(score_fold(y_test, y_pred, y_score))

    return fold_scores


# --------------------------------------------------------------------------- #
# sklearn/XGBoost nested CV -- ComBat + StandardScaler + estimator, all
# inside one Pipeline that GridSearchCV cross-validates (inner loop) and the
# outer StratifiedKFold below wraps (outer loop). Both steps get refit fresh
# per inner fold, same leak-free pattern as ml/run_baseline.py's scaler.
# --------------------------------------------------------------------------- #

def run_sklearn_nested_cv(X, y, covars, algorithm):
    est, grid = build_algorithm(algorithm)
    n_features = X.shape[1]
    X_packed = _pack_with_covars(X, covars)

    pipeline = Pipeline([
        ("combat", ComBatHarmonizer(n_features=n_features)),
        ("scaler", StandardScaler()),
        ("clf", est),
    ])
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_scores = []
    for train_idx, test_idx in outer_cv.split(X_packed, y):
        X_train, y_train = X_packed[train_idx], y[train_idx]
        X_test, y_test = X_packed[test_idx], y[test_idx]

        search = GridSearchCV(pipeline, grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)  # refits best params (incl. ComBat) on the full outer-train set

        y_pred = search.predict(X_test)
        if hasattr(search, "decision_function"):
            y_score = search.decision_function(X_test)
        else:
            y_score = search.predict_proba(X_test)[:, 1]
        fold_scores.append(score_fold(y_test, y_pred, y_score))

    return fold_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True,
                        choices=["elasticnet", "linear_svm", "rbf_svm", "random_forest",
                                 "gradient_boosting", "knn", "mlp"])
    parser.add_argument("--out-dir", default="/users/3171356m/muhammad/GraSTIACL/ml_combat_gPCC/results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(SEED)

    print("Loading feature set: global_pcc (+ SITE/AGE/SEX covariates)")
    X, y, covars = load_features_and_covars()
    print(f"X shape: {X.shape}, y counts: {np.bincount(y)}, sites: {covars['SITE'].nunique()}")

    if args.algorithm == "mlp":
        fold_scores = run_mlp_nested_cv(X, y, covars)
    else:
        fold_scores = run_sklearn_nested_cv(X, y, covars, args.algorithm)

    metrics = list(fold_scores[0].keys())
    summary = {m: {"mean": float(np.mean([f[m] for f in fold_scores])),
                    "std": float(np.std([f[m] for f in fold_scores]))} for m in metrics}

    result = {
        "algorithm": args.algorithm,
        "feature_set": "global_pcc_combat",
        "n_subjects": int(len(y)),
        "fold_scores": fold_scores,
        "summary": summary,
    }
    out_path = osp.join(args.out_dir, f"{args.algorithm}__global_pcc_combat.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
