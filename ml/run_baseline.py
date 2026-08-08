"""Classical ML baseline: one (algorithm, feature-set) combination per run.

Nested 5-fold stratified CV throughout: an outer split gives the honest,
unbiased performance estimate; an inner split (via GridSearchCV, or a manual
inner loop for the PyTorch MLP) tunes hyperparameters using only that outer
fold's training data. No encoder/upstream unsupervised stage exists here, so
unlike the deep GraSTI-ACL pipeline, nesting covers the entire pipeline
end-to-end -- there is no earlier stage left unsplit.

Usage:
    python run_baseline.py --algorithm elasticnet --feature-set global_pcc
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

DATA_ROOT = "/users/3171356m/muhammad/GraSTIACL/data/GraSTIACL_ABIDE_979/raw"
SEED = 123
N_ROI = 90


# --------------------------------------------------------------------------- #
# Data loading -- same 956-subject cohort, same file-matching-by-ID as
# Dataset.py (never by os.listdir() position -- see Issue "file-matching bug"
# in ISSUES.md for why that matters).
# --------------------------------------------------------------------------- #

def _subject_ids(folder, suffix):
    return {f.rsplit(suffix, 1)[0]: f for f in os.listdir(folder)}


def _upper_tri(mat):
    iu = np.triu_indices(N_ROI, k=1)
    return mat[iu]


def _load_group_features(adj_dir, nf_dir, dw_dir, y, feature_set):
    adj_files = _subject_ids(adj_dir, "_adj.mat")
    nf_files = _subject_ids(nf_dir, "_nf.mat")
    dw_files = _subject_ids(dw_dir, "_dw.mat")
    subject_ids = sorted(set(adj_files) & set(nf_files) & set(dw_files))
    assert subject_ids == sorted(set(adj_files)) == sorted(set(nf_files)) == sorted(set(dw_files)), (
        f"Subject sets differ across {adj_dir}, {nf_dir}, {dw_dir}"
    )

    rows = []
    for sid in subject_ids:
        if feature_set == "global_pcc":
            mat = sio.loadmat(osp.join(adj_dir, adj_files[sid]))
            pcc = np.nan_to_num(mat["cropped_matrix"])
            rows.append(_upper_tri(pcc))
        elif feature_set == "local_pcc":
            mat = sio.loadmat(osp.join(dw_dir, dw_files[sid]))
            dw_array = mat["correlation_matrices"]
            windows = np.stack([np.nan_to_num(dw_array[j, 0]) for j in range(dw_array.shape[0])])
            mean_conn = _upper_tri(windows.mean(axis=0))
            std_conn = _upper_tri(windows.std(axis=0))
            rows.append(np.concatenate([mean_conn, std_conn]))
        elif feature_set == "alff":
            mat = sio.loadmat(osp.join(nf_dir, nf_files[sid]))
            nf = np.nan_to_num(mat["norm_matrix"])
            rows.append(nf.flatten())
        else:
            raise ValueError(f"Unknown feature_set {feature_set}")
    X = np.stack(rows)
    y_arr = np.full(len(subject_ids), y, dtype=np.int64)
    return X, y_arr


def load_features(feature_set):
    X_asd, y_asd = _load_group_features(
        osp.join(DATA_ROOT, "ASD_ADJ"), osp.join(DATA_ROOT, "ASD_NF"), osp.join(DATA_ROOT, "ASD_DW"),
        y=1, feature_set=feature_set,
    )
    X_nc, y_nc = _load_group_features(
        osp.join(DATA_ROOT, "NC_ADJ"), osp.join(DATA_ROOT, "NC_NF"), osp.join(DATA_ROOT, "NC_DW"),
        y=0, feature_set=feature_set,
    )
    X = np.concatenate([X_asd, X_nc], axis=0)
    y = np.concatenate([y_asd, y_nc], axis=0)
    return X, y


# --------------------------------------------------------------------------- #
# Algorithm registry -- each returns (estimator, param_grid) for use inside
# GridSearchCV, except "mlp" which is handled separately (PyTorch, GPU).
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
# Metrics -- same set/definitions as unsupervised/embedding_evaluation.py, for
# direct comparability with the deep pipeline's own reported numbers.
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
# PyTorch MLP -- the one algorithm that genuinely uses the GPU. Manual nested
# CV (outer StratifiedKFold; inner StratifiedKFold for hyperparameter search
# over hidden width / weight decay), since a raw PyTorch model doesn't drop
# into GridSearchCV directly.
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


def run_mlp_nested_cv(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_grid = [64, 128]
    wd_grid = [1e-4, 1e-3, 1e-2]

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_outer, y_train_outer = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train_outer)
        X_train_outer_s = scaler.transform(X_train_outer)
        X_test_s = scaler.transform(X_test)

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
# sklearn/XGBoost nested CV -- GridSearchCV IS the inner loop; the outer loop
# is the StratifiedKFold below. Scaler lives inside the pipeline GridSearchCV
# cross-validates, matching the leakage fix already applied to the main
# pipeline's own SVM evaluation (Issue #23).
# --------------------------------------------------------------------------- #

def run_sklearn_nested_cv(X, y, algorithm):
    est, grid = build_algorithm(algorithm)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        search = GridSearchCV(pipeline, grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)  # refits best params on the full outer-train set automatically

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
    parser.add_argument("--feature-set", required=True,
                        choices=["global_pcc", "local_pcc", "alff"])
    parser.add_argument("--out-dir", default="/users/3171356m/muhammad/GraSTIACL/ml/results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(SEED)

    print(f"Loading feature set: {args.feature_set}")
    X, y = load_features(args.feature_set)
    print(f"X shape: {X.shape}, y counts: {np.bincount(y)}")

    if args.algorithm == "mlp":
        fold_scores = run_mlp_nested_cv(X, y)
    else:
        fold_scores = run_sklearn_nested_cv(X, y, args.algorithm)

    metrics = list(fold_scores[0].keys())
    summary = {m: {"mean": float(np.mean([f[m] for f in fold_scores])),
                    "std": float(np.std([f[m] for f in fold_scores]))} for m in metrics}

    result = {
        "algorithm": args.algorithm,
        "feature_set": args.feature_set,
        "n_subjects": int(len(y)),
        "fold_scores": fold_scores,
        "summary": summary,
    }
    out_path = osp.join(args.out_dir, f"{args.algorithm}__{args.feature_set}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
