"""Combine the 5 per-fold JSON results for each full config (combat, emb_dim,
batch_size, node_feature_mode, gamma_mode, mij_source, contrastive_mode) into
a mean/std summary, same aggregation style as
ml_combat_combinations/run_baseline_combat.py's fold_scores/summary.

The grouping key originally covered only (combat, emb_dim, batch_size) --
the same gap that let the output filename collide (see run_nested_cv.py's
out_path comment). Fixed together: if two different mij_source/gamma_mode/
contrastive_mode runs ever land in the results/ directory at once, this now
keeps them in separate summary rows instead of silently averaging them as
if they were 5 folds of one experiment.

Usage: python -m nested_cv.aggregate_results
"""
import glob
import json
import os.path as osp

import numpy as np

RESULTS_DIR = osp.join(osp.dirname(__file__), "results")
METRICS = [
    "val_accuracy", "test_accuracy", "test_f1", "test_sensitivity", "test_specificity",
    "test_precision", "test_auc",
]
CONFIG_KEYS = ["combat", "emb_dim", "batch_size", "node_feature_mode", "gamma_mode", "mij_source", "contrastive_mode"]


def main():
    files = sorted(glob.glob(osp.join(RESULTS_DIR, "*__fold*.json")))
    configs = {}
    for path in files:
        with open(path) as f:
            r = json.load(f)
        # Older result files (written before this provenance fix) won't have
        # node_feature_mode/gamma_mode/mij_source keys at all -- fall back to
        # "unknown" rather than KeyError, so pre-fix files still aggregate
        # (each under its own honestly-labeled "unknown" bucket) instead of
        # crashing this script.
        key = tuple(r.get(k, "unknown") for k in CONFIG_KEYS)
        configs.setdefault(key, []).append(r)

    summaries = []
    for key, folds in sorted(configs.items(), key=lambda kv: str(kv[0])):
        combat, emb_dim, batch_size, node_feature_mode, gamma_mode, mij_source, contrastive_mode = key
        if len(folds) != 5:
            print(f"WARNING: {dict(zip(CONFIG_KEYS, key))} "
                  f"has {len(folds)}/5 folds -- skipping until all 5 complete")
            continue
        summary = {
            "combat": combat, "emb_dim": emb_dim, "batch_size": batch_size,
            "node_feature_mode": node_feature_mode, "gamma_mode": gamma_mode,
            "mij_source": mij_source, "contrastive_mode": contrastive_mode, "n_folds": 5,
        }
        for m in METRICS:
            vals = [f[m] for f in folds]
            summary[m + "_mean"] = float(np.mean(vals))
            summary[m + "_std"] = float(np.std(vals))
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    out_path = osp.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {out_path} ({len(summaries)} configs summarized)")


if __name__ == "__main__":
    main()
