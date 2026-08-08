"""Combine the 5 per-fold JSON results for each (emb_dim, batch_size, combat)
config into a mean/std summary, same aggregation style as
ml_combat_combinations/run_baseline_combat.py's fold_scores/summary.

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


def main():
    files = sorted(glob.glob(osp.join(RESULTS_DIR, "*__fold*.json")))
    configs = {}
    for path in files:
        with open(path) as f:
            r = json.load(f)
        key = (r["combat"], r["emb_dim"], r["batch_size"])
        configs.setdefault(key, []).append(r)

    summaries = []
    for (combat, emb_dim, batch_size), folds in sorted(configs.items()):
        if len(folds) != 5:
            print(f"WARNING: combat={combat} emb_dim={emb_dim} batch_size={batch_size} "
                  f"has {len(folds)}/5 folds -- skipping until all 5 complete")
            continue
        summary = {"combat": combat, "emb_dim": emb_dim, "batch_size": batch_size, "n_folds": 5}
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
