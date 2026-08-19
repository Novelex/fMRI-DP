# Stage 8E arm results (metrics only — checkpoints are excluded by stage8e/.gitignore)

Each directory holds the arm's `metrics.json` (per-epoch, four measurement surfaces),
`config.json` (full hyperparameters + git commit + sha256 of `training.py`,
`Dataset.py` and `PREREGISTERED_CRITERIA.md`), `status.json`, `train.log`,
`environment.txt`, `git_commit.txt` and `command.txt`.

Score them with `python3 stage8e/score_arms.py` (six governing criteria) and
`python3 stage8e/svc_probe.py` (locked secondary LinearSVC probe; needs checkpoints).
