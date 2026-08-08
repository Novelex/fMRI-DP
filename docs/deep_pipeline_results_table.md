# Deep Pipeline (GraSTI-ACL) Results — 18 Configs Across 3 Groups

Each config: emb_dim in {32,128,256} x batch_size in {32,128} = 6 configs per group, 18 total. Downstream classifier: LinearSVC (as in GraSTI-ACL).

## Group 1 — No Nested CV, No ComBat (transductive encoder, single run per config)

Implementation: `GraSTIACL.py` (training script), `scripts/run_training_emb{32,128,256}[_bs128].sh` (SLURM wrappers, 6 total).

| emb_dim | batch_size | Val Acc | Test Acc | Test AUC | Result source file | Script used |
|---|---|---|---|---|---|---|
| 32 | 32 | 0.520 | 0.516 | 0.513 | `logs/grasti-acl-emb32_1556161.err` | `scripts/run_training_emb32.sh` |
| 128 | 32 | 0.522 | 0.517 | 0.512 | `logs/grasti-acl-emb128_1556162.err` | `scripts/run_training_emb128.sh` |
| 256 | 32 | 0.537 | 0.507 | 0.506 | `logs/grasti-acl-emb256_1556163.err` | `scripts/run_training_emb256.sh` |
| 32 | 128 | 0.522 | 0.517 | 0.514 | `logs/grasti-acl-emb32-bs128_1556164.err` | `scripts/run_training_emb32_bs128.sh` |
| 128 | 128 | 0.527 | 0.517 | 0.511 | `logs/grasti-acl-emb128-bs128_1556165.err` | `scripts/run_training_emb128_bs128.sh` |
| 256 | 128 | 0.537 | 0.517 | 0.510 | `logs/grasti-acl-emb256-bs128_1556166.err` | `scripts/run_training_emb256_bs128.sh` |

## Group 2 — Nested CV, No ComBat (5-fold nested CV, encoder retrained per fold, mean across folds)

Implementation: `nested_cv/data.py` (raw loading + ComBat), `nested_cv/run_nested_cv.py` (training script), `nested_cv/run_nocombat_array.sh` (SLURM array, 30 tasks).

| emb_dim | batch_size | Val Acc | Test Acc | Test AUC | Result source files |
|---|---|---|---|---|---|
| 32 | 32 | 0.539 | 0.516 | 0.516 | `nested_cv/results/nocombat__emb32_bs32__fold*.json` (5 files) |
| 32 | 128 | 0.539 | 0.525 | 0.519 | `nested_cv/results/nocombat__emb32_bs128__fold*.json` (5 files) |
| 128 | 32 | 0.542 | 0.518 | 0.516 | `nested_cv/results/nocombat__emb128_bs32__fold*.json` (5 files) |
| 128 | 128 | 0.554 | 0.523 | 0.515 | `nested_cv/results/nocombat__emb128_bs128__fold*.json` (5 files) |
| 256 | 32 | 0.544 | 0.511 | 0.512 | `nested_cv/results/nocombat__emb256_bs32__fold*.json` (5 files) |
| 256 | 128 | 0.553 | 0.513 | 0.509 | `nested_cv/results/nocombat__emb256_bs128__fold*.json` (5 files) |

## Group 3 — Nested CV + ComBat (5-fold nested CV, encoder retrained per fold, mean across folds)

Implementation: `nested_cv/data.py` (raw loading + ComBat), `nested_cv/run_nested_cv.py` (training script), `nested_cv/run_combat_array.sh` (SLURM array, 30 tasks).

| emb_dim | batch_size | Val Acc | Test Acc | Test AUC | Result source files |
|---|---|---|---|---|---|
| 32 | 32 | 0.532 | 0.531 | 0.514 | `nested_cv/results/combat__emb32_bs32__fold*.json` (5 files) |
| 32 | 128 | 0.529 | 0.529 | 0.510 | `nested_cv/results/combat__emb32_bs128__fold*.json` (5 files) |
| 128 | 32 | 0.528 | 0.525 | 0.510 | `nested_cv/results/combat__emb128_bs32__fold*.json` (5 files) |
| 128 | 128 | 0.525 | 0.533 | 0.513 | `nested_cv/results/combat__emb128_bs128__fold*.json` (5 files) |
| 256 | 32 | 0.533 | 0.532 | 0.515 | `nested_cv/results/combat__emb256_bs32__fold*.json` (5 files) |
| 256 | 128 | 0.531 | 0.532 | 0.514 | `nested_cv/results/combat__emb256_bs128__fold*.json` (5 files) |

