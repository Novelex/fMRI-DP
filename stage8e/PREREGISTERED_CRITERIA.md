# Stage 8E — PRE-REGISTERED RESCUE CRITERIA

Written to disk **before** any Stage-8E arm was launched. Nothing below may be
edited after the first arm starts; a violation invalidates the experiment.

Frozen at: (see stage8e/PREREGISTERED_CRITERIA.sha256 and the git commit that
introduced this file, which precedes every stage8e run directory).

## Measurement protocol (fixed in advance)

* Probe set: the **same fixed 96 subjects** used throughout Stage 8D/8E,
  `IDX = [(k*37) % len(ds) for k in range(96)]`, batch size 32, `alff_new_z`.
* Every metric is reported in **BOTH** forward states:
  * `TRAIN` — model.train(), which is the state the Phase-2 objective is
    actually optimised in.
  * `EVAL`  — model.eval(), which is the state every downstream/linear-probe
    consumer sees.
  Stage 8E established these two differ radically; reporting only one is not
  admissible evidence.
* Epochs probed: 0, 1, 3, 5, 10, 20, 30. Seed 42 for the main sweep.
* Accuracy is **not** a criterion at any point in Stage 8E.

## Null reference (what "no learning" looks like)

For B = 32 with the InfoNCE form used here, a representation carrying no
same-subject identity gives:

| quantity | null value |
|---|---|
| `top1` (positive is arg-max over the row) | 1/32 = 0.03125 |
| `posRank` (mean rank of the positive, 1 = best) | (32+1)/2 = 16.5 |
| `CL` | log(32) = 3.4657 |
| `margin` = pos − mean-neg | 0 |

## Primary rescue criterion (decides RESCUED / NOT RESCUED)

An arm is **RESCUED** only if, at epoch 30, in **EVAL** state:

1. `top1 ≥ 0.25` — at least 8× the 0.03125 chance rate, **and**
2. `posRank ≤ 8.0` — better than half the null rank of 16.5, **and**
3. `CL ≤ 3.20` — at least 0.2657 nats below log(32), **and**
4. `subj_eRank ≥ 4.0` — the subject-embedding matrix has not collapsed.

All four must hold simultaneously. Any subset is **PARTIAL**.
None is **NOT RESCUED**.

## Secondary (descriptive only — never used to declare a rescue)

* `margin` at epoch 30 (EVAL)
* `uniformity` and `alignment` (Wang & Isola 2020)
* `FIRST_COMPRESSION_STAGE` from the 7-stage layer localisation
* TRAIN-vs-EVAL gap in CL (the transfer gap this stage identified)
* gate mean / std / stochastic fraction

## Failure / abort conditions (pre-declared)

* Any non-finite loss at any epoch ⇒ the arm is reported `NON_FINITE`, its
  metrics are excluded, and the abort epoch is recorded.
* An arm that does not reach epoch 30 within the wall-clock budget is reported
  `INCOMPLETE_<last epoch>` — it is **not** silently compared at a shorter horizon.
* If the resource budget prevents an arm from running at all, it is reported as
  `<ARM>_NOT_RUN_RESOURCE_LIMIT`, never omitted.

## Replication rule (pre-declared, so it cannot be chosen after the fact)

* If **exactly one** arm is RESCUED, that arm is re-run at seeds **7** and **2024**.
* If **more than one** arm is RESCUED, the one with the highest epoch-30 EVAL
  `top1` is replicated; ties broken by lower `CL`.
* If **no** arm is RESCUED, **no** replication is run and the verdict is
  `NO_ARM_RESCUES`.
* A rescue is called **REPLICATED** only if **both** extra seeds also satisfy all
  four primary criteria. 2/3 is `PARTIAL_REPLICATION`. 1/3 is `NOT_REPLICATED`.

## Arms (pre-declared; exactly one factor changes per arm)

| arm | changes vs E0 | everything else |
|---|---|---|
| E0 | nothing (production baseline) | frozen |
| E1 | `lambda_pairing_mode=matched` | frozen |
| E2 | `lambda_pairing_mode=attention_off` (λ = 1e-4 clamp floor, **not** literally 0) | frozen |
| E3 | `lambda_pairing_mode=balanced` (λ = 0.5) | frozen |
| E4 | `kld_lambda 0.003 → 0.001` | frozen |
| E5 | `batch_size 32 → 128` | frozen |

No arm changes dropout, GCN depth, `reg_lambda`, learning rate, the ALFF source,
edge signedness, the projection head, or the temperature. E5 changes the batch
size, which changes the InfoNCE null (log 128 = 4.852, chance top1 = 1/128); its
criteria are therefore evaluated against a **B=128 probe** with the null values
recomputed, and it is judged on `top1 ≥ 8× chance`, `posRank ≤ B/4`,
`CL ≤ log(B) − 0.2657`, `subj_eRank ≥ 4.0` — the same four criteria expressed
relative to its own null.

---

## Addendum (written before any arm was launched — see git history)

Arms E1–E3 change **how the positive pair is encoded**, which changes the ruler as
well as the representation. Stage 8E Section 7 already showed that re-pairing λ at
*frozen* weights lifts top-1 from 0.031 to 0.53–0.59. An arm measured only under its
own pairing could therefore look "rescued" without having learned anything new.

Every arm is therefore probed on **four surfaces** each epoch:

| surface | meaning |
|---|---|
| `eval_own`  | eval state, the arm's own training pairing — the **PRIMARY** criterion surface |
| `eval_prod` | eval state, fixed `production` pairing — the **common yardstick** across arms |
| `train_own` | train state, own pairing — the surface the Phase-2 objective is actually optimised on |
| `train_prod`| train state, production pairing |

Pre-declared interpretation rule:

* An arm that passes on `eval_own` **and** improves on `eval_prod` versus E0's
  `eval_prod` has changed the representation → `RESCUED_REPRESENTATION`.
* An arm that passes on `eval_own` but does **not** beat E0 on `eval_prod` has
  mostly changed the measurement → `RESCUED_MEASUREMENT_ONLY`. This is reported
  as a rescue of the *pairing defect*, not of the encoder.
* `posRank` is the scale-free identity measure (null `(B+1)/2`), and is what makes
  E5 (B=128) comparable to the B=32 arms.

The train-state probe snapshots and restores BatchNorm buffers, so probing can never
perturb the run it measures.
