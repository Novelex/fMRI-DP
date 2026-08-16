# LAYER 1 VERIFICATION — `alff_new/` (imported artifact)

Date: 2026-08-16 · HEAD at audit: `cade2cc` · Interpreter: `/users/3171356m/miniconda3/envs/grastiacl/bin/python3` (Python 3.12.13, torch 2.5.0+cu121) · Diagnostic only, no training, no other file changed.

## Section 1 — Provenance of the input · **VERDICT: BLOCKED (cannot verify locally)**

`alff_new/` contains only two npz files (`non_combat/alff_new.npz`, `combat/alff_new_combat.npz`).
Searched the entire `/users/3171356m/muhammad` tree: **no computation script, no `subject_tr.csv`,
no `rois_aal` `.1D` time series exist here.** The artifact was imported from another repository
(consistent with the user's own recollection). The npz's internal timestamps are zeroed
(numpy default), so no build date is recoverable from the file.

- The external "ALFF Phase 1" log (documented separately) states the source was
  `cpac / nofilt_noglobal / rois_aal.1D` — **stated, not provable here.**
- The empirical filtered-vs-unfiltered spectrum test **cannot be run**: it requires the source
  time series, which are absent.
- **To unblock:** copy from the source repository: `scripts/compute_alff.py`,
  `data/subject_tr.csv`, and at least a few `data/ALFF_need/rois_aal/*.1D` files.

## Section 2 — The computation, line by line · **VERDICT: BLOCKED (script absent)**

Cannot quote lines from a script that is not in this repository. The external log documents:
detrend along time axis; FFT zero-padded to 2^nextpow2; per-subject TR from `subject_tr.csv`;
amplitude `2*|FFT|/n`; bands slow-5 (0.010–0.027), slow-4 (0.027–0.073), classical (0.010–0.080);
mean amplitude across bins; `malff = alff / alff.mean(axis=0)` per band. **All of that is
testimony, not verification.** What the values themselves *prove* is in Sections 3–4 below.

Paper cross-check on the *documented* recipe: bands match A-GCL §2.1 verbatim; GraSTI-ACL §4.1
names the same three bands (d=3) without specifying the computation. No divergence *in the
documented recipe* — but the recipe itself is unverified here.

## Section 3 — Shape, identity, alignment · **VERDICT: PASS**

```
alff shape: (956, 90, 3)   malff shape: (956, 90, 3)
unique IDs: 956            ok flags: all True
IDs vs raw-dir cohort (source of data.pt / data_alff_pcc93.pt / data_alff_raw.pt):
    missing = 0, extra = 0
dx_group vs ASD/NC directory membership mismatches: 0
```
ROI selection method and column order cannot be read from a script (absent), but the
band-nesting correlation pattern in Section 4 independently confirms the documented column
order (slow-5, slow-4, classical): the two *nested* pairs correlate higher than the
*disjoint* pair, exactly as the physics requires.

## Section 4 — Distribution sanity · **VERDICT: PASS**

```
non-finite entries: alff 0, malff 0
malff per-subject-band mean across 90 ROIs: min=1.000000 mean=1.000000 max=1.000000
exact zeros: 0
entries >5 SD below cohort median (per ROI/band): 0
band correlations (mean within-subject across 90 ROIs):
    slow5–slow4  (disjoint bands) = 0.8516   <- the critical pair: NOT >0.98, masks took
    slow5–classical (nested)      = 0.9269
    slow4–classical (nested)      = 0.9834   <- expected: classical CONTAINS slow-4
```
The 0.9834 is not a mask failure: classical (0.01–0.08) mathematically contains slow-4
(0.027–0.073), so near-identity is expected for that pair and only that pair. The disjoint
pair at 0.85 plus the nesting pattern is the signature of correctly applied masks.

## Section 5 — Old vs new · **VERDICT: MEANINGFULLY DIFFERENT (not equivalent)**

Per-ROI-per-band correlation between old and new ALFF **across subjects** (the ordering that
classification actually uses), 270 values:

```
min=-0.003   p25=0.276   median=0.393   p75=0.548   max=0.805
most-disagreeing ROI indices (0-89, AAL order): 0, 32, 29, 66, 60  (min-r ≈ 0.00–0.05)
```

Nowhere near r>0.95: the two ALFF computations order subjects very differently per region.
Note: an earlier within-subject *spatial* comparison gave mean r≈0.74 — both are true; they
measure different things. Between-subject ordering (this section) is the classification-relevant
one, and it disagrees substantially.

## Section 6 — Leakage audit on normalisation · **VERDICT: PASS for what is in use, one FLAG**

- `alff` (raw): no normalisation at all → **SAFE**.
- `malff`: divided by the subject's **own** whole-brain mean, per band — proven by the exact
  1.000000 invariant in Section 4 → within-subject → **SAFE**.
- `data_alff_raw.pt` (campaign arm F / T1d): built from the raw array only, no cohort statistics
  → **SAFE**.
- **FLAG:** `alff_new_combat.npz` (keys: file_ids, alff, dx_group, ok, site) is cross-subject
  site harmonisation *by construction*; whether it was fold-fitted is unknowable here (script
  absent). **Do not use the combat variant in any honest evaluation** until its fitting scope is
  verified in the source repository. It is not currently used anywhere in this repo.

## Section 7 — Decisive classical head-to-head · **VERDICT: PASS (control clean); old ≈ new**

LinearSVC + C-grid {0.001…1000}, GridSearchCV(inner StratifiedKFold 5, seed 123), outer
StratifiedKFold(5, shuffle, seed 123) — same protocol family as the recorded classical baseline.

```
OLD ALFF (z-scored, current)         acc = 0.5889 ± 0.0248   auc = 0.6015 ± 0.0273
NEW ALFF (raw, imported)             acc = 0.5900 ± 0.0194   auc = 0.6113 ± 0.0230
NEW ALFF, labels SHUFFLED (control)  acc = 0.4707 ± 0.0200   auc = 0.4528 ± 0.0252
```

- Shuffle control collapses below chance → harness leak-free → the two real numbers stand.
- Old vs new: statistically identical accuracy (+0.1pp), new +1.0pp AUC — within noise.
- **Correction to an earlier reading:** a previous quick comparison (default-C logistic
  regression) had scored raw-new at 52.9% and concluded raw was worse classically. Under the
  proper C-grid LinearSVC protocol, raw-new reaches 59.0% — equal to old. That earlier
  conclusion was classifier-dependent and is retracted; this section supersedes it.

## Overall verdict

| Section | Verdict |
|---|---|
| 1 Provenance | **BLOCKED** — imported artifact; needs source repo files to verify |
| 2 Computation | **BLOCKED** — script absent; documented recipe matches papers, unverified |
| 3 Identity/alignment | PASS |
| 4 Distribution | PASS |
| 5 Old vs new | MEANINGFULLY DIFFERENT (median r=0.39 between-subject) |
| 6 Leakage | PASS for everything in use; combat variant flagged, unused |
| 7 Head-to-head | PASS — old ≈ new (59% / AUC 0.60–0.61), control clean |

**Bottom line:** everything checkable about `alff_new` locally is clean, internally consistent,
ID-aligned to the cohort, and leak-free as used. What cannot be certified from this repository
is *where the numbers came from* (filter status of the source time series, per-subject TR,
exact FFT recipe). Those require three files from the source repository, listed in Section 1.
Until then, `alff_new` is safe to *use* but must be labeled "computation externally sourced,
provenance unverified" in any write-up.
