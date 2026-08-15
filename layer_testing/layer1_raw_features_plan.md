# Layer 1 — Raw Features: Test Plan

Scope: ALFF source, subject-ID matching, NaN handling, label convention, ROI ordering.
`node_feature_mode='alff'` only (raw 90x3, no `alff_pcc`). Output of this layer is a
standalone, verified new-ALFF array ready to hand to Layer 2 — nothing in `datasets/Dataset.py`
or `nested_cv/data.py` gets touched yet.

## Already verified (do not re-derive, just re-run as a regression check)

| Check | Result | Evidence |
|---|---|---|
| Subject-ID coverage | 956/956 exact match, 0 missing/extra either direction | set-intersection of `alff_new`'s `file_ids` vs `ASD_ADJ`+`NC_ADJ` filenames |
| ALFF formula provenance | Exact match (2.7e-15, float noise) | recomputed `compute_alff()` from raw `.1D` for `CMU_a_0050642`, diffed against stored `alff_new.npz` |
| `malff` invariant | `malff.mean(axis=1) == 1.0` per band, all subjects | direct check, max deviation 1.3e-15 |
| NaN / `ok` flags | 0 NaNs in `alff`/`malff`, `ok` all `True` | direct check |
| Combat sanity | combat `alff` differs from non-combat (max diff 19.67, not a no-op); `file_ids` order identical across both files | direct check |
| Old `norm_matrix` scale | Confirmed z-scored per subject per band (mean~0, std=1 exactly) on a real `.mat` file | direct check on `Caltech_0051456_nf.mat` |
| Band order match | Old: `["slow5","slow4","classical"]` (notebook `03_alff_node_features.ipynb`, Step 8 printout). New: `BANDS=[(0.01,0.027),(0.027,0.073),(0.01,0.08)]` = slow5, slow4, classical. | identical order, column-for-column |
| Label convention | Old routes `DX_GROUP==1`→`ASD_NF/`(y=1), `DX_GROUP==2`→`NC_NF/`(y=0) (notebook Step 8). `alff_new`'s `dx_group` uses the same raw ABIDE values (bincount: 455 at index 1, 501 at index 2). | mapping is `y = 1 if dx_group==1 else 0` — same rule the old pipeline already used, not a new decision |
| Paper-faithful normalization | A-GCL §2.1: per-subject min-max to `[0,1]` over the full 90x3 block, on raw `alff` (not `malff`). GraSTI-ACL's own paper specifies nothing, but reuses A-GCL's node-feature paradigm for its baselines. | direct PDF read, both papers |

## Open — what this layer still needs to check or build

1. **ROI ordering equivalence.** The old pipeline's notebook claims its 90-region order (`aal_mask_pad.nii.gz`, labels sorted ascending, <9001) was validated at r=1.0000 against the official `rois_aal` output — which is what `alff_new` was itself computed from. This is strong self-reported evidence, but not yet independently re-checked in this session. Do a direct spot check: take the `.1D` header's label sequence for one subject (already confirmed ascending: `2001,2002,2101,...`), confirm it matches the atlas notebook's `region_labels = sorted(...)` ascending-label construction exactly (same 90 labels, same order) — not just "same count."
   - **Pass criterion:** identical label sequence, not just identical label set.
   - Secondary supporting evidence already in hand: the old-vs-new correlation study (sibling repo) found a *uniform* ~0.65–0.665 mean r across all three bands, not near-zero — a genuine ROI misalignment would be expected to destroy correlation almost entirely rather than produce a stable moderate value, so this is indirectly consistent with matching order. Worth stating in the writeup, not sufficient on its own.

2. **Build the new-ALFF per-subject feature array** (`layer_testing/` script, not wired into the training pipeline):
   - Load `alff_new/non_combat/alff_new.npz`'s raw `alff` (956, 90, 3).
   - Remap `dx_group` → `y` per the rule above; cross-check against which directory (`ASD_ADJ`/`NC_ADJ`) each subject's ID currently lives in under the old pipeline — **956/956 must agree**. This doubles as a second, independent confirmation of the label mapping (catches the case where either source mislabels a subject).
   - Apply per-subject min-max to `[0,1]` over the full 90x3 block (A-GCL's exact formula: `(x - x.min()) / (x.max() - x.min())`, min/max taken jointly over all 270 values for that subject).
   - Sanity: output must be finite, within `[0,1]` inclusive, no subject with `max == min` (would divide by zero — check for this degenerate case explicitly rather than assuming it can't happen).

3. **Old vs new correlation re-check, narrowed to this repo's actual 956 subjects.** The sibling repo already ran this study, but reproducing it here (even as a quick pass, reusing their z-score-based fairness correction) is a cheap wiring-correctness canary: if a subject-alignment bug is accidentally introduced while building the loader above, correlation would collapse or look qualitatively different from the already-published ~0.66 mean / 0.38–0.85 range. Treat "roughly reproduces that range" as a pass signal for the loader's correctness, not as new evidence about which ALFF is "better" (that question is already answered — neither is a strong standalone predictor).

4. **Write findings to `layer_testing/layer1_results.md`** (or similar) once the above runs: pass/fail table matching the format above, plus the actual computed correlation numbers and the label cross-check agreement count.

## Explicitly out of scope for Layer 1 (deferred)

- `alff_pcc` (93-dim) mode — scope is `alff`-only per your instruction.
- Anything touching `datasets/Dataset.py` or `nested_cv/data.py` directly — that wiring is shared groundwork used by Layers 3/4/6/7, tracked separately.
- PCC edge weights, `dyn_weight`, M_ij — Layer 2.
- Any GNN/training code — Layers 3+.

## Deliverables

- `layer_testing/layer1_build_new_alff.py` — builds the verified, min-max-scaled new-ALFF array + remapped labels, saves as an intermediate artifact (e.g. `.npz`) for later layers to consume.
- `layer_testing/layer1_verify.py` — runs checks 1, 2 (label cross-check), 3 (correlation) above and prints a pass/fail summary.
- `layer_testing/layer1_results.md` — the filled-in results table.

Not yet written — this is the plan only, per your ask. Say go-ahead and I'll implement the two scripts.
