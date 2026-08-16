# GraSTIACL — ABIDE-I Reproduction: The Full Project Journey

This repository reproduces *GraSTI-ACL* (He et al., **Medical Image Analysis** 107 (2026) 103815) — Graph Spatial-Temporal Infomax with Adversarial Contrastive Learning for brain-disorder diagnosis from resting-state fMRI. The original paper validates on AD/MDD/BD; this repository never had ABIDE data or code to start from — everything here, from the raw data download through every model fix, was built and diagnosed from scratch over the course of this project. This document is the complete, in-order narrative of that work: how the data was obtained and validated, what each preprocessing notebook does and why, how the model code was audited against the paper's own equations, and every bug that was found and fixed along the way.

For the original paper's figure/citation, see the bottom of this file. For a compact per-issue technical reference (file, exact diff, paper equation, verification test), see **[ISSUES.md](ISSUES.md)** — this README tells the *story*; ISSUES.md is the *ledger*.

<p align="center">
<img src="main_fig.jpg" alt="main_fig" width="600"/>
  <p align="center"><b>Main figure of GraSTIACL.</b></p>
</p>

---

## Table of contents

1. [Setting up the workspace](#1-setting-up-the-workspace)
2. [Acquiring the ABIDE-I data](#2-acquiring-the-abide-i-data)
3. [Validating the atlas/ROI-averaging methodology](#3-validating-the-atlasroi-averaging-methodology)
4. [Notebook 1 — data inspection and QC](#4-notebook-1--data-inspection-and-qc)
5. [ALFF computation — the pilot-then-scale journey](#5-alff-computation--the-pilot-then-scale-journey)
6. [Notebook 2 — Global and Local PCC](#6-notebook-2--global-and-local-pcc)
7. [Notebook 3 — ALFF node features](#7-notebook-3--alff-node-features)
8. [Finalizing the cohort](#8-finalizing-the-cohort)
9. [Fixing the dataset loader (`Dataset.py`)](#9-fixing-the-dataset-loader-datasetpy)
10. [Setting up a working PyTorch/PyG environment](#10-setting-up-a-working-pytorchpyg-environment)
11. [The paper vs. the code — a full architectural audit](#11-the-paper-vs-the-code--a-full-architectural-audit)
12. [A second environment blocker, found mid-audit](#12-a-second-environment-blocker-found-mid-audit)
13. [ComBat harmonization — planned, not yet implemented](#13-combat-harmonization--planned-not-yet-implemented)
14. [Getting this repository onto GitHub](#14-getting-this-repository-onto-github)
15. [How to run](#how-to-run)
16. [Citation](#citation)

---

## 1. Setting up the workspace

The project started with a deliberately fresh, isolated workspace, because an adjacent (older, separately-maintained) reproduction of the same paper already existed on the same HPC account, at `~/reproduction/GraSTIACL`, with its own ~1TB populated `data/` directory. The very first mistake made in this project was reusing that old, populated data folder for the new work — it was the wrong data (different preprocessing, different scope), and it was corrected immediately once flagged: a brand-new, empty directory was created at `~/sharedscratch/muhammad-GraSTIACL/data`, and `data/` in this repository was symlinked to it, per standard HPC etiquette (never store large data in the home directory).

Two required MATLAB toolboxes — **SPM12** and **DPABI** (which bundles DPARSFA, used later for ALFF) — needed to be installed somewhere accessible. The first attempt placed them as a sibling of `data/`; this was also corrected once flagged, moving them to nest *inside* the shared-scratch-backed data directory instead (`data/software/`), keeping large software installs off the home filesystem alongside the data itself.

## 2. Acquiring the ABIDE-I data

`download/downloadData.py` and `download/download_func_mask.py` pull three specific ABIDE-I C-PAC preprocessed derivatives — deliberately *only* these three, not the broader default set that public sample download scripts typically include:

- **`rois_aal`, pipeline variant `filt_noglobal`** — AAL-atlas-averaged ROI time series. This is the input to both static and dynamic PCC.
- **`func_preproc`, pipeline variant `nofilt_noglobal`** — preprocessed 4D functional volumes, unfiltered. This is the input to ALFF (which needs the raw frequency spectrum before band-pass filtering, since ALFF itself computes power in specific frequency bands).
- **`func_mask`, pipeline variant `nofilt_noglobal`** — brain masks matching `func_preproc`, needed to define which voxels belong to each ROI during ALFF region-averaging.

The scope of this download went through more than one correction. Early drafts, working from a general-purpose ABIDE download sample script, pulled in extra derivatives and filter variants the sample happened to include (additional ALFF-adjacent derivatives, `func_mask` under the wrong filter variant, unneeded site filtering) that were never actually requested. Each time, the instruction was the same: implement *exactly* the literal, stated scope — a pasted sample script is style reference only, never license to carry over whatever extra defaults it happens to contain. The final download scripts reflect only the three derivatives above, nothing more.

## 3. Validating the atlas/ROI-averaging methodology

Before trusting *any* ROI-averaged value in this pipeline — PCC or ALFF — the exact averaging methodology needed to be independently verified against ABIDE's own official `rois_aal` derivative, since this project's PCC extraction needed to reproduce that same averaging by hand (to also apply it to ALFF, which the official derivative release doesn't provide pre-averaged).

The first attempt got the resampling direction backwards — resampling the AAL atlas into each subject's native functional space. This was rejected, with the correct historical C-PAC procedure specified precisely: **FLIRT, with an identity transformation matrix and trilinear interpolation, resampling the *functional* data into the atlas's (MNI152, 3mm) grid** — the opposite direction. This was implemented via an `apptainer` container running FSL 6.0.7.4 (`vnmd/fsl_6.0.7.4`), and checked against the real `rois_aal` derivative for real subjects.

Result: **Pearson r = 1.0000** between the reproduced ROI-averaged time series and the official derivative (see `public/flirt_correlation.png` and `scripts/validate_aal_averaging_flirt.py`). Only after this exact match was the same averaging code trusted to extract ALFF node features later — meaning PCC and ALFF, despite coming from entirely different underlying computations, share one validated, node-index-consistent atlas methodology.

## 4. Notebook 1 — data inspection and QC

**`notebooks/01_data_inspection.ipynb`** performs the cohort's quality-control filtering, using two criteria only:

- **Framewise displacement (FD) ≤ 0.5 mm** — standard motion-scrubbing threshold.
- **Scan duration ≥ 288 seconds** — chosen specifically because the dynamic-PCC design in this reproduction uses three non-overlapping 96-second windows (3 × 96 = 288s), a deliberate deviation from the paper's own 40-TR-segment windowing (see Notebook 2 below); any subject too short to fill all three windows is excluded.

A deliberate, explicit design decision: this notebook does **not** use ABIDE's own QC rater columns at all (the manual quality-rating fields some ABIDE releases include) — filtering is done purely on the two objective, reproducible criteria above.

## 5. ALFF computation — the pilot-then-scale journey

ALFF (Amplitude of Low-Frequency Fluctuation) is computed via MATLAB's DPARSFA (part of DPABI), across three canonical frequency bands: **classical** (0.01–0.08 Hz), **slow-4** (0.027–0.073 Hz), **slow-5** (0.01–0.027 Hz) — run on `func_preproc` (nofilt_noglobal), driven via `matlab -batch` under `module load matlab/r2025a`.

Given how expensive and easy-to-silently-misconfigure a full-cohort MATLAB batch job is, this was validated in an unusually rigorous, explicitly staged sequence before ever running at scale:

1. **Spectrum sanity check** — confirming the raw BOLD signal's frequency content actually spans the bands being extracted.
2. **Single-subject pilot** (`scripts/matlab/pilot_classical.m`, `pilot_slow4.m`, `pilot_slow5.m`) — one subject, one band at a time, inspecting output by hand.
3. **6-subject, multi-TR pilot** (`scripts/matlab/inspect_tr.m`, `inspect_tr2.m`) — since ABIDE-I spans sites with different repetition times (TRs), this step confirmed DPARSFA handles the TR variation correctly across a small, deliberately TR-diverse subject sample before scaling further.
4. **Full-cohort run** — only after all pilots passed. A naive single SLURM job across the whole 979-subject cohort stalled outright; the workaround was chunking the job into **10 chunks × 3 bands = 30 parallel SLURM array jobs** (`scripts/run_dparsf_alff_chunk.sh`, `scripts/matlab/run_band.m`).

**A bug found and fixed mid-pipeline:** an added "voxel coverage" QC check (meant to flag ROIs with poor brain-mask coverage) accidentally *excluded zero-valued voxels from the ALFF region-averaging mean* — i.e., a legitimate ALFF value of exactly zero at a voxel was being treated as "missing" and dropped from the average, rather than averaged in normally. This produced a false-positive coverage-QC flag on all 956/956 subjects. The fix reverted the mean calculation to average all region voxels unconditionally, keeping the zero-voxel count only as a separate, non-exclusionary diagnostic statistic.

## 6. Notebook 2 — Global and Local PCC

**`notebooks/02_local_global_pcc.ipynb`** computes, from `rois_aal` (filt_noglobal):

- **Global/Static PCC** — one 90×90 Pearson correlation matrix per subject, over the subject's full scan.
- **Local/Dynamic PCC** — three separate 90×90 Pearson correlation matrices per subject, one per non-overlapping 96-second window.

The three-window, 96-second design is a **deliberate, explicitly confirmed deviation** from the paper's own methodology, which instead divides the BOLD signal into 40-TR segments (Section 3.1 of the paper). This project chose fixed-duration (time-based) windows instead of fixed-TR windows specifically because ABIDE-I's 19 acquisition sites use different TRs — a fixed-TR window would correspond to a *different* real-world duration at every site, whereas a fixed-duration window is comparable across all sites regardless of TR. This is why Notebook 1's QC duration threshold (288s) is exactly `3 × 96`.

Output is saved directly in the exact `.mat` layout `datasets/Dataset.py` expects: `cropped_matrix` (90×90) for static PCC, `correlation_matrices` (a 3×1 MATLAB cell array, not a plain 3D array — this distinction matters, since `Dataset.py` indexes it as `dw_array[j, 0]`) for dynamic PCC.

## 7. Notebook 3 — ALFF node features

**`notebooks/03_alff_node_features.ipynb`** extracts each subject's per-ROI ALFF value (across all three bands, 90 ROIs × 3 bands = 270 values per subject) using the *same* FLIRT-validated atlas methodology as Notebook 2's PCC extraction — deliberately not DPABI's own separately-bundled atlas — so that ROI index `i` means the exact same anatomical region in both the node features and the edge weights. Output is saved as `norm_matrix` (90×3), matching `Dataset.py`'s expected key.

## 8. Finalizing the cohort

Global PCC's own NaN check surfaced 23 subjects with dead (all-NaN or degenerate) ROIs — these were dropped entirely (not zero-filled), bringing the final cohort from 979 down to **956 subjects**. The final phenotypic table (`data/raw/phenotypic_filtered_v2.csv`) reflects this final 956-subject cohort: **455 ASD, 501 NC**, across 19 sites, with `SITE_ID`, `DX_GROUP`, `AGE_AT_SCAN`, and `SEX` all available per subject.

## 9. Fixing the dataset loader (`Dataset.py`)

With the raw `.mat` files ready, `datasets/Dataset.py` — inherited from the paper's own released code, which was built around MDD-vs-control folder naming — needed several fixes:

- **Folder renaming**: `MDD_ADJ`/`MDD_NF`/`MDD_DW` → `ASD_ADJ`/`ASD_NF`/`ASD_DW`, since this reproduction classifies ASD vs. NC, not MDD vs. control.
- **Label convention flip**: the original code assigned `y=0` to the patient group and `y=1` to controls (backwards from clinical convention). This was corrected to `ASD_*` → `y=1`, `NC_*` → `y=0` — patient group as the positive class, matching standard sensitivity/specificity convention. (This flip was initially applied only to the folder rename and missed the actual `y` value in the first pass — caught and corrected once the mismatch was spotted.)
- **A file-matching bug**, discovered independently while re-verifying the label flip: `process()` paired files across the `ADJ`/`NF`/`DW` folders by `os.listdir()` **list position**, trusting that three separate directory listings return subjects in the same order. They do not. A direct test against the actual 455-subject ASD folders confirmed **455/455 position mismatches** — every single subject would have had their adjacency matrix paired with a *different* subject's node features and dynamic weights, silently. This was fixed by matching subjects by filename (subject ID) across all three folders instead, via a `_load_group()` helper that asserts the three folders' subject sets are identical before proceeding.

## 10. Setting up a working PyTorch/PyG environment

The repository's original `.venv` runs Python 3.13, for which the required PyTorch Geometric compiled dependencies (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`, pinned to match this project's CUDA/PyTorch version) have **no published wheels at all** — the PyG wheel index only goes up to `cp312`. A new conda environment, **`grastiacl`** (Python 3.12), was created and populated with the full pinned stack (`torch==2.5.0+cu121`, `torch-geometric==2.6.1`, and the compiled auxiliary packages) — `torchaudio`/`torchvision` were deliberately dropped from the install after confirming nothing in the codebase actually imports either.

One persistent environment gotcha worth documenting: this session's shell snapshot bakes in `VIRTUAL_ENV=.../.venv`, which gets re-sourced on every fresh shell invocation, silently overriding `conda activate` even after it appears to succeed. The reliable workaround is invoking the conda environment's Python via its full explicit path (`/users/3171356m/miniconda3/envs/grastiacl/bin/python3`) rather than trusting `conda activate` to have taken effect.

## 11. The paper vs. the code — a full architectural audit

With data and environment both working, the released model code was audited issue-by-issue against the paper's own stated equations — never against assumption, and never carrying a fix over from the older, separate reproduction project without independently re-verifying it against *this* repository's actual code first. Every item below was empirically verified (unit tests constructing the real modules, not just code review) before being considered resolved. Full technical detail for each is in [ISSUES.md](ISSUES.md); the narrative:

**Issue #1 — a dead import chain.** `unsupervised/convs/__init__.py` imported `gine_conv.py`, a file absent from the release, breaking every downstream import outright. Confirmed `GINEConv` was never actually instantiated anywhere in the active forward path — the import was removed rather than fabricating a stub file.

**Issues #2-3 — the topology-attention fusion was dead code.** `TAEncoder.forward` computed both a GCN branch and an attention branch, but the line combining them into the final representation was commented out, so the entire attention branch received zero gradient during training. Fixed per Eq. (19) (`X_Update = X_Topo + λ·X_Atte`, additive — not the convex blend that had been commented out, which wouldn't have matched the paper even if re-enabled as-is), with `λ` sampled from `Beta(γ, 1-γ)` per Eq. (18). Verified: every parameter of the attention module now receives nonzero gradient after a backward pass.

**Issue #4 — a variable named `beta` had no effect.** Once fusion was rebuilt (above), the literal `beta` argument threaded through the code became genuinely unused — but tracing *why* revealed the real story: `GraSTIACL.py`'s own `fin_reg`/`beta` computation was built from `1 - sigmoid(gate_inputs)` (the mean *drop* probability), the mirror image of what Eq. (18) actually defines (`γ` = mean *retained*-edge ratio). Rather than patch this mismatch by guessing at a sign flip — an initial proposal that was explicitly rejected as unfounded — the correct resolution was found by computing `γ` directly from the adjacency tensor itself inside `TAEncoder.forward` (see #2-3), bypassing the incorrectly-derived variable entirely.

**Issue #5 — dynamic PCC was wired through every function signature but never numerically used.** `ToyNet.get_mu_std_logits` accepted a `dyn_weight` argument that was checked for `None` but discarded; only the static PCC informed the VIB. Fixed by concatenating each ROI's static PCC row with its value across all three temporal windows before encoding, per Eq. (13). Verified: perturbing `dyn_weight` alone (holding static PCC fixed) now measurably changes the model's output, where before it had zero effect.

**Issues #9-11 — `ToyNet`'s two-stage dimension bug.** The VIB's `mu`/`std` layers were built as a two-stage pipeline — collapse each ROI to a single scalar (`Linear(hidden_dim, 1)`), then expand a *collected* 90-length vector of all ROIs' scalars up to the 800-dimensional latent space (`Linear(90, 800)`) — but the surrounding loop calls `forward()` once per individual ROI, so the second stage never actually received the 90-length vector it needed, only ever a single leftover scalar. Depending on arithmetic coincidence this either crashed outright or, worse, silently produced a wrong-shaped, scrambled tensor. Fixed by mapping `hidden_dim → latent_dim` directly in one layer, matching an alternative already sitting commented-out in the original source.

**Issues #6-8 — signed PCC and the GCN's edge weight (superseded by Stage 2; see `docs/STAGE2_AW_FINAL.md`).** The authors' released code fed raw signed PCC straight into standard GCN normalization; negative correlations produce negative weighted node degrees and NaN under `pow(-0.5)` — measured on this cohort: 377 NaN degrees across 246/956 subjects. The historical interim fix described here (unweighted `A` for the original view, gate-only weight for the augmented view) has been **replaced** by the certified Stage-2 semantics: the graph is genuinely fully-connected (all N×N pairs, explicit self-loops); the default profile weights the GCN by `|PCC|`; the corrected signed profile (`--signed_edges`) feeds **raw signed PCC** into a signed-safe convolution (degree normalized by `Σ|W|` while message passing retains the signed weight — anticorrelation preserved, no NaN); and the augmented view is universally `original-edge-weight × gate` (Sec. 3.2's "weaken their connections"), never the bare gate. `gamma` is the per-subject mean of the gate. Raw signed PCC additionally remains `ToyNet`'s VIB input in every profile. All of this is certified in `docs/STAGE2_AW_FINAL.md` (25/25 + 16/16 integration tests, main and nested-CV paths proven identical).

**Issue #15 — attention silently mixed different subjects together within a training batch.** PyTorch Geometric batches multiple subjects' graphs by concatenating all their nodes into one flat tensor; `full_attention_conv`'s kernelized attention pooled over *every* node in that flat tensor with no per-subject restriction, so one subject's output depended on which other subjects happened to share its minibatch. Fixed by threading a `batch` argument through the attention stack and restricting every reduction to within each subject's own block (block-diagonal attention). Verified: perturbing one subject's input now leaves every other subject's output in the same batch bit-for-bit unchanged, where before it did not.

**Issue #16 — the main model and the adversarial view-learner had completely independent encoders.** The paper reports (page 6) that the main model and its auxiliary view module share roughly 415k of their ~422k/420k parameters; the released code built two entirely separate `TAEncoder`+`ToyNet` pairs, sharing nothing. Fixed by constructing one shared backbone and passing the same object into both wrappers (each keeping its own small, genuinely-unshared head). This surfaced a real secondary bug: once shared, `nn.Module.eval()`/`.train()` calls on either wrapper recurse into the *same* shared submodules, and the existing training loop's call order would have left the shared backbone stuck in eval mode during both training phases — fixed by re-asserting train mode on the shared backbone immediately after each such call. Verified: true object identity between the two wrappers' encoders, a parameter-overlap ratio matching the paper's reported figure, and a full two-phase forward/backward/optimizer-step test confirming both optimizers genuinely update the same shared weights.

**Issues #12-14 — the attention mechanism didn't implement what the paper says it implements.** The paper's own text (Eq. 16) states plainly: *"we draw inspiration from Graphormer (Ying et al., 2021) and apply the Transformer principle to compute attention scores."* But the actual code had no softmax anywhere — it implemented a linear/kernelized attention scheme (with a suspicious whole-tensor L2 normalization, dividing every node, head, and dimension by one shared scalar) that, on inspection, is a verbatim match for a *different*, uncited paper's own released code (DIFFormer, Wu et al., ICLR 2023). Graphormer's real equation (fetched and read directly from the NeurIPS proceedings PDF) is standard scaled dot-product softmax attention: `A = QKᵀ/√d`, `Attn(H) = softmax(A)·V` — no whole-tensor norm at all. The fix replaces the linear-attention block with this actual equation, deliberately *not* porting Graphormer's other structural-encoding additions (centrality/spatial/edge bias terms), since this graph is already fully connected (making shortest-path-distance encoding degenerate) and the paper's own text frames this branch as deliberately moving *away* from topology, which the parallel GCN branch already handles. Verified: attention weights now genuinely sum to 1.0 across the key axis, the cross-subject leakage test from #15 still holds under the new math, and gradients still flow correctly.

**Issue #17 — an interpretability accumulator that was always zero, and, once fixed, needed a second correction.** `aug_edge_weight_all` (meant to track the model's learned adjacency across training, for the kind of connectivity-strength interpretability the paper shows in its own Fig. 3/4) was initialized and divided every epoch but never actually incremented anywhere — always zero. Fixing the missing accumulation wasn't enough on its own: the paper's own interpretability method (page 9) explicitly compares disease-group vs. healthy-control-group statistics, not one pooled average across everyone — so the fix accumulates two separate running averages, one for ASD subjects and one for NC subjects, each normalized by its own subject count.

**Issue #30 — `drop_last=True` was silently discarding subjects, propping up a separate hidden bug.** With 956 subjects and a batch size of 32, the final partial batch of 28 subjects was dropped every single epoch — roughly 3% of the dataset never trained on in any given epoch. This wasn't an isolated issue: three other lines hardcoded `args.batch_size` as the assumed subject count in a batch (a reg-computation loop and two tensor reshapes), and removing `drop_last` without fixing those first would have caused a partial batch to either crash outright or — demonstrated directly — silently produce a wrong-shaped, scrambled tensor with no error at all, since the numbers can coincidentally divide evenly. Fixed by replacing every hardcoded `args.batch_size` with the batch's actual `num_graphs`, then safely setting `drop_last=False`.

**Issue #29 — reported training loss was scaled by batch count, not subject count.** The loss accumulator was correctly weighted per batch by subject count going in, but divided by `len(dataloader)` (the number of *batches*) at the end — a formula that only happened to be numerically correct back when every batch had exactly the same size (the old `drop_last=True`). Once #30 made batch sizes uneven, this broke outright: simulating the real batch-size distribution showed the old formula inflated the reported loss by a factor of **31.87x** relative to the true per-subject average. Fixed by dividing by the actual total subject count instead.

**Issue #20-21 — cross-validation was unseeded and non-stratified.** Neither the outer 5-fold split nor the inner train/validation split was seeded (`random_state=None`) or stratified by class, so no two runs of the same configuration were comparable, and a given fold could end up with a skewed ASD/NC ratio by chance. Fixed with `StratifiedKFold` and seeded, stratified `train_test_split`, threaded from a single `seed` parameter. Verified against the real 956-subject dataset: identical seed produces bit-for-bit identical folds, different seed produces different folds, and per-fold class ratios stay within 0.002 of the true overall ratio.

**Issue #23 — the feature scaler leaked information across the inner cross-validation loop.** `StandardScaler` was fit on the *entire* training fold before `GridSearchCV`'s own inner folds were even drawn, meaning every inner-validation subject's own values had already influenced the scaling statistics applied to it. Fixed by moving the scaler inside the pipeline that `GridSearchCV` itself cross-validates, so it's refit fresh per inner fold. Verified directly: the old code's scaler mean exactly equaled the *global* training-set mean (proving it was fit before any split happened); the fixed code's per-fold scaler means now genuinely differ from each other.

**Issue #24 — AUC was computed from hard 0/1 predictions.** Since AUC requires a continuous, threshold-sweepable score, and the configured classifiers (`LinearSVC`/`SVC()` without `probability=True`) expose `decision_function()` rather than `predict_proba()`, the fix computes AUC from that continuous decision score instead of the already-thresholded hard predictions used for accuracy/F1/etc. Proven mathematically: the *old* AUC value matched balanced accuracy `(sensitivity+specificity)/2` **exactly**, confirming it was never measuring anything beyond a disguised duplicate of a metric already tracked separately; the new AUC value is measurably different.

**Issue #25-27 — roughly fifteen separate places independently searched for their own "best" epoch, including the test set.** After training, the code ran `np.argmax` on every metric's own curve — train, validation, *and test* — for accuracy, F1, sensitivity, specificity, precision, and AUC, each finding its own independent peak epoch. Two compounding problems: using the test curve to select an epoch at all is a real violation of train/val/test separation, and even setting that aside, combining each metric's own independently-chosen epoch into one "BestTestScore" summary produced numbers that no single trained model checkpoint ever actually achieved simultaneously (a synthetic test confirmed the old approach would report inflated values like 90% accuracy, 85% F1, and 90% sensitivity, cherry-picked from three different epochs, versus honest, internally-consistent values once every metric is read from the single genuine best-validation epoch). A separate copy-paste bug was also found and fixed along the way: the AUC value was being indexed by the *F1* curve's chosen epoch rather than its own.

**Left deliberately unchanged, and why:** `np.nan_to_num()` silently masking any NaN/Inf in the raw feature loading — verified directly that zero NaN/Inf values exist anywhere across all 956 real subjects, and since this dataset is closed (no new subjects will ever be added), the usual justification for hardening this code path doesn't apply here. And the **nested cross-validation gap** — the unsupervised encoder currently trains on the entire 956-subject dataset before the downstream SVM's own 5-fold split exists, meaning every "test" subject has already influenced representation learning. This is the single largest remaining methodological limitation in this reproduction; a correct fix (retraining the encoder from scratch inside each outer fold) would cost roughly 5x the current compute, and the decision was made to acknowledge this limitation explicitly rather than pursue it.

## 12. A second environment blocker, found mid-audit

While trying to actually *run* the CV-related fixes above (rather than just review them), `GraSTIACL.py` turned out to be completely unimportable in the `grastiacl` conda environment: `unsupervised/embedding_evaluation.py` did `from scipy import interp` (removed from top-level SciPy in modern versions) and `import matplotlib.pyplot` (not installed at all in the new environment), and `GraSTIACL.py` had its own separate, previously-unnoticed copy of the same broken `scipy.interp` import, plus entirely dead `matplotlib`/`plotly`/`pandas` imports with zero actual call sites anywhere in the file. Fixed by swapping both `scipy.interp` occurrences to `numpy.interp` (matching how an adjacent reproduction project had already resolved the identical issue), installing `matplotlib` (which *is* genuinely used, by an otherwise-unused visualization helper), and removing the genuinely dead `plotly`/`pandas`/duplicate-`matplotlib` imports outright rather than installing unused dependencies for code that never runs.

## 13. ComBat harmonization — planned, not yet implemented

ABIDE-I spans 19 different acquisition sites, each with its own scanner and protocol — a classic multi-site batch-effect problem. The plan, still being finalized, is to harmonize the raw ALFF/PCC features using **ComBat** (via `neuroHarmonize`'s `harmonizationLearn`/`harmonizationApply` split), fit only on a training subset and applied to held-out subjects — with diagnosis deliberately **excluded** as a protected covariate (only `SITE_ID`, `AGE_AT_SCAN`, `SEX`), to avoid a subtle leakage mode identified in the literature (arXiv:2410.19643) where protecting the prediction target itself as a ComBat covariate requires knowing a test subject's true label just to harmonize their own features. Because this pipeline doesn't currently use nested cross-validation, fitting ComBat separately per evaluation fold without retraining the encoder five times over requires **cross-fitting** (Chernozhukov et al., 2018): fit ComBat on four of five folds, apply to the fifth, repeat across all five, so every subject is harmonized using parameters that never saw their own data, while still producing one single dataset the rest of the existing pipeline can consume unchanged. A closely related published architecture using the identical ALFF+PCC feature construction (Zhang et al., 2023, "A-GCL", Medical Image Analysis 90) was checked directly and does *not* perform any site harmonization at all on its own ABIDE experiments — useful context on how optional this step is, without settling whether it's worth doing here.

## 14. Getting this repository onto GitHub

The working tree was pushed to `github.com/Novelex/fMRI-DP`, after first ensuring the copyrighted paper PDFs (`grastical.pdf`, the Zhang et al. A-GCL PDF) and exported conversation transcripts were excluded via `.gitignore` rather than committed to a public repository.

---

## How to run

```bash
python GraSTIACL.py --path /path/to/data/GraSTIACL_ABIDE_979 --name GraSTIACL_ABIDE_979 --epochs 30
```

Key CLI flags (see `arg_parse()` in `GraSTIACL.py`): `--template` (ROI count, default 90), `--num_dyn_windows` (dynamic PCC windows, default 3), `--epochs`, `--batch_size`, `--eval_interval`, `--downstream_classifier` (`linear`/other), `--seed`, `--model_lr`, `--view_lr`, `--num_gc_layers`, `--pooling_type`, `--emb_dim`, `--vib_hidden_dim`, `--drop_ratio`, `--kld_lambda`, `--ce_lambda`. Device (`cuda`/`cpu`) is selected automatically.

### Requirements

Targets Python 3.12 and CUDA 12.1, via a dedicated conda environment (`grastiacl`):

```
torch==2.5.0+cu121
torch-geometric==2.6.1
torch-scatter==2.1.2+pt25cu121
torch-sparse==0.6.18+pt25cu121
torch-cluster==1.6.3+pt25cu121
torch-spline-conv==1.2.2+pt25cu121
pyg-lib==0.4.0+pt25cu121
```

```bash
pip install torch==2.5.0+cu121 \
  torch-scatter==2.1.2+pt25cu121 torch-sparse==0.6.18+pt25cu121 \
  torch-cluster==1.6.3+pt25cu121 torch-spline-conv==1.2.2+pt25cu121 \
  torch-geometric==2.6.1 pyg-lib==0.4.0+pt25cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

Notebook/preprocessing-side dependencies are in `requirements.txt` (pandas, jupyter, nbconvert, ipykernel).

### Data format and organization

`data/` (symlinked to shared scratch storage, never tracked in git) holds the processed dataset:

```
data/GraSTIACL_ABIDE_979/raw/
├── ASD_ADJ/   ASD_DW/   ASD_NF/
├── NC_ADJ/    NC_DW/    NC_NF/
```

Each subject has the same filename across the `ADJ`/`NF`/`DW` folders for its group (matched by filename, not directory-listing order).

### Project structure

```
├── GraSTIACL.py           # Main training pipeline (entry point)
├── unsupervised/          # Encoder, VIB, contrastive-learning, evaluation components
├── datasets/              # Dataset loader (see Data format above)
├── download/              # ABIDE-I C-PAC derivative download scripts
├── scripts/                # ALFF (DPARSFA/SLURM) and atlas-validation pipeline
├── notebooks/             # QC, PCC extraction, ALFF extraction notebooks
├── ISSUES.md              # Per-issue technical reference (file, diff, paper equation, verification)
└── README.md              # This file
```

---

## Citation

If you use GraSTIACL in your research, please cite
```bibtex
@article{He2025GraSTIACL,
  author = {Biao He and Erni Ji and Xiaofen Zong and Zhen Liang and Gan Huang and Li Zhang},
  title   = {GraSTI-ACL: Graph Spatial-Temporal Infomax with Adversarial Contrastive Learning for Brain Disorders Diagnosis Based on Resting-State fMRI},
  journal = {Medical Image Analysis},
  volume = {107},
  pages = {103815},
  year = {2026},
  issn = {1361-8415},
  doi = {https://doi.org/10.1016/j.media.2025.103815},
}
```

## Contact

For questions about the code, open an issue or contact the repository owner.
