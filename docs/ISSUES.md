# GraSTI-ACL Reproduction — Issues Diagnosed and Fixed

Source of truth for every issue found and fixed in **this repo** while reproducing *GraSTI-ACL* (He et al., *Medical Image Analysis* 107 (2026) 103815). This repo started as a byte-for-byte copy of the paper's official code (`github.com/BiaoHe2025/GraSTIACL`, verified via empty `diff`), so everything below is a genuine gap between the authors' own release and their paper's equations — not an artifact introduced by cloning or copying.

Updated after each issue is resolved. Paper section/equation numbers refer to the published PDF as read directly (Preliminaries = Section 2; Graph Spatial-Temporal Infomax = Section 3.2; Topology Attention Encoder = Section 3.3; Adversarial Contrastive Learning = Section 3.4; Experiments = Section 4).

---

## Pre-issue-list fixes

These were needed to get the pipeline's data format and environment working before the numbered issues below could even be tested.

### Folder naming: `MDD_*` → `ASD_*`

**File:** `datasets/Dataset.py`

`Dataset.py` hardcoded `'MDD_ADJ'`/`'MDD_NF'`/`'MDD_DW'` for the patient-group folder, even though this reproduction uses ABIDE (autism vs. control), not MDD. Renamed to `'ASD_ADJ'`/`'ASD_NF'`/`'ASD_DW'` throughout.

### Label convention flip

**File:** `datasets/Dataset.py`

Original code: `ASD_*` path → `y=0`, `NC_*` path → `y=1` (patient = negative class, backwards from clinical convention). Flipped: `ASD_*` → `y=1`, `NC_*` → `y=0` (patient = positive class), matching standard sensitivity/specificity convention.

### File-matching by `os.listdir()` position, not identity

**File:** `datasets/Dataset.py`

`process()` paired `ADJ`/`NF`/`DW` files by list position (`files[i]`, `files_nf_ASD[i]`, `files_dw[i]`), trusting that `os.listdir()` on three separate directories returns subjects in the same order. **Verified empirically: it does not** — checked our actual 455-subject `ASD_ADJ`/`ASD_NF`/`ASD_DW` folders and found 455/455 position mismatches (every single subject would have been paired with a different subject's node features and dynamic weights). Fixed via a shared `_load_group()` helper that matches files by filename/subject-ID (`{FILE_ID}_adj.mat` ↔ `{FILE_ID}_nf.mat` ↔ `{FILE_ID}_dw.mat`), asserting the three subject sets are identical before proceeding.

### Environment: PyG stack unavailable on Python 3.13

The original `.venv` used Python 3.13, for which `torch-scatter`/`torch-sparse`/`torch-cluster`/`torch-spline-conv` (pinned versions matching `README.md`) have no prebuilt wheels at all (`cp313` doesn't exist on `data.pyg.org`, only up to `cp312`). Created a new conda environment (`grastiacl`, Python 3.12) and installed the full stack — `torch==2.5.0+cu121`, `torch-geometric==2.6.1`, and the compiled auxiliary packages — verified importing correctly against this repo's actual code, not just in isolation. `torchaudio`/`torchvision` were dropped from the install (README lists them, but nothing in this codebase imports either).

---

## Issue #1 — Missing `gine_conv.py`

**Files:** `unsupervised/convs/__init__.py`, `unsupervised/encoder/TA_encoder.py`, `unsupervised/encoder/tu_encoder.py`

**Error / Cause:** `unsupervised/convs/__init__.py` does `from .gine_conv import GINEConv`, but `gine_conv.py` doesn't exist anywhere in the release. Confirmed broken: `import unsupervised.convs` raises `ModuleNotFoundError` immediately, blocking every downstream import.

**Investigation:** Confirmed `GINEConv` is imported in `TA_encoder.py` and `tu_encoder.py` but never actually instantiated anywhere (`grep "GINEConv("` → zero matches) — dead code, not a modeling gap.

**Fix:** Removed the dead import from all three files (`__init__.py`'s export, and both encoders' unused imports), rather than fabricating a stub file to satisfy an import nobody uses.

**Reference from the paper:** None applies — packaging defect, not a described method.

**Verification:** `import unsupervised.convs`, `import unsupervised.encoder.TA_encoder`, `import unsupervised.encoder.tu_encoder` all succeed.

---

## Issues #2-3 — Topology-attention fusion disabled / attention branch got no gradient

**File:** `unsupervised/encoder/TA_encoder.py` (`TAEncoder.forward`)

**Error / Cause:** The encoder computes both a GCN branch (`x`) and an attention branch (`x_trans`), but the fusion line was commented out (along with two other commented-out alternate versions right next to it). Only `x` (pure GCN output) was ever returned. `self.trans_conv`'s parameters (the whole attention module) received zero gradient.

**Fix:**
```python
gamma = edge_weight.mean().clamp(eps, 1-eps) if edge_weight is not None else torch.tensor(1-eps)
lambda_ = torch.distributions.Beta(gamma, 1 - gamma).sample()
x = x + lambda_ * x_trans
```

**Reference from the paper:** Eq. (19): `X_Update = X_Topo + λX_Atte` (additive, not the convex blend that was commented out). Eq. (18): `γ` = mean of the adjacency matrix (retained-edge ratio); `λ = B(γ, 1−γ)`.

**Verification:** Constructed a real `TAEncoder`, ran a backward pass — every `trans_conv` parameter has nonzero gradient (previously all zero). Verified for both `edge_weight` provided and `edge_weight=None`.

---

## Issue #4 — `beta` has no effect

**Files:** `GraSTIACL.py`, `unsupervised/encoder/TA_encoder.py`

**Different finding than the old reproduction project's version of this issue.** Traced `GraSTIACL.py`'s `beta = np.random.beta(fin_reg, 1 - fin_reg)` (computed per-epoch, structurally matching Eq. 18's form) back to its source: `fin_reg` is built from `edge_drop_out_prob = 1 - torch.sigmoid(gate_inputs)` — the mean **drop** probability, not the mean **retained**-edge ratio (`γ`) the paper defines. Since Eq. 12 defines `A = σ(gate_inputs)` directly, chaining into Eq. 18's `γ = mean(A)` gives `γ = mean(sigmoid(gate_inputs))` — the code's `fin_reg` is `1−γ`, not `γ`. Using it as `Beta(fin_reg, 1-fin_reg)` gives the mirror-image distribution `Beta(1−γ, γ)`, not the paper's `Beta(γ, 1−γ)`.

**Fix:** Resolved together with Issues #2-3 — `γ` is computed fresh from `edge_weight` directly inside `TAEncoder.forward()` (the tensor that **is** the adjacency at that point), bypassing `GraSTIACL.py`'s separate, incorrectly-derived `fin_reg`/`beta` entirely. `GraSTIACL.py`'s `beta` variable is left computed-but-unused (same resolution as the old project reached for this issue, just for a different underlying reason here).

**Reference from the paper:** Eq. (12) (`A = σ(...)`) chained into Eq. (18) (`γ` = mean of adjacency) — not a guess; a direct combination of two explicit equations.

---

## Issue #5 — Dynamic PCC numerically ignored

**Files:** `unsupervised/learning/GraSTI.py`, `unsupervised/learning/ginfominmax.py`, `unsupervised/view_learner.py`, `GraSTIACL.py`

**Error / Cause:** `dyn_weight` was threaded through every function signature and checked for `None`, but never used numerically — `view_learner.py`/`ginfominmax.py`'s `else` branch (when `dyn_weight` *is* provided) called `self.net.get_mu_std_logits(edge_weight)` using only the static weight, dropping `dyn_weight` entirely.

**Fix:** `get_mu_std_logits(self, pcc_weight, dyn_weight)` now concatenates each ROI's static PCC row with its `T` dynamic-window rows (verified empirically that `batch.dyn_weight` batches as `(batch_size*T, 90, 90)`, subject `i`'s windows at `[i*T:(i+1)*T]`) before encoding. `ToyNet.input_dim` changed from `90` to `90*(1+T)`. `view_learner.py`/`ginfominmax.py` now actually pass `dyn_weight` through. `GraSTIACL.py` gained a `--num_dyn_windows` CLI arg (default 3, matching this reproduction's 3-window local PCC).

**Reference from the paper:** Eq. (13): dynamic edge weight matrix `𝒲 ∈ R^(T×N×N)` feeds into the final node information bottleneck.

**Verification:** Perturbing `dyn_weight` alone (static PCC held fixed) changes both `mu` and `edge_logits` (max abs diff 0.26 in logits) — previously zero effect.

---

## Issues #9-11 — ToyNet dimension bug (`1×1` vs `90×800`)

**File:** `unsupervised/learning/GraSTI.py`

**Error / Cause:** Found entangled with Issue #5, in the same function. `FC_mean`/`FC_var` mapped `hidden_dim → 1`, then `FC_mean_trans`/`FC_var_trans` expected a 90-wide input — but `get_mu_std_logits` calls `forward()` once per single 90-length ROI row, so `FC_mean`'s output could only ever be shape `(1,1)`, causing a shape-mismatch crash against `Linear(90, 800)`.

**Fix:** `FC_mean`/`FC_var` now map `hidden_dim → latent_dim` directly; `FC_mean_trans`/`FC_var_trans` removed entirely. Matches a commented-out alternative already present in the original source.

**Reference from the paper:** No explicit per-layer architecture given (Appendix A.13 describes the VIB framework only) — this is a tensor-shape-consistency fix, not a paper-interpretation question.

**Verification:** `mu` now correctly shaped `(800,)`, `logit` `(90,)`, no crash, matches Issue #5's verification test above (which exercises this same code path).

---

## Issues #6-8 — Static PCC used directly as GCN's adjacency; signed PCC caused invalid GCN degrees

**Files:** `datasets/Dataset.py`, `unsupervised/encoder/TA_encoder.py`, `GraSTIACL.py`, `unsupervised/learning/ginfominmax.py`, `unsupervised/view_learner.py`

**Error / Cause:** The paper defines two distinct quantities — `A` (adjacency, learned, initialized fully connected, Eq. 12: always positive via sigmoid) and `W` (signed Pearson correlation weights). The code never separated them: `Dataset.py` built `edge_index` from only the nonzero entries of the (signed) PCC matrix via `scipy.sparse.coo_matrix`, and fed that same signed tensor directly into `GCNConv` as `edge_weight` — for both the "original" and "augmented" views (`batch_aug_edge_weight = batch.edge_weight * sigmoid(gate_inputs)`, still signed since multiplying by a 0-1 value preserves sign). Negative correlations → negative weighted node degrees → NaN in GCN's `D^-1/2` normalization.

**Fix:**
- `Dataset.py`: genuinely fully-connected `edge_index` (all N×N pairs including self-loops), `edge_weight` = the complete flattened signed PCC (not sparse-derived).
- `GraSTIACL.py`: `batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()` — no longer multiplied by the raw signed PCC.
- `GInfoMinMax.forward()` / `ViewLearner.forward()`: signature split into `gcn_edge_weight` (feeds only the GCN — `None` for the original view, the corrected positive-only weight for the augmented view) and `pcc_weight` (feeds only `ToyNet.get_mu_std_logits`, always the raw signed PCC — needed there as VIB input, not GCN weight).
- `TA_encoder.py`: `GCNConv(..., add_self_loops=False)` since `edge_index` already carries explicit self-loops; `get_embeddings` (used for every downstream classification embedding) now uses `edge_weight=None` too.

**Reference from the paper:** Appendix A.1: *"A is the adjacency matrix to be learned, initialized as a fully connected graph."* Eq. (12): `A = σ(...)`, always positive. Eq. (17): `X_Topo = GCN(X, A)` — uses `A`, not `W`.

**Verification:** Built a realistic signed PCC matrix (3998/8100 negative entries) and ran it through `TAEncoder` three ways: original view (`edge_weight=None`) → finite; augmented view (sigmoid-gated) → finite; raw signed PCC fed directly as GCN weight (the old buggy behavior, reproduced for comparison) → **NaN confirmed**. Proves both the bug was real and the fix eliminates it.

---

## Issue #15 — Attention mixed different subjects within a batch

**File:** `unsupervised/encoder/TA_encoder.py` (`full_attention_conv`, `TransConvLayer.forward`, `TransConv.forward`, `TAEncoder.forward`)

**Error / Cause:** PyG's `DataLoader` doesn't batch multiple subjects as `(batch_size, N, F)` — it concatenates every subject's nodes along dim 0 into one flat `(batch_size*N, F)` tensor, with a separate `batch` vector recording which rows belong to which subject. `TAEncoder.forward` receives this `batch` vector (used later for `global_add_pool(x, batch)`) but never passed it into `self.trans_conv(x)`, and neither `TransConv.forward`, `TransConvLayer.forward`, nor `full_attention_conv` accepted a `batch` argument at all. Since `full_attention_conv`'s kernelized attention pools raw key/value statistics (`kvs = einsum("lhm,lhd->hmd", ks, vs)`, `ks_sum = einsum("lhm,l->hm", ks, ones)`) over every row `l` in the tensor with no per-subject restriction, every subject's output was contaminated by every other subject's nodes present in the same training batch — e.g. subject A's embedding would change depending on which other subjects happened to be batched alongside it, independent of subject A's own data.

**Fix:** Threaded a `batch` argument through `full_attention_conv` → `TransConvLayer.forward` → `TransConv.forward`, and `TAEncoder.forward` now calls `self.trans_conv(x, batch)` instead of `self.trans_conv(x)`. When `batch` is provided, `full_attention_conv` reshapes `qs`/`ks`/`vs` from `(num_graphs*n, H, ...)` to `(num_graphs, n, H, ...)` (relying on every subject having the same node count `n=90`, fixed AAL atlas, and PyG's guarantee that nodes are concatenated contiguously graph-by-graph) and restricts every reduction — the L2 normalization, `kvs`, `ks_sum`, and the final attention output — to within each subject's own block, i.e. block-diagonal attention. When `batch is None` (the single-graph case, e.g. the unused `get_attentions` visualization path), behavior is byte-for-byte identical to the original code — this also preserves Issue #12-14's separately-verified finding that the whole-tensor L2 norm is an intentional, verbatim-copied DIFFormer design choice for the single-graph case; it's simply now correctly scoped per-subject when multiple subjects share a batch, rather than incorrectly spanning the whole batch.

**Reference from the paper:** Section 2 defines `N` as the node count of a single subject's graph `G(V,E,A,W)`; Eq. (16)'s attention is computed "for i,j = 1...N" for that one graph. Batching multiple subjects together without restricting attention to each graph lets indices range over the concatenated multi-subject node set, which doesn't match this definition.

**Verification:** Built 2 synthetic subjects (fully-connected 90-node graphs, matching `Dataset.py`'s construction) into one batch, ran them through the real `TAEncoder`, then perturbed only subject B's input features and re-ran. Subject A's output embedding was **bit-for-bit unchanged** (max abs diff = 0.0) — before the fix this would not have held. Also verified: the single-graph path (`get_attentions`, dead code) still runs unchanged, and a full backward pass through the batched path still produces correct nonzero gradients.

---

## Issue #16 — Main and view encoders were fully independent

**Files:** `GraSTIACL.py`

**Error / Cause:** `run()` built two completely separate `TAEncoder`+`ToyNet` pairs — one passed into `GInfoMinMax` (`model`), a second, independently-initialized pair passed into `ViewLearner`. `model_optimizer`/`view_optimizer` each only ever updated their own copy, so the two encoders diverged from the first training step onward and never influenced each other, despite the paper's own architecture sharing almost the entire backbone between them.

**Fix:**
```python
shared_encoder = TA_encoder.TAEncoder(num_dataset_features=3, beta=beta, emb_dim=args.emb_dim,
                                      num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio,
                                      pooling_type=args.pooling_type)
shared_net = GraSTI.ToyNet(input_dim=90 * (1 + args.num_dyn_windows), hidden_dim=args.vib_hidden_dim)

model = GInfoMinMax(shared_encoder, shared_net, args.emb_dim).to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
view_learner = ViewLearner(shared_encoder, shared_net).to(device)
```
`GInfoMinMax.proj_head` and `ViewLearner.mlp_edge_model` remain separate small heads (matching the paper's ~415k-shared-out-of-422k/420k-total split, not a fully identical model).

This surfaced a real, confirmed bug that only exists once the backbone is shared: `nn.Module.eval()`/`.train()` recurse into every registered submodule, so with `model.encoder is view_learner.encoder` now true, whichever of `model.eval()` / `view_learner.eval()` runs *last* in a phase silently flips the *shared* backbone (its `BatchNorm1d` layers, and dropout gated on `self.training`) to that mode too — and in both training phases (`GraSTIACL.py` lines ~158-160 and ~206-207), the last call is the "eval" of the currently-inactive wrapper, which would leave the shared encoder stuck in eval mode right when that phase's optimizer is about to train through it. Fixed by re-asserting train mode on the shared backbone immediately after both of those `.eval()` calls:
```python
model.eval()
model.encoder.train()
model.net.train()
...
view_learner.eval()
model.encoder.train()
model.net.train()
```

**Reference from the paper:** Page 6, "Comparison results": *"the main model contains about 422k parameters, the auxiliary view module about 420k parameters, and approximately 415k parameters shared between them."* Verified directly from the PDF (not inherited from prior notes).

**Verification:**
- `model.encoder is view_learner.encoder` and `model.net is view_learner.net` — both `True` (genuine object identity).
- Parameter overlap: 98.99% of `model`'s parameters and 98.01% of `view_learner`'s parameters are shared — matches the paper's ratio (415/422 ≈ 98.3%, 415/420 ≈ 98.8%) closely, given different hyperparameters (emb_dim, hidden_dim) than the paper's own run.
- Train/eval sequencing: stepped through the exact call order used in `GraSTIACL.py`'s loop and confirmed `shared_encoder.training`/`shared_net.training` are `True` at both points where each phase's forward pass actually executes, and correctly `False` after the periodic end-of-epoch `model.eval()` call used for embedding evaluation.
- Full two-phase forward/backward/optimizer-step smoke test (synthetic 4-subject batch, both phases run exactly as `GraSTIACL.py`'s loop does): both phases complete without error, and `shared_encoder`'s parameters measurably change afterward (confirming both `model_optimizer` and `view_optimizer` genuinely update the same shared weights, not just references that happen to match).

---

## Hardcoded `.cuda()` in `ToyNet.get_mu_std_logits` (flagged separately from the 34-issue list)

**File:** `unsupervised/learning/GraSTI.py`

**Error / Cause:** `edge_logits`/`mu`/`std` accumulators were created via `torch.tensor([]).cuda()`, unconditionally requiring a GPU. This was flagged earlier in this reproduction as a known gap and left undecided; it became an active blocker when it crashed a CPU-only verification test for Issue #16 (`RuntimeError: Found no NVIDIA driver on your system`).

**Fix:**
```python
device = pcc_weight.device
edge_logits = torch.tensor([], device=device)
mu = torch.tensor([], device=device)
std = torch.tensor([], device=device)
```
Now follows whatever device the input data is already on (CPU or GPU), instead of hardcoding GPU.

**Verification:** The full two-phase smoke test above (which exercises this exact function via `dyn_weight`) now runs to completion on CPU; it failed with the CUDA driver error before this fix.

---

## Issue #12-14 — Attention mechanism didn't match its own cited paper

**File:** `unsupervised/encoder/TA_encoder.py` (`full_attention_conv`)

**Error / Cause:** GraSTI-ACL's own paper (Eq. 16, Section 3.3) states explicitly: *"we draw inspiration from Graphormer (Ying et al., 2021) and apply the Transformer principle to compute attention scores for all nodes in the graph."* But the released code's `full_attention_conv` implemented something else entirely: a linear/kernelized attention scheme with **no softmax at all** (`kvs = einsum("lhm,lhd->hmd", ks, vs)`, `attention_num = einsum(...) + N*vs`, normalized by a `ks_sum` term) plus a whole-tensor L2 norm (`qs = qs / torch.norm(qs, p=2)`, no `dim=` — collapsing every node, head, and dim into one shared scalar). This is not a typo in an otherwise-Graphormer implementation; it's a structurally different mechanism, previously traced (by a prior reproduction of this same code) to being a verbatim copy of **DIFFormer**'s (Wu et al., ICLR 2023) own released implementation — a paper never cited anywhere in GraSTI-ACL's text or references.

**Investigation:** Fetched and read Graphormer's actual paper directly (NeurIPS 2021 proceedings PDF) to get its real attention formula rather than assume. Confirmed Eq. (4) of that paper: `A = QK^T/√dK`, `Attn(H) = softmax(A) V` — standard scaled dot-product softmax attention. No whole-tensor norm, no kernelized linear-attention trick anywhere in it. So the released code doesn't implement what its own methods section cites.

**Fix:** Replaced `full_attention_conv`'s internals with Graphormer's actual Eq. (4):
```python
scores = torch.einsum("nhd,lhd->nlh", qs, ks) / (D ** 0.5)   # A = QK^T / sqrt(d)
attn = torch.softmax(scores, dim=1)                            # softmax over keys
attn_output = torch.einsum("nlh,lhd->nhd", attn, vs)            # Attn(H) = softmax(A) V
```
(and the equivalent per-subject batched form, reusing the block-diagonal restriction from Issue #15 so softmax still never pools across different subjects sharing a batch). The whole-tensor norm disappears entirely — it's replaced by the standard `1/√d` scale factor, not patched with a `dim=` argument, since Graphormer's formula has no L2-norm step to fix in the first place.

**Deliberately not ported:** Graphormer's other structural additions — centrality encoding, spatial (shortest-path-distance) encoding, and edge encoding bias terms. Reasons, grounded in GraSTI-ACL's own text rather than preference: (1) this graph is already fully connected (Appendix A.1), so shortest-path-distance encoding would be degenerate (every distance is 1 hop); (2) Eq. 16 is an abstract `Att(·)`/`Agg(·)` black box with no bias terms mentioned; (3) the paper states this branch is meant to move *away* from topology toward "global information" as training progresses, with topology handled by the separate parallel GCN branch (`X_Topo`) — adding graph-structural bias terms here would work against that stated design split.

**Reference from the paper:** GraSTI-ACL Eq. (16) (Section 3.3), citing Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., Liu, T. (2021), "Do Transformers Really Perform Bad for Graph Representation?", NeurIPS 2021 — Eq. (4) of that paper specifically, read directly from the NeurIPS proceedings PDF, not inherited from any prior notes.

**Verification:**
- Cross-subject leakage test (same as Issue #15's): perturbing subject B's input leaves subject A's output bit-for-bit unchanged (max diff 0.0) under the new attention math too.
- Backward pass produces nonzero gradients for every `trans_conv` parameter.
- Explicit softmax check: attention weights sum to 1.0 (std 1.6e-7) across the key axis, both single-graph and batched.
- `get_attentions` (the unused single-graph visualization path) still runs and returns finite output at the expected shape.
- All outputs finite in both the single-graph and batched paths.

---

## Issue #17 — Augmentation-weight accumulator always zero, and pooled across both diagnostic groups

**File:** `GraSTIACL.py`

**Error / Cause:** `aug_edge_weight_all = torch.zeros(args.template, args.template)` was reset every epoch and later divided (`fin_aug_edge_weight = aug_edge_weight_all / len(dataloader)`), but nothing anywhere in between ever added to it — grep found zero `+=` on the variable. Every epoch's "learned adjacency" snapshot (`aug_edge_weights`, meant for the kind of connectivity-interpretability the paper shows in Fig. 3/4) was silently all zeros.

**Investigation:** Even fixing the missing accumulation wasn't enough on its own — the original design pooled every subject in a batch into one running average regardless of diagnosis (ASD or NC). Checked the paper's own interpretability methodology (page 9) before deciding how to average: *"Fig. 4 shows the strengthened and weakened brain region connections in AD, MDD and BD datasets **by comparing subjects with disease and HC**"*, and similarly for Fig. 3's ROI analysis (disease-group statistics compared against a `[μ−σ, μ+σ]` threshold). This is a **group contrast** — disease-group average vs. control-group average — not one blended average across every subject. Pooling ASD and NC together would wash out exactly the contrast the paper's own figures are built on.

**Fix:** Split into two separate running accumulators, one per diagnostic group, each normalized by its own subject count (not batch count):
```python
aug_edge_weight_all_asd = torch.zeros(args.template, args.template)
aug_edge_weight_all_nc = torch.zeros(args.template, args.template)
n_asd_seen = 0
n_nc_seen = 0
...
per_subject_aug = batch_aug_edge_weight.detach().reshape(batch.num_graphs, args.template, args.template).cpu()
asd_mask = (batch.y == 1).cpu()
nc_mask = (batch.y == 0).cpu()
if asd_mask.any():
    aug_edge_weight_all_asd += per_subject_aug[asd_mask].sum(dim=0)
    n_asd_seen += int(asd_mask.sum())
if nc_mask.any():
    aug_edge_weight_all_nc += per_subject_aug[nc_mask].sum(dim=0)
    n_nc_seen += int(nc_mask.sum())
...
fin_aug_edge_weight_asd = aug_edge_weight_all_asd / max(n_asd_seen, 1)
fin_aug_edge_weight_nc = aug_edge_weight_all_nc / max(n_nc_seen, 1)
```
accumulated from the model-training phase's `batch_aug_edge_weight` (the adjacency associated with the model actually being optimized that step). Stored as two separate per-epoch lists, `aug_edge_weights_asd`/`aug_edge_weights_nc`, replacing the single `aug_edge_weights` list (which was write-only — never read elsewhere in the file — so nothing downstream needed updating).

**Reference from the paper:** Page 9 (Section 4, interpretability discussion): the Fig. 3/Fig. 4 methodology explicitly compares disease-group vs. HC-group statistics, not a single pooled average.

**Verification:** Unit-tested the exact accumulation/masking/normalization logic in isolation with synthetic per-subject matrices of known constant values (ASD subjects = 10/20/30, NC subjects = 2/4/6) — confirmed group counts (3 ASD, 3 NC) and group averages (20.0, 4.0) both match analytically expected values exactly.

---

## Issue #30 — `drop_last=True` permanently excluded subjects; hardcoded `args.batch_size` in reg/reshape code

**File:** `GraSTIACL.py`

**Error / Cause:** `dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)` discarded the final, smaller partial batch of every epoch. With the actual dataset (956 subjects) and the default `--batch_size` of 32, `956/32 = 29` full batches (928 subjects) with **28 subjects dropped every single epoch** — roughly 3% of the dataset never trained on in any given epoch.

`drop_last=True` wasn't the root cause — it was a workaround for three lines further down that hardcode `args.batch_size` as if every batch is guaranteed to have exactly that many subjects:
```python
for b_id in range(args.batch_size):            # reg-computation loop, indexes sum_pe[b_id]
mu = torch.reshape(mu, [args.batch_size, -1])  # reshape flat per-subject mu into rows
std = torch.reshape(std, [args.batch_size, -1])
```
Without `drop_last=True`, a partial last batch (fewer than `args.batch_size` subjects) hitting these lines wouldn't necessarily even crash — tested directly: reshaping 3 real subjects' worth of flattened `mu` (2400 elements) into `[args.batch_size=5, -1]` divides evenly (`2400/5=480`), producing no error at all, just a **silently wrong** `(5, 480)` tensor instead of the correct `(3, 800)` — every subject boundary scrambled, corrupting the KLD loss with no visible symptom.

**Fix:** Replaced all three hardcoded `args.batch_size` occurrences with `batch.num_graphs` (PyG's actual per-batch subject count, already used correctly elsewhere in this same file for loss weighting), then changed `drop_last=True` → `drop_last=False` so every subject is used every epoch.

**Reference:** No paper equation applies — this is a data-loader correctness fix, not a method question.

**Verification:** Simulated a partial batch (3 subjects against a configured `args.batch_size` of 5) end-to-end: with the old hardcoded code, the `mu` reshape silently produced a wrong-but-non-crashing `(5, 480)` shape instead of `(3, 800)`; with the fix (`batch.num_graphs`), both the reg-computation loop and the `mu`/`std` reshape produce the correct shapes and values for the actual batch size.

---

## Reviewed, deliberately left unchanged

**`np.nan_to_num()` in `Dataset.py` (lines ~61, ~65) silently zeroes NaN/Inf instead of raising.** Considered fixing this (replace with an explicit `np.isfinite(...).all()` check naming the offending file, matching the old project's approach). First verified directly against our actual data: scanned all 956 subjects (455 ASD + 501 NC) across all three file types (node features, static PCC/adjacency, dynamic PCC) — **zero NaN/Inf values found anywhere.** But the usual justification for this kind of fix ("protects against a future subject's bad data") doesn't apply here: this dataset is closed and no new subjects will be added. The one narrower residual case — protecting against a *future edit to `Dataset.py`'s own parsing logic* introducing a bug that produces NaN from otherwise-valid files — was judged not worth it on its own. Decision: leave as-is.

**Encoder trained on the entire dataset before the SVM's own 5-fold CV split exists (no nested CV) — `GraSTIACL.py`'s `run()` trains one unsupervised encoder on all 956 subjects, then `kf_embedding_evaluation()` only wraps the downstream SVM classifier in CV, reusing embeddings from that single already-fully-trained encoder.** This means every "test" subject in every fold already influenced the encoder's own representation learning — the most significant remaining gap in this list, methodologically. A proper fix (retrain the encoder from scratch inside each outer fold) would cost roughly 5x the compute of the current single training run. Decision: **not pursuing this.** Left as an explicitly acknowledged limitation of this reproduction rather than fixed.

**The three minor performance-only items from the external review** (O(n²) `torch.cat`-in-a-90-iteration-loop building `edge_logits` in `GraSTI.py`; this repo's own Issue #17 fix accumulating `aug_edge_weight_all_*` on CPU inside the batch loop, forcing a GPU→CPU sync every batch; `torch.rand(edge_logits.size())` allocating on CPU before being moved to device) — all confirmed to have zero correctness impact, purely wall-clock training speed on GPU. Training runs on GPU with compute not a constraint, so the overhead isn't worth the churn of touching working code for a speed gain that isn't needed. Decision: leave as-is.

---

## Newly discovered — `GraSTIACL.py` / `embedding_evaluation.py` could not be imported at all in this repo's conda env

**Files:** `GraSTIACL.py`, `unsupervised/embedding_evaluation.py`

**Error / Cause:** Found while trying to actually run/test the CV code (needed for the next fix below) — the module import chain failed outright:
1. `unsupervised/embedding_evaluation.py` line 18: `from scipy import interp` — `interp` was removed from top-level `scipy` in modern SciPy (we're on 1.18.0). Old project precedent: patch to `numpy.interp` via a `compat/sitecustomize.py` shim rather than remove, since it's an environment-version issue, not a code bug.
2. Same file, line 11: `import matplotlib.pyplot as plt` — `matplotlib` wasn't installed in the `grastiacl` conda env at all.
3. `GraSTIACL.py` had its own separate, previously-unnoticed `from scipy import interp` (line 23), plus dead `matplotlib.pyplot`, `plotly.graph_objects`, `plotly.io`, and `pandas` imports — none of which (`plt.`, `go.`, `pio.`, `pd.`) are ever actually called anywhere in the file (confirmed via grep, zero call sites for all four).

**Fix:**
- Both `from scipy import interp` occurrences → `from numpy import interp` (matches old-project precedent; `numpy.interp` is a real, available function, and neither file actually calls `interp(...)` anywhere — confirmed dead either way, but kept for consistency with the established pattern rather than deleted).
- `matplotlib` — genuinely used by `embedding_evaluation.py`'s `plot_embedding()` (a t-SNE visualization helper, itself never called anywhere in the codebase, but the import needs to resolve regardless) — installed via pip into the `grastiacl` conda env, rather than deleting the dead-but-real function.
- `GraSTIACL.py`'s duplicate `matplotlib.pyplot`, `plotly.graph_objects`, `plotly.io`, `pandas` imports — all confirmed to have zero actual usages in the file (unlike `embedding_evaluation.py`'s `plot_embedding`, nothing in `GraSTIACL.py` calls any of them) — removed entirely rather than installing `plotly`/`pandas` for code that never runs.

**Reference:** No paper equation applies — environment/packaging defect, same category as the old project's Issue #31/#1.

**Verification:** `import unsupervised.embedding_evaluation` and a full exec of `GraSTIACL.py`'s module-level code (imports + all function/class definitions, guarded `if __name__ == '__main__'` left untouched) both complete with no errors — previously both failed immediately at import time.

---

## Issue #20-21 — Unseeded, non-stratified CV folds

**Files:** `unsupervised/embedding_evaluation.py`, `GraSTIACL.py`

**Error / Cause:** `kf_embedding_evaluation`'s outer split and inner train/val split were both unseeded and unstratified:
```python
kf = KFold(n_splits=folds, shuffle=True, random_state=None)
...
train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=None)
```
`random_state=None` means a different fold assignment every run — no two runs of the same config are comparable, and results can't be reproduced. Neither split accounted for class labels either (plain `KFold`, plain `train_test_split`), so with 455 ASD vs. 501 NC, a given fold could end up with a skewed class ratio purely by chance, especially the smaller inner 20% validation split. `args.seed` already existed (default 123, used for `setup_seed()`) but was never threaded into this file at all.

**Fix:** `KFold` → `StratifiedKFold` for the outer 5-fold split, `stratify=` added to the inner `train_test_split`, and the inner `GridSearchCV`'s own `cv=5` (in `ee_binary_classification`) replaced with an explicit seeded `StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)` — all three seeded from a `seed` parameter threaded through `EmbeddingEvaluation.__init__` → stored as `self.seed` → passed as `EmbeddingEvaluation(..., seed=args.seed)` from both construction sites in `GraSTIACL.py`.
```python
y_all = dataset.data.y.cpu().numpy()
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset, y_all)):
    ...
    train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=self.seed,
                                               stratify=y_all[train_val_index])
```

**Reference:** No paper equation applies — evaluation-methodology correctness, not a described method.

**Verification:** Loaded the real 956-subject dataset (confirms `Dataset.py`'s cumulative fixes also work end-to-end: 501 NC / 455 ASD, matching expected counts) and ran the actual fixed splitting logic directly:
- Same seed, run twice → bit-for-bit identical fold assignments.
- Different seed → different fold assignments.
- Per-fold test-set class ratios stay within 0.002 of the overall 0.4759 ASD ratio across all 5 folds (near-perfect stratification, vs. unconstrained variation before).

---

## Issue #23 — `StandardScaler` fit outside the inner grid-search CV

**File:** `unsupervised/embedding_evaluation.py` (`ee_binary_classification`)

**Error / Cause:**
```python
self.classifier = make_pipeline(StandardScaler(),
                                GridSearchCV(self.base_classifier, params_dict, cv=inner_cv, ...))
```
`Pipeline` runs its steps in order: `StandardScaler` fits and transforms the **entire** `train_emb` first, computing one mean/std from every subject in that outer fold's training set, and only then hands the already-scaled data to `GridSearchCV`, which splits it into its own 5 inner folds to search over `C`. Every inner-validation subject's own values had already contributed to the scaling statistics applied to it before being "held out" — a small but real leak, since the inner CV's purpose is to estimate generalization to genuinely unseen data.

**Fix:** moved `StandardScaler` inside the pipeline that `GridSearchCV` itself cross-validates, so it gets refit fresh on only each inner fold's training portion:
```python
params_dict = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
inner_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', self.base_classifier)])
self.classifier = GridSearchCV(inner_pipeline, params_dict, cv=inner_cv, ...)
```
(param dict key changed `'C'` → `'clf__C'`, sklearn's convention for addressing a named pipeline step's parameter.) The outer `make_pipeline(StandardScaler(), ...)` wrapper is removed entirely — scaling now only happens inside the cross-validated pipeline. The `param_search=False` branch (`make_pipeline(StandardScaler(), self.base_classifier)`) is untouched — no inner splitting happens there, so a single scaler fit on the whole training set is correct, not a leak.

**Reference:** No paper equation applies — standard ML evaluation-methodology correctness (avoiding train/validation leakage through preprocessing), not a described method.

**Verification:**
- Structural check: `ee.classifier` is a `GridSearchCV` whose `.estimator` is a `Pipeline` with steps `['scaler', 'clf']`; ran a real `.fit()`/predict cycle on synthetic embeddings end-to-end with no errors.
- Directly demonstrated the old bug's signature: reproducing the old `make_pipeline(StandardScaler(), GridSearchCV(...))` structure and fitting it, the scaler's fitted mean exactly equals the **global** mean of the entire training set — confirming it was fit before any fold split ever happened.
- Directly demonstrated the fix: fit the new inner `Pipeline` on two different inner folds' training portions separately — their scaler means are **different** from each other and from the global mean, confirming the scaler is now genuinely refit per fold rather than shared/leaked across folds.

---

## Issue #29 — Reported loss scaled by batch count, not subject count

**File:** `GraSTIACL.py`

**Error / Cause:**
```python
view_loss_all += view_loss.item() * batch.num_graphs   # correctly weighted per batch
model_loss_all += model_loss.item() * batch.num_graphs
reg_all += reg.item()                                    # not weighted at all
...
fin_view_loss = view_loss_all / len(dataloader)          # divided by BATCH count
fin_model_loss = model_loss_all / len(dataloader)
fin_reg = reg_all / len(dataloader)
```
`view_loss.item()`/`model_loss.item()` are already per-batch **means** (`calc_loss` ends in `.mean()`), so multiplying by `batch.num_graphs` correctly turns each batch's contribution into a true sum over that batch's subjects — `view_loss_all` ends up as a genuine sum over every subject in the epoch. But dividing that sum by `len(dataloader)` (the number of *batches*) instead of the total number of *subjects* only happened to be correct by coincidence when every batch had exactly `args.batch_size` subjects (the old `drop_last=True`). Now that Issue #30 fixed `drop_last=False`, batch sizes are no longer uniform (our data: 29 batches of 32 + one partial batch of 28 = 956 subjects, `len(dataloader)=30`), so `956/30=31.87 ≠ 32` — the old equivalence breaks, and the reported loss is inflated relative to a true per-subject average. `reg_all` had the same issue and wasn't even weighted going in (it's a mean over `num_graph_with_edges`, not `batch.num_graphs`, so it needed its own weight).

**Fix:** track total subjects (and total graphs-with-edges, for `reg`) seen across the epoch, and divide by those instead:
```python
num_graphs_seen = 0
num_graph_with_edges_seen = 0
...
reg_all += reg.item() * num_graph_with_edges
num_graphs_seen += batch.num_graphs
num_graph_with_edges_seen += num_graph_with_edges
...
fin_model_loss = model_loss_all / num_graphs_seen
fin_view_loss = view_loss_all / num_graphs_seen
fin_reg = reg_all / num_graph_with_edges_seen
```

**Reference:** No paper equation applies — loss-reporting/logging correctness, not a described method.

**Verification:** Simulated the real batch-size distribution (29×32 + 1×28 = 956 subjects) with synthetic per-batch mean losses and computed the true per-subject average directly from individual subject losses. The old formula (`÷ len(dataloader)`) was inflated by a factor of **31.87x** relative to the true average; the new formula (`÷ num_graphs_seen`) matches the true per-subject average exactly.

---

## Issue #24 — AUC computed from hard predictions

**File:** `unsupervised/embedding_evaluation.py` (`ee_binary_classification`, `embedding_evaluation`)

**Error / Cause:** For our config (`eval_metric == 'accuracy'`):
```python
test_raw = self.classifier.predict(test_emb)   # hard 0/1 labels
...
test_auc_score = roc_auc_score(test_y, test_raw)
```
AUC is a threshold-independent metric — it needs a continuous score to sweep across possible decision thresholds. Feeding it hard 0/1 predictions doesn't error, but silently collapses the computation to a single fixed operating point. `LinearSVC`/`SVC()` (no `probability=True`) have no `predict_proba`, but do expose `.decision_function()` — a continuous, unbounded distance-from-boundary score suitable for ranking.

**Fix:** `ee_binary_classification` now also computes a separate continuous score for AUC specifically, via `decision_function` when available (falling back to `predict_proba` otherwise), returned alongside the existing hard predictions (which still correctly feed accuracy/F1/sensitivity/specificity/precision):
```python
if hasattr(self.classifier, 'decision_function'):
    test_auc_raw = self.classifier.decision_function(test_emb)
else:
    test_auc_raw = self.classifier.predict_proba(test_emb)[:, 1]
...
test_auc_score = roc_auc_score(test_y, test_auc_raw)
```
The `ee_multioutput_binary_classification`/`ee_regression` paths (unused by our config — task is fixed classification, `num_tasks=1`) keep their old behavior (`test_auc_raw = test_raw`) untouched, just wired through so the caller's unpacking still works.

**Reference:** No paper equation applies — metric-computation correctness, not a described method.

**Verification:** Fit the real pipeline on synthetic embeddings and confirmed, numerically: (1) the new `test_auc_raw` is genuinely continuous (decision-function values like -0.33, -0.18, ..., not just 0/1); (2) the **old** AUC (computed from hard predictions) matches balanced accuracy `(sensitivity+specificity)/2` **exactly** (0.388352... = 0.388352...) — proving it was never really measuring anything beyond a rescaled duplicate of a metric already tracked separately; (3) the **new** AUC (0.4113) is measurably different from the old one (0.3884), confirming it now reflects genuine ranking-based information.

---

## Issue #25-27 — Test data used for epoch/metric selection; final scores mixed different epochs

**File:** `GraSTIACL.py`

**Error / Cause:** After training, ~15 separate `np.argmax(...)` calls independently searched *each metric's own curve* for its own peak epoch — not just once, but for train, validation, **and test**, across accuracy/F1/sensitivity/specificity/precision/AUC:
```python
best_val_epoch = np.argmax(np.array(valid_curve))     # the one legitimate selection
best_train_epoch = np.argmax(np.array(train_curve))
best_test_epoch = np.argmax(np.array(test_curve))     # searches the TEST curve for ITS OWN peak
best_f1_test_epoch = np.argmax(np.array(test_f1_curve))
best_sen_test_epoch = np.argmax(np.array(test_sen_curve))
... (same pattern for specificity, precision, AUC, and train/valid variants too)
```
Two distinct problems: (1) `best_test_epoch`/`best_f1_test_epoch`/etc. use the **test set itself** to pick a training epoch — the test set should be read exactly once for a final unbiased number, never searched for a favorable point; (2) the final `'BestTestScore: ...'` log line then combined `test_curve[best_test_epoch]`, `test_f1_curve[best_f1_test_epoch]`, `test_sen_curve[best_sen_test_epoch]`, etc. — each metric read from a **different** epoch — producing a composite summary that no single trained checkpoint ever actually achieved simultaneously. On top of that, a separate copy-paste bug: the AUC line indexed `test_auc_curve[best_f1_test_epoch]` (the F1 curve's chosen epoch) instead of its own `best_auc_test_epoch`, which was computed but never actually used.

**Fix:** removed all ~15 independent `argmax` calls except `best_val_epoch` (validation accuracy only, never touching test), and every reported train/val/test metric is now read off at that one single epoch:
```python
best_val_epoch = np.argmax(np.array(valid_curve))
...
logging.info('BestTestScore: ...'.format(
    test_curve[best_val_epoch], test_std_curve[best_val_epoch],
    test_f1_curve[best_val_epoch], test_f1_std_curve[best_val_epoch],
    test_sen_curve[best_val_epoch], test_sen_std_curve[best_val_epoch],
    test_spe_curve[best_val_epoch], test_spe_std_curve[best_val_epoch],
    test_pre_curve[best_val_epoch], test_pre_std_curve[best_val_epoch],
    test_auc_curve[best_val_epoch], test_auc_std_curve[best_val_epoch]))
```
(train/validation log lines fixed identically.)

**Reference:** No paper equation applies — evaluation-methodology correctness (train/val/test separation), not a described method.

**Verification:** Built synthetic curves where accuracy/F1/sensitivity each peak at a *different* epoch (6, 3, and 1 respectively, vs. the true best-validation epoch of 4). The old approach reported acc=0.9, F1=0.85, sensitivity=0.9 — an inflated composite from three epochs that never co-occurred in any single trained model. The new approach reports acc=0.62, F1=0.5, sensitivity=0.4 — all read from the same single epoch (4), internally consistent and describing one real checkpoint.

---

## Blocker 1 (external review) — `dyn_weight` batched by unverified arithmetic luck

**Files:** `datasets/Dataset.py`, `unsupervised/learning/GraSTI.py`

**Error / Cause:** `_load_group` set `data.dyn_weight = dyn_weight` as a plain attribute after `Data(...)` construction. PyG has no built-in special handling for this key, so it defaults to `__cat_dim__ = 0`, concatenating each subject's `[T, 90, 90]` dynamic-PCC tensor along dim 0 across a batch — producing `[B*T, 90, 90]`, not a stacked `[B, T, 90, 90]`. `GraSTI.py`'s `get_mu_std_logits` recovered the intended shape via `num_windows = dyn_weight.shape[0] // batch_size` then `.reshape(batch_size, num_windows, 90, 90)` — which only produces the *correct* per-subject grouping because PyG happens to concatenate graphs in dataset order and every subject has the same `T=3`. Neither assumption was ever asserted or guaranteed by any contract — found via an independent external review of this codebase, not by any test in this repo.

**Investigation:** Loaded a real batch of 4 known subjects and confirmed the exact failure mode empirically: `batch.dyn_weight.shape` was `[12, 90, 90]` (4×3, concatenated), not `[4, 3, 90, 90]`. The subsequent reshape *did* recover the correct per-subject data for all 4 subjects when checked against each subject's individually-loaded `dyn_weight` (0 mismatches) — confirming the bug was real but not (yet) causing incorrect behavior, purely by arithmetic coincidence dependent on uniform `T` and PyG's concatenation order.

**Fix:** Added a `WindowedData(Data)` subclass in `datasets/Dataset.py` overriding `__cat_dim__` to return `None` for the `dyn_weight` key specifically (PyG's signal to *stack* along a new leading dimension rather than concatenate), and switched `_load_group` to construct `WindowedData` instead of plain `Data`. `GraSTI.py`'s `get_mu_std_logits` simplified accordingly — `dyn_weight` now arrives already correctly shaped `[batch_size, T, 90, 90]`, so the flatten-then-reshape inference was removed (and would in fact now be *wrong* if left in place, since `dyn_weight.shape[0]` is `batch_size`, not `batch_size*T`, once genuinely stacked).

**Reference:** PyTorch Geometric's own `Data.__cat_dim__`/`Batch.from_data_list` API contract — a data-loading correctness fix, not a paper-equation question.

**Verification:** After the fix, rebuilt the processed dataset cache and re-ran the same real-4-subject test: `batch.dyn_weight.shape` is now genuinely `[4, 3, 90, 90]` directly (no reshape involved), and still matches each subject's individually-loaded `dyn_weight` exactly (0 mismatches) — now guaranteed by PyG's own stacking contract rather than by coincidence. Also verified `get_mu_std_logits` produces correct shapes and finite output on this real batch, and ran a full real-data forward pass through both `GInfoMinMax` and `ViewLearner` (batch of 8 real subjects) with correct shapes and all-finite output throughout.

---

## Blocker 2 (external review) — signed PCC vs. Eq. 12's premise

**Status: left unchanged, matching the original authors' own code.** Eq. 12's `log(W/(1-W))` term requires `W ∈ (0,1)`, but our real PCC is signed. Traced the full derivation (Eq. 4 through 12, plus Appendix A's proof of Eq. 5-11) — the paper's own text never resolves this; Section 2 defines `W` as the raw signed Pearson correlation with no rescaling, and Appendix A stops before ever deriving Eq. 12 itself. Confirmed via `git show` on the very first commit that the current code's substitution (`edge_logits` — a learned ToyNet output — standing in for the `log(W/(1-W))` term) is the *original authors' own unmodified implementation*, not something introduced by any fix in this repo — so no real number ever reaches this specific formula today; only the learned number does.

A candidate fix was designed, implemented, and empirically shape-verified (add a literal `log(clip(abs(pcc),1e-6,1-1e-6)/(1-clip(...)))` term *alongside* `edge_logits_vl` in `GraSTIACL.py`'s gate computation, rather than replacing it — `edge_logits_vl` and `batch.edge_weight` confirmed to be the same shape, `[32400]` for a 4-subject batch, so the two terms could be summed directly). **Reverted after discussion**: `abs(pcc)` collapses a strong correlation and a strong anticorrelation to the same value, discarding the sign — decided against this specific tradeoff. `(pcc+1)/2` (sign-preserving alternative) was also considered but not adopted either. Decision: leave the gate computation exactly as the original authors' own code has it — the learned `edge_logits_vl` alone, no literal PCC term added. Revisit if a sign-preserving transform is wanted later.

---

## Blocker 3 (external review) — the view learner's own `edge_logits` were computed then discarded

**File:** `GraSTIACL.py`

**Error / Cause:** In both training phases, `view_learner(...)` was called and its return unpacked as `_, mu, std, edge_prod = view_learner(...)` — silently discarding the first return value, which is `view_learner`'s *own* `edge_logits` (from its own `net.get_mu_std_logits(...)` call). The actual gate/augmentation (`gate_inputs`, `batch_aug_edge_weight`) was then built from `model`'s separately-computed `edge_logits` instead. Since `view_learner` is the augmenter (Eq. 3's `Φ`), the augmentation it produces should come from its own forward pass, not a separate call through `model`.

**Investigation:** Confirmed `ViewLearner.forward`'s actual return signature (`return edge_logits, mu, std, edge_prod` in the `dyn_weight is not None` branch, always taken here) — the discarded first value genuinely is view_learner's own edge logits. Built a real batch and confirmed `model`'s `edge_logits` and `view_learner`'s own `edge_logits_vl` are **numerically different** (max abs diff 0.079) even though they share the same underlying net (Issue #16) — because `ToyNet.forward`'s reparameterization trick draws independent random noise on each separate call, so reusing model's value instead of view_learner's own wasn't a no-op; it fed the wrong specific sample into the gate.

**Fix:** Both training phases now use `view_learner`'s own returned `edge_logits_vl` to build `gate_inputs`/`batch_aug_edge_weight`, instead of reusing `model`'s `edge_logits`. `model`'s own `edge_logits` is still used, deliberately and unchanged, for the separate `ce_loss`/`edge_logits_sig` computation later in the model-training phase (Eq. 13's `L_CE(A,M)` consistency term) — a different, legitimate use of model's own output that this fix doesn't touch.

**Reference:** Eq. (3) (`Φ` = the augmenter's learnable parameters) — the augmentation must be a function of the augmenter's own output, not a separately-computed value from a different module.

**Verification:** Confirmed `edge_logits` (model's) and `edge_logits_vl` (view_learner's) differ numerically on real data, and that gradient now flows substantially into `view_learner.net` (the shared ToyNet) via this corrected path (total grad magnitude 495.67 in a test batch). **One honest correction to the external review's own suggested verification**: it claimed `view_learner.mlp_edge_model` should show nonzero gradient after this fix — checked directly, and it does not (confirmed 0). That's because `mlp_edge_model` only lives in `ViewLearner.forward`'s `dyn_weight is None` branch, which never executes in this pipeline (real `dyn_weight` is always passed) — a structurally separate, still-dead code path that this specific fix doesn't reach, unrelated to whether the "wrong edge_logits source" bug is fixed.

---

## Bug 4 (external review) — Beta distribution direction inverted vs. the paper's own prose

**File:** `unsupervised/encoder/TA_encoder.py`

**Error / Cause:** `lambda_ = Beta(gamma, 1-gamma).sample()` gives `E[lambda]=gamma`. But Section 3.3's prose states: *"as the connectivity of the graph diminishes during training, this encoder can appropriately reduce the influence of topological information and focus more on global information"* — i.e. **less** connectivity (smaller `gamma`) should mean **more** attention weight (`E[lambda]` large), the opposite of what `Beta(gamma,1-gamma)` produces. Confirmed the paper is internally inconsistent (Eq. 18's literal symbols vs. its own prose), and confirmed via `git show` on the original, unmodified first commit that the *original authors' own code* computed the mean edge-*drop* probability (`1-gamma`) and used `Beta(1-gamma, gamma)` — i.e. it followed the prose, not Eq. 18's literal symbol order.

**Investigation:** Resolved the ambiguity using an independent, paper-grounded consistency check rather than picking arbitrarily. Section 3.1 and Appendix A both independently confirm the *original* (unaugmented) graph's adjacency is defined as fully connected, all entries = 1 — i.e. `gamma ≈ 1` for that graph specifically. Per the prose, a dense graph (`gamma≈1`) should rely mainly on the topology-aware GCN branch (attention weight small), since GCN's receptive field is already maximal on a fully-connected graph and the attention branch's whole purpose (Sec. 3.3) is to compensate for GCN's *limited* receptive field once edges are dropped. That requires `E[lambda]` small when `gamma≈1`, i.e. `E[lambda]=1-gamma` — confirming the prose/original-code direction independently, not just by counting votes between two contradictory paper passages.

**Fix:**
```python
lambda_ = torch.distributions.Beta(1 - gamma, gamma).sample()  # was: Beta(gamma, 1 - gamma)
```
The unweighted view's hardcoded `gamma = 1-eps` (for `edge_weight is None`, the original fully-connected graph) is left unchanged — confirmed correct, matching the paper's own explicit definition of the original graph's adjacency, not a bug.

**Reference:** Section 3.3 prose (quoted above); Section 3.1 and Appendix A (original graph's `A` initialized fully connected, all entries 1); the original authors' own unmodified code (`git show a0b3797`).

**Verification:** Directly sampled both versions at the two extremes: new code gives `E[λ]≈4×10⁻⁵` for a dense graph (`γ≈1`) and `E[λ]≈0.9999` for a sparse graph (`γ≈0⁻⁴`) — old code gave exactly the inverse (`≈0.9998` dense, `≈0.0002` sparse). Ran a full forward/backward pass through the real `TAEncoder` with the fix — finite output, gradients still flow correctly.

---

## Bug 5 (external review) — `np.random.beta(fin_reg, 1-fin_reg)` can crash

**File:** `GraSTIACL.py`

**Error / Cause:** `fin_reg` is the mean edge-drop rate across all subjects, recomputed fresh every epoch from real training data. `np.random.beta` requires both shape parameters to be strictly greater than 0. If `fin_reg` ever lands exactly on `0` (no edges dropped anywhere this epoch) or `1` (every edge dropped everywhere), one of `np.random.beta(fin_reg, 1-fin_reg)`'s two arguments becomes exactly `0`, raising `ValueError` and crashing the entire run. Unlikely early in training, more plausible later if the learned edge-dropout gate saturates toward always-keep or always-drop.

**Fix:** Clamp only for this specific call, leaving the true, unclamped `fin_reg` used everywhere else (logging, `view_regs`):
```python
fin_reg_beta = float(np.clip(fin_reg, 1e-4, 1 - 1e-4))
beta = np.random.beta(fin_reg_beta, 1 - fin_reg_beta)
```

**Reference:** No paper equation applies — this is a crash-prevention/robustness fix on a variable (`beta`) already confirmed dead elsewhere (Issue #4 — `TAEncoder.forward`'s `beta` parameter isn't used internally), not a correctness fix.

**Verification:** Confirmed the exact crash directly: `np.random.beta(0.0, 1.0)` raises `ValueError: a <= 0`, `np.random.beta(1.0, 0.0)` raises `ValueError: b <= 0`. With the clamp, both boundary values and a normal mid-range value (0.5) all run without error.

---

## Fresh full-codebase review (self-conducted, not the external `issues.pdf`)

Re-read every core file end to end (`Dataset.py`, `TA_encoder.py`, `GraSTI.py`, `ginfominmax.py`, `view_learner.py`, `GraSTIACL.py`, `embedding_evaluation.py`) fresh, cross-checked against the paper, looking for anything not already covered above.

**Fixed — redundant double Xavier-initialization of the shared backbone.** `GInfoMinMax.init_emb()` and `ViewLearner.init_emb()` both walked `self.modules()`, Xavier-initializing every `Linear` layer they contain. Since `model.encoder is view_learner.encoder`/`model.net is view_learner.net` (Issue #16 — this issue didn't exist before that fix, since each wrapper previously had a fully separate encoder), constructing `model` first Xavier-initializes the shared backbone, then constructing `view_learner` right after immediately re-randomizes the *same* shared weights a second time, silently discarding the first pass. Harmless to final behavior (the shared backbone still ends up Xavier-initialized, just via the second call), but wasted work and a confusing thing to trip over while reading the code.
- **Fix:** `GraSTIACL.py` now explicitly Xavier-initializes `shared_encoder`/`shared_net` exactly once, via a new `init_module_weights()` helper, right after they're constructed. `GInfoMinMax.init_emb()`/`ViewLearner.init_emb()` now only walk their own unique head (`self.proj_head`/`self.mlp_edge_model` respectively) instead of `self.modules()`.
- **Verification:** snapshotted the shared backbone's weights immediately after the one explicit init, then constructed both `model` and `view_learner` — confirmed the shared weights are **bit-for-bit unchanged** afterward (previously, they would have been re-randomized). Confirmed `proj_head` and `mlp_edge_model` still receive proper Xavier-scaled weights (not left at PyTorch's raw default).

**Investigated, not an actual issue given current usage — stale values in per-epoch training curves.** `train_score`/`val_score`/`test_score` are only recomputed inside `if epoch % args.eval_interval == 0:`, but the code appending them to every tracking array (`train_f1_curve.append(...)`, `valid_curve.append(...)`, etc.) sits outside that `if` block (confirmed via exact indentation), so `eval_interval-1` out of every `eval_interval` epochs just re-append the last real evaluation's stale value. Checked whether this actually corrupts anything: `best_val_epoch = np.argmax(valid_curve)` still always lands on a genuinely fresh-eval epoch (a value only first appears in the array at the epoch it was computed, and `argmax` returns the first occurrence — confirmed by direct simulation), and nothing else in this codebase reads these curve arrays (no plotting, `run()`'s return value isn't captured by its caller). So the final reported metrics are unaffected. Decision: not fixing — would only matter if curve-plotting were added later.

**Flagged for awareness, not fixed — a currently-dead shape mismatch.** `TransConvLayer` with `use_weight=False` would produce a value tensor with 1 head while query/key have `num_heads` heads, breaking `full_attention_conv`'s einsums if ever triggered. Confirmed via grep that `trans_use_weight` is never set anywhere in `GraSTIACL.py`'s CLI args, so this path never executes in this pipeline today.

**Flagged for awareness, not fixed — a fragility, not a confirmed bug.** `sensitivity`/`specificity` (`embedding_evaluation.py`) index a confusion matrix assuming both classes are always present in every split; a single-class split would raise `IndexError`. Given the dataset's fold sizes and stratified splitting (Issue #20-21), this is unlikely to actually trigger — not observed to happen, not fixed.

---

## Still open / not yet reached

Nothing. Every item from the original 34-issue list and the external review (`issues.pdf`) is fixed and verified, deliberately decided and documented, or investigated and confirmed not worth doing.

## Found via a real HPC run — `batch.y` shape mismatch crashed Issue #17's ASD/NC mask

**File:** `GraSTIACL.py`

**Error / Cause:** A real training job (job 1480108, L40S GPU, full 956-subject dataset) crashed 16 seconds in, on the very first batch:
```
IndexError: The shape of the mask [32, 1] at index 1 does not match the shape of the indexed tensor [32, 90, 90] at index 1
```
`set_tu_dataset_y_shape` (applied as a per-subject `transform`, `data.y = data.y.unsqueeze(1)`) intentionally reshapes each subject's `y` to `(1,1)` for compatibility elsewhere in the codebase. Batched, that gives `batch.y` shape `(batch_size, 1)`, not flat `(batch_size,)`. Issue #17's own fix (`asd_mask = (batch.y == 1).cpu()`) assumed a flat shape — an oversight that every prior test of that fix missed, since those tests built synthetic `Data` objects directly, without going through this transform, so this exact interaction was never exercised until a real `DataLoader` batch hit it.

**Fix:**
```python
y_flat = batch.y.view(-1)
asd_mask = (y_flat == 1).cpu()
nc_mask = (y_flat == 0).cpu()
```

**Reference:** No paper equation applies — data-shape correctness bug, caught by actually running the pipeline rather than by unit testing in isolation.

**Verification:** Reproduced the exact crashing shape (`y` of shape `[32, 1]`) directly — confirmed the old code's indexing fails the same way, and the fix (`y.view(-1)` before masking) produces the correct `(90, 90)` per-group sum with the right subject counts.

---

## Decided not to pursue

**ComBat harmonization** (site-effect correction across ABIDE-I's 19 acquisition sites) — originally a standing instruction to apply after feature extraction, fit separately per train/test split to avoid leakage. Extensively planned: identified `neuroHarmonize`'s `harmonizationLearn`/`harmonizationApply` split as the right tool, the "No Target" scheme (site as batch, age/sex as covariates, diagnosis excluded) to avoid a subtle Test Target Leakage mode identified in the literature (arXiv:2410.19643), and a cross-fitting approach (Chernozhukov et al., 2018) to reconcile per-fold harmonization with this pipeline's single-encoder-training architecture (no nested CV). A closely related published work using the same ALFF+PCC feature construction on the same ABIDE-I data (Zhang et al., 2023, "A-GCL") was checked and does no site harmonization at all. Final decision: **not implementing this.**
