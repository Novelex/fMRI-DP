# Layer 2 — Graph Construction: Test Plan

Scope: PCC adjacency, M_ij similarity, dynamic windowed weights. Traced against the actual
code path (`unsupervised/training.py` Phase 1, `unsupervised/view_learner.py`,
`unsupervised/learning/GraSTI.py`), not assumed from the paper alone.

## The real structure of this layer (three pieces, not one)

Tracing `train_one_epoch`'s Phase 1 (`unsupervised/training.py:114-192`) shows graph
construction actually splits into three independently-sourced pieces:

**(a) Static PCC edge weight** — `batch.edge_weight` (from `cropped_matrix`). ALFF-independent.

**(b) Dynamic windowed weights** — `batch.dyn_weight` (from `_dw.mat`). Feeds, together with
(a), into `GraSTI.get_mu_std_logits(pcc_weight, dyn_weight)` (`GraSTI.py:63-92`) to produce
`mu`, `std`, and `edge_logits_vl` — the actual learned adjacency structure (Eq. 5-13's IB
pathway) that determines edge-drop probability (`gate_inputs` → `batch_aug_edge_weight`).
**This function never touches node features `x` at all** — confirmed by reading its signature
and body. ALFF-independent.

**(c) M_ij similarity** — `edge_prod`, returned by `ViewLearner.forward` (`view_learner.py:66-71`)
as `torch.sum(x_mij[src] * x_mij[dst], dim=1)`, where `x_mij = x[:, :3]` when `mij_source='alff'`
(our scope) — literally the raw ALFF node features, dotted. **This is the only ALFF-dependent
piece of graph construction.**

## Why (c) is not a minor detail — it directly drives training

`edge_prod` (M_ij) isn't just computed and discarded. In `training.py:175-186`:

```python
edge_prod_sig = torch.sigmoid(edge_prod.squeeze()).detach()   # M_ij, Eq. 4 -- fixed target
edge_logits_sig = torch.sigmoid((edge_logits_vl + gate_inputs_) / temperature)
ce_loss = F.binary_cross_entropy(edge_logits_sig, edge_prod_sig)
...
view_loss = model.calc_loss(x, x_aug) - ce_lambda * ce_loss - reg_lambda * reg - kld_lambda * kld_loss
```

`ce_loss` is a real term in `view_loss`, weighted by `ce_lambda=2.0` — **the largest of the
three lambda terms** (`reg_lambda=0.2`, `kld_lambda=0.003`, both far smaller). M_ij acts as a
frozen supervision target that `edge_logits_vl` (the learned adjacency, from PCC+dynamic
weights) is pushed toward matching. Swapping ALFF source changes this target directly, at the
heaviest-weighted term in the view_learner's objective — this is where old-vs-new ALFF will
materially change what the adversary learns, not a side effect.

## The landmine: M_ij needs signed input, and Layer 1's [0,1] convention breaks that

Eq. 4 is `M_ij = σ(v_i · v_j)`. For σ to ever read below 0.5 (i.e., for M_ij to express
"dissimilar"), `v_i · v_j` must be able to go negative — which requires `v_i`, `v_j` to be
signed.

Layer 1's plan (already written) settled on A-GCL's paper-faithful per-subject min-max to
`[0,1]` for the new ALFF node features. If that same `[0,1]`-scaled array is what feeds
`x[:, :3]` here, **every entry is non-negative**, so `v_i · v_j >= 0` for every node pair,
every subject, always — `σ(v_i · v_j) >= 0.5` structurally, for the entire dataset. M_ij could
never express "dissimilar," full stop.

This is not a hypothetical — it's the *exact same bug*, in the exact same equation, that this
codebase's own comments already documented and fixed once, for a different code path.
`datasets/Dataset.py:110-120` explicitly did **not** reuse `alff_pcc`'s own `[0,1]`-style
squashing for this reason:

> "Signed `[-1,1]` rescale ... not `[0,1]`. Raw ALFF ... is already signed ... squashing it to
> `[0,1]` here made every ROI-pair dot product `v_i.v_j >= 0` ... structurally unable to
> express 'dissimilar' for any pair, in every run using this feature mode."

So: **Layer 1's node-feature scaling and Layer 2's M_ij-input scaling are not the same
decision**, even though both start from the same raw `alff` array. Two legitimate options,
to decide before building anything:

1. Feed Layer 1's `[0,1]`-scaled array into `x[:, :3]` for M_ij too (paper-literal for the
   node features, but reopens the exact dissimilarity-collapse bug this repo already fixed
   once).
2. Give M_ij's input a **separate, signed** transform — e.g. per-subject min-max to `[-1,1]`,
   mirroring `Dataset.py`'s own `alff_pcc` fix — while Layer 1's `[0,1]` array is still what
   feeds the encoder as general node content. Two different scalings of the same underlying
   `alff` array, used in two different places, matching the precedent this repo already set.

Recommendation: **option 2**, for the same reason the codebase's own comment gives — it's a
concrete, already-litigated bug in this exact repo, not a new theoretical worry.

## Test plan

1. **Canary — (a) and (b) must be unaffected by the ALFF swap.** With encoder weights and
   seed fixed, run `get_mu_std_logits` and `batch.edge_weight` construction under old ALFF vs.
   Layer 1's new-ALFF array as `x`. `edge_weight`, `mu`, `std`, `edge_logits_vl` must come out
   **bit-identical** — any difference here means ALFF is leaking into a pathway it structurally
   shouldn't touch, and that's a bug to find before touching anything else.

2. **M_ij distribution check, old vs new, signed-input version.** Compute `edge_prod` for a
   real batch of subjects under old `norm_matrix` vs. new ALFF with the signed `[-1,1]`
   transform (option 2 above). Compare: sign distribution (`% negative`, i.e. `% pairs
   expressing "dissimilar"` — old ALFF is already signed and should show a real negative
   fraction; new-signed should too, and the two fractions shouldn't be wildly different), and
   correlation of `edge_prod` values between old and new for the same subject/edge pairs
   (expect the same moderate ~0.6-ish range already found at the raw-ALFF level, not perfect
   agreement, not near-zero).

3. **Confirm the `[0,1]` landmine empirically, not just by argument.** As a sanity check (and
   to make the decision above concrete rather than theoretical), compute `edge_prod`'s sign
   distribution under the *unsigned* `[0,1]` array too, and confirm it collapses to ~0% negative
   — directly reproducing the failure mode `Dataset.py`'s comment describes, on the new data,
   before ruling it out.

4. **`ce_loss` sensitivity, frozen network.** With a fixed (untrained, seeded) `view_learner`
   and `model`, compute `ce_loss` for the same batch under old vs. new ALFF (signed M_ij
   input). This isolates exactly how much the *heaviest-weighted* term in `view_loss` shifts
   from the ALFF swap alone, before any training — a magnitude check, not a training run.

5. **Write findings to `layer_testing/layer2_results.md`**: canary pass/fail, M_ij sign-
   distribution comparison (signed vs. the `[0,1]` landmine reproduction), correlation number,
   `ce_loss` delta.

## Explicitly out of scope for Layer 2

- Actual training/convergence effects of the `ce_loss` shift — that's Layer 6.
- GCN/attention encoder internals — Layers 3/4.
- Wiring into `Dataset.py`/`nested_cv/data.py` — shared groundwork, tracked separately, needed
  before any full run but not required to run this layer's isolated checks (all of the above
  can run from standalone tensors in `layer_testing/`, same as Layer 1).

## Deliverables

- `layer_testing/layer2_verify.py` — runs checks 1-4 above on real subjects, prints pass/fail
  + the sign-distribution / correlation / ce_loss numbers.
- `layer_testing/layer2_results.md` — filled-in results.

Not yet written — plan only. Confirm the signed-vs-unsigned M_ij decision (recommend option 2)
and I'll implement.
