# STAGE 2 FINAL CERTIFICATION — GraSTI-ACL GRAPH SEMANTICS

Date: 2026-08-16 · Baseline at start: `8b8f3ad` · Interpreter: `/users/3171356m/miniconda3/envs/grastiacl/bin/python3` (Py 3.12.13, torch 2.5.0+cu121) · Diagnostics seed: 42 · Correctness stage only: no classifier, no accuracy, no epoch campaigns. Dataset.py frozen and untouched.

## A. Pre-change behavior

- Original graph: ALFF nodes + `|PCC|` edges (`abs().clamp(1e-6,1)`).
- Augmented graph (default profile): ALFF + **bare gate** (`sigmoid(gate_inputs)` REPLACED the connectivity — original FC strength discarded). The multiplicative form existed only behind `--replicate_original_code`.
- `gamma_aug` was `mean(batch_aug_edge_weight)` — i.e. mean of the product, not of the gate.

## B. Fix A (commit `3f52c20`)

- `gate = sigmoid(gate_inputs)` defined once per phase — the **retention mask**.
- Augmented GCN edge weight = `current-original-edge-weight * gate` — **universal**, both phases, every profile; no longer flag-controlled. Matches Sec. 3.2 ("weaken their connections") and the authors' released code.
- `gamma_aug = per-subject mean(gate)` — retention information. NOT `mean(weight*gate)`, which is connectivity-strength × retention. Real-data demonstration of the difference: gamma = 0.501/0.501 while mean(W_aug) = 0.234/0.145.
- `edge_drop_out_prob = 1 - gate` — pure reuse; `allclose` against the old expression verified. **fin_reg semantics unchanged: mean drop probability.** `reg_lambda` untouched.
- Synthetic unit test: gcn_w=0.2, gate=0.5 → W_aug=0.1 but gamma=0.5 (both asserted; the silent mistake `gamma=mean(PCC*gate)`=0.1 demonstrated and excluded).
- Fix-A battery: **20/20** (13 semantic asserts + saturation report [0 exact-0, 0 exact-1 gates] + forward/backward smoke: finite losses, finite gradients both networks).

## C. 956-subject signed-degree diagnostic (read-only)

Raw PCC over all 956 subjects × 8100 edges: all finite; range **[−0.7600, +1.0000]** (within theoretical [−1,1]); mean 0.3387, std 0.2499; 9.03% negative, 90.97% positive, 0.0000% exactly zero.

**Authors-literal signed degree** `d_signed = Σ_j W_ij` (86,040 node degrees):
min −28.43, max 72.13, mean 30.49, std 12.54; **377 degrees < 0**, 0 near-zero;
`d_signed^-0.5` before sanitizing: **377 NaN, 0 Inf, across 246 / 956 subjects (25.7%)**.
→ Literal signed GCN normalization is numerically broken on this data — exactly quantified.

**Corrected signed-safe degree** `d_safe = Σ_j |W_ij|`:
min 6.97, max 72.35, mean 32.33, std 10.69, CV 0.331; **all > 0, `d_safe^-0.5` all finite**.
Documented (not a failure): complete graph → modest degree variability.

## D. Corrected signed profile (Fix B, commit `866bb80`)

- `gcn_w = batch.edge_weight` — RAW signed PCC, **unclamped** (range certified at the dataset layer; no silent clamping). Eval path (`get_embeddings`) matched identically.
- `gcn_conv.py` verified EXACTLY correct and untouched: `deg_source = edge_weight.abs()` → `scatter_add` → `deg^-0.5`, while the final coefficient retains the **signed** `edge_weight`. Toy proof: edge −0.8 → coefficient −0.444 (negative message preserved); +0.8 → positive.
- Therefore: normalization uses `D_i = Σ_j |W_ij|`; messages retain signed `W_ij`.

## E. Static vs dynamic PCC routing (verify-only)

Call-site traced: `view_learner(..., gcn_w, batch.edge_weight, batch.dyn_weight, ...)` — `pcc_weight` is the RAW signed static PCC verbatim; `dyn_weight` is `batch.dyn_weight` verbatim. Dynamic PCC never becomes GCN adjacency / abs / gcn_w / gate / W_aug. Flow confirmed: static+dynamic → ToyNet/VIB → edge_logits → Eq.12 → gate; then static × gate → augmented GCN. (ToyNet's internal mathematics = Stage 3.)

## F. Eq. 12 verification

Executable path: `logistic_noise = log(eps) − log(1−eps)`; `gate_inputs = (noise + learned_edge_logits)/temperature`; `gate = sigmoid(gate_inputs)`. Raw PCC never enters `log(PCC/(1−PCC))` anywhere (repo-wide search). **EQ12 = PASS, unchanged.**

## G. Train/eval consistency

Train original view and `get_embeddings` both feed RAW signed PCC under the signed profile (both previously clamped; both now raw — identical semantics). Same subject run twice in `model.eval()`: embeddings identical (max diff 0.0e+00 within allclose).

## H. Interpretability-output audit

`aug_edge_weight_all_asd/nc` accumulate `batch_aug_edge_weight = weight × gate` = **EFFECTIVE AUGMENTED FC**, not the learned retention gate A. Decision: **follow the authors' released implementation**, which accumulates exactly this product for its figures, consistent with the paper's Fig. 3/4 ("strengthened and weakened connectivities" — an FC-like quantity). Comments corrected from "learned adjacency" to "effective augmented FC". No separate gate tensor needed; ambiguity resolved by the authors' own code, documented here.

## I. Deterministic test results

- Equivalence: `1 − sigmoid(gate_inputs)` == `1 − gate` (allclose) ✓
- Synthetic gamma test: 3/3 ✓ (Section B).
- Fix-A real-data battery: 20/20 ✓ (raw PCC 10.0% negative on the two test subjects; gate ∈ [0.000162, 0.999897], zero exact saturation; γ=0.501/0.501).
- Final two-subject certification (signed profile, seed 42): **25/25** ✓ — including: negative PCC stays negative through gcn_w AND through ×gate; signed-safe normalization finite for original and augmented graphs; gamma == mean(gate) and provably ≠ mean(PCC×gate); eval determinism; forward/backward with finite losses and finite gradients in both networks.

## J. Authors-literal vs corrected normalization — the distinction

- **AUTHORS-LITERAL**: degree from the signed sum `Σ_j W_ij` → 377 negative degrees → NaN under `pow(-0.5)` on 246/956 subjects. Broken.
- **OUR CORRECTED signed-safe**: degree from `Σ_j |W_ij|` (always positive) while messages keep the signed weight. All finite, sign preserved.
- Also distinguish: `abs()` for **degree normalization** (corrected profile, keeps signed messages) vs `abs(PCC)` as the **actual edge weight** (default legacy profile, destroys sign). The corrected profile does the former, never the latter.

## Final summary

```
NODE_FEATURE                       = ALFF_PAPER_90x3
ORIGINAL_GRAPH                     = ALFF + SIGNED_STATIC_PCC
ORIGINAL_GCN_DEGREE                = SUM_ABSOLUTE_STATIC_EDGE_MAGNITUDES
DYNAMIC_PCC_ROLE                   = TOYNET_VIB_INPUT_FOR_LEARNING_AUGMENTATION
EQ12_CHANGED                       = NO
GATE                               = LEARNED_RETENTION_MASK
AUGMENTED_GRAPH                    = ALFF + SIGNED_STATIC_PCC_TIMES_GATE
AUGMENTED_GCN_DEGREE               = SUM_ABSOLUTE_AUGMENTED_EDGE_MAGNITUDES
GAMMA                              = PER_SUBJECT_MEAN_GATE
DROP_PROBABILITY                   = ONE_MINUS_GATE
FIN_REG_MEANING_CHANGED            = NO
PCC_MAPPED_TO_01                   = NO
PCC_ABSOLUTIZED_AS_ACTUAL_EDGE     = NO
SIGNED_PCC_PRESERVED_IN_MESSAGES   = YES
TRAIN_EVAL_EDGE_SEMANTICS_IDENTICAL = YES
ALL_STAGE2_TESTS                   = PASS
```
