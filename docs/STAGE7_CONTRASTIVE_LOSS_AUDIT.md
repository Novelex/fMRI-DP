# STAGE 7 — ADVERSARIAL CONTRASTIVE REPRESENTATION / LOSS AUDIT

Date: 2026-08-18 · Stages 1–6 frozen · **Read-only: ZERO production edits** · No epochs, no
accuracy, no Stage-8 ownership conclusions · Primary `alff_new_z` + `paper_intent`;
robustness `alff_m1_z`; sensitivity `paper_printed` · signed_edges=True, standard pooling,
mij alff, paper Mij, static PCC×gate augmentation, current dyn-PCC/ToyNet path.

Method: one shared harness (`stage7_harness.py`) reproducing `train_one_epoch`'s Phase-1/2
exactly; 8 independent diagnostic agents; consequential verdicts then re-measured by
**adversarial verifiers instructed to refute them**. Verifier corrections are marked ⚠ and
override the first-pass claim.

## 1. Frozen-state verification

```
git status        clean (0 modified tracked files)
HEAD              d761219f14b20e4eaa25e31a0157c352ca7dc0ff
origin/main       0cd9e01673abea6d495cf13a7228eb16a622efba
```
⚠ **PROVENANCE GAP**: the Stage-6E production-integration commits `3b1355a` and `d761219`
are on HEAD but **NOT on origin/main** (unpushed). The audit ran against HEAD, which carries
the correct code. `git push origin main` is required before this stage can be called
provenance-complete.

Stage-6 production re-verified: `alff_new_z` len 956 PASS · `alff_m1_z` len 954 PASS ·
per-band mean≈0/std≈1 PASS · finite PASS · **`alff_paper` cache md5 `da1adb254504` unchanged**.

## 2. Paper vs authors-release vs our HEAD

Paper read directly from `docs/grastical.pdf` p.5 §3.4. Eq. 20 as printed:
```
Î_R = (1/m) Σ_{i=1..m} log [ exp(sim(z_{i,1}, z_{i,2})) / Σ_{i'=1, i'≠i}^{m} exp(sim(z_{i,1}, z_{i',2})) ]
```
Eq. 21 `min_Ψ max_Ω I_R` (Ω = projection head, Ψ = TAE). Eq. 22 `min_{Φ,Ψ} max_{Θ,Ω}(I_R+I_N)`.
Prose: "a **shared projection head** … all other augmented graph representations in the batch
as negative samples."

| item | paper | authors-release | our HEAD | verdict |
|---|---|---|---|---|
| positive | same graph i (z_{i,1},z_{i,2}) | `sim_matrix[range(B),range(B)]` | identical | MATCH |
| negatives | all other augmented, i'≠i | `sum(dim)-pos_sim` | identical | MATCH |
| similarity | cosine | cosine | identical | MATCH |
| temperature | **not printed** | 0.2 | 0.2 | authors-only |
| positive in denominator | **NO** | NO | NO | MATCH |
| loss sign | Î_R maximized ⇒ L=−Î_R | `-log(...).mean()` | identical | MATCH |
| one-way vs symmetric | **one-directional** | `sym=True` | `sym=True` | AUTHORS EXTENSION |
| projection sharing | shared head | one `proj_head` | identical | MATCH |
| projection dim | R^D unspecified | Linear-ReLU-Linear | same, 32→32→32 | paper-unspecified |
| label use | none | none | none by default | MATCH |
| memory bank | not in Eq.20 | `calc_regloss` **0 call sites** | **0 call sites** | DEAD IN BOTH |
| augmentation noise | Eq.12 ε₂~U(0,1) | fresh per phase | fresh per phase (194/285) | MATCH |
| train/eval mode | unspecified | P1 eval / P2 train | identical (176-178, 271-273) | MATCH (see §18) |

Source diff of `calc_loss` bodies: **IDENTICAL** (byte-for-byte).

## 3. Manual Eq.20 equality — 66 synthetic cases + real embeddings

| comparison | max abs diff | interpretation |
|---|---|---|
| PROD vs **authors' released file** (imported directly) | **0.000e+00** | bit-exact |
| PROD vs C AUTHORS_SYMMETRIC_T02 (independent re-impl) | 4.77e-07 | float32 reduction noise |
| PROD vs B PAPER_ONEWAY_T02 | 6.12e-01 | symmetric extension alone |
| PROD vs A PAPER_ONEWAY (printed, no τ) | 5.27e+00 | symmetry + temperature |
| **PROD(sym=False) vs B PAPER_ONEWAY_T02** | **2.38e-06** | with symmetry removed, production IS the printed one-way form at τ=0.2 |

```
CURRENT_LOSS_MATCHES_AUTHORS_RELEASE     = YES  (bit-exact)
CURRENT_LOSS_MATCHES_PRINTED_EQ20_EXACTLY = NO  (exactly two differences: τ=0.2, sym extension)
```

## 4. Loss range — negative values are VALID

Per-row `L_i = -s_ii/τ + log(Σ_{j≠i} exp(s_ij/τ))`, cosine∈[-1,1] ⇒ range
**[−2/τ + ln(B−1), +2/τ + ln(B−1)]** = [−10+ln(B−1), +10+ln(B−1)] at τ=0.2. Constructed
antipodal B=2 attains **−9.99905** against the analytic minimum −10.0000. Exact identity
verified to 1e-06: `L_GraSTI = L_NTXent + mean_i log(1−p_ii)`, and log(1−p)<0 always ⇒
GraSTI ≤ NT-Xent. Loss is structurally non-negative only for B ≥ e^10+1 ≈ 22,027 — no
practical batch. No log(0) risk (min exp term 6.74e-03). 240-config finiteness sweep: all
finite, zero NaN. **NEGATIVE_LOSS_VALUES_VALID = YES.**

B=1: denominator exactly 0 ⇒ loss **−inf silently** (no exception). Production guard
**PRESENT**: `assert batch.num_graphs >= 2` at `training.py:162`, before any loss call.
⚠ Operational sharp edge: both entry points build the loader with `drop_last=False`, so
`--batch_size 5` or `191` on N=956 leaves a trailing 1-graph batch and the assert fires
mid-epoch. Fail-loud (never silent −inf), but reachable from a legal command line.

## 5. Positive-pair indexing — PASS

Perturbing one subject's features (gate pinned) moves that subject's row of **both** z_orig
and z_aug by 74–94 (B=8) / 210–371 (B=32) while **every other row moves exactly 0.000000e+00**.
Permuting only the augmented order raises the loss in **10/10** permutations (ΔL +0.0122 to
+1.2784); the inverse permutation restores the canonical loss **bit-for-bit** (|ΔL| = 0.000e+00).
AST trace: all four `model()` call sites take the identical `(batch.batch, batch.x,
batch.edge_index)`; `batch` bound exactly once (line 156); **no reorder token anywhere** in
training.py. Independent runtime recovery of the row→subject map from x and from x_aug:
**orig_ID_order == aug_ID_order EXACTLY** at B=8 and B=32.

## 6. Negative-pair contract — PASS

Exactly **B−1** negatives per anchor in both directions; index set == {0..B−1}\{i} for every
anchor (float64 relative error 1.5e-16 / 4.4e-16); zero duplicates; anchor's own index never
present; no out-of-batch index; no registered buffers (no queue/bank). `calc_loss` is a
staticmethod with `__closure__ = None` and `torch` as its only global. **No label filtering**:
permuting *or* zeroing `batch.y` leaves L bit-identical (1.939869642, |diff| 0.000e+00); a
counterfactual label-filtered denominator would give 1.372827291 with 4 instead of 7 negatives.
Descriptive: fraction of an anchor's negatives sharing its diagnosis = 0.4998 (B=8) …
0.5006 (B=64), matching the analytic 0.500635. **Intended self-supervised behavior, not a defect.**

## 7. Shared encoder + projection — PASS

Forward hooks over two consecutive production calls fired on the **same Python objects**
(`id(encoder)` 140065169952976, `id(proj_head)` 140059303018720 in both); all 18 encoder and
4 proj_head parameters had identical `data_ptr()` and unchanged `_version`. Only one
projection exists (`proj_head.0/1/2`, Linear-ReLU-Linear, no BN/Dropout). `view_learner.encoder
is model.encoder` → **False**, 0 shared data_ptr — Θ and Φ are genuinely disjoint networks.

## 8. Pre- vs post-projection identity — both AT CHANCE

| space | N | top-1 | chance | binomial p | reverse | median rank | pos−meanNeg | pos−maxNeg |
|---|---|---|---|---|---|---|---|---|
| **PRE (h)** | 956 | **0.00105** (1/956) | 0.001046 | 0.632 | 0.00105 | 367 (chance 478) | +0.00702 | −0.09197 |
| **POST (z)** | 956 | **0.00314** (3/956) | 0.001046 | 0.080 | 0.00105 | 337 | +0.00544 | −0.03788 |

Neither space is distinguishable from chance at 0.05. **The projection PRESERVES** (neither
creates nor destroys) same-subject identifiability — the collapse is upstream of the head.
Anisotropy measured: within-space mean pairwise cosine **z_aug 0.9970**, h_aug 0.9925,
z_orig 0.8313.

## 9. Zero-norm safety — CONCERN

Full cohort (956): min‖z_orig‖ **1.397862**, min‖z_aug‖ 115.18, **0 exact zeros, 0 below 1e-8,
0 non-finite**; holds across 5 seeds × 2 profiles (worst 1.377975) and both feature modes —
~22 orders of magnitude above the float32 failure boundary 5.914e-23. **But `calc_loss`
divides by `x.norm(dim=1)` with no guard**: one genuine zero row returns `nan` **silently**,
the `sum(dim=0)` term spreads it to 8/8 columns, backward yields 6784/6784 nan gradients, and
one optimizer step nan-poisons 6784 parameters permanently. A structural exact-zero route
exists (`proj_head[2].bias` initializes to exactly 0, so an all-dead ReLU gives z==0).
Only the untrained state could be measured (epochs frozen) ⇒ **CONCERN, not PASS**.
*Proposed (not applied): a fail-loud assert on min-norm before the division.*

## 10–13. Permutation, gradient direction, one-step Θ, gate-logit adversary

**Permutation (§10).** Correct pairing beats mismatched in every arm, but effect size is
regime-dependent: **model.eval geometry** (production Phase 1) ΔL = **+0.0122 ± 0.0138** nats
(12/15 cells, sign-test p=0.035, top-1 0.175 vs chance 0.125) — *marginal, CONCERN*;
**model.train geometry** (Phase 2) ΔL = **+2.9825 ± 0.4998** (15/15, top-1 0.792). A 4-way
isolation attributes this **100% to BatchNorm mode, ~0% to Beta-λ**. Root cause measured: BN
buffers untouched at init (`num_batches_tracked=[0,0]`, `running_var=1.0`) against true
activation variance ≤7.7e-3 and ≤1e-4 ⇒ eval-BN under-scales by 10–100×.

**Free-embedding gradient (§11).** **Zero strict sign violations of dLoss and of dMargin in
all 20 (geometry × direction × η) rows**, 15 cells each; measured/predicted dL agree to
0.996±0.003. Minimizing L provably improves positive-vs-negative separation.
**THETA_LOCAL_CONTRASTIVE_STEP: §12** one fresh-Adam step on Θ alone (Φ frozen, gate frozen)
lowers the loss in **23/24** configurations. ⚠ *Specificity caveat*: in the eval geometry
**83.5% ± 11.0%** of that reduction is reproduced by stepping on a deliberately scrambled
pairing; in the Phase-2 geometry the scrambled objective instead *raises* the correct loss
(leak −0.253) — i.e. the step is pair-specific only in Phase 2.

**Gate-logit adversary (§13).** Ascent on detached gate logits raises the contrastive loss in
**6/6** configurations at every step size (up to **+2.443 nats**); gradient non-zero in 12/12.
Pathway is **~97% through W_aug = PCC×gate** (cos 0.9975–1.0000) and only 3–5% through
γ_aug→λ. Character: **redistribution, not net retention change** — mean|Δgate| 0.135 per edge
while |mean Δgate| is only 0.0016 (net/abs 0.012), fraction of edges weakened 0.505–0.508.
⚠ *Monitoring trap*: a monitor watching only mean-gate or mean augmented weight would see
essentially nothing while the adversary is acting strongly.

## 14. Stochasticity decomposition

CASE 1 (deterministic gate + deterministic λ) is **bitwise identical across 200 draws**
(variance exactly 0.0, 1 distinct row by `np.unique`).

| source (isolated) | loss sd | loss var | ratio vs gate |
|---|---|---|---|
| gate noise only (CASE 2) | 7.422e-04 | 5.508e-07 | 1× |
| **pure Beta λ (CASE 3c, injected, no BN/dropout confound)** | **5.886e-02** | 3.465e-03 | **6291×** |
| TransConv dropout p=0.5 alone | 1.816e-02 | 3.296e-04 | 598× |
| ToyNet VIB reparameterization | 6.617e-06 | 4.378e-11 | 8e-05× |

Mechanism: γ_aug≈0.49 makes Beta(1−γ,γ) **U-shaped** — the augmented attention branch is
driven fully on (λ>0.9, 22.2% of draws) or fully off (λ<0.1, 19.9%) at random, while the
batch-mean gate barely moves (it averages 8100 near-independent edges).

**STOCHASTIC_VARIANCE_EXCEEDS_BETWEEN_SUBJECT_SIGNAL = YES**: per-subject stochastic sd of the
positive cosine **0.1075** vs between-subject spread at fixed noise 0.0718 (B=8) / 0.0721
(B=32) / 0.0814 (B=64) — **1.3–1.5×**. A single stochastic draw carries more spread than the
biological signal it should encode.

⚠ **Regime inversion (must be carried, not collapsed):** under **train-mode BN** the ordering
**inverts** — gate marginal +2.073e-02 vs Beta marginal +1.016e-02 (2.04×). Production
Phase 2 runs the train-mode regime. Sources are also **super-additive** (var sum 3.47e-03 vs
observed 4.28e-03), so these are marginals, not a clean partition.

⚠ **Two stochastic sources beyond the spec's two**, both newly identified:
1. **ToyNet's VIB `torch.randn_like` is NOT gated on `self.training`** (`GraSTI.py:56`) — it
   resamples in **eval mode too**. Magnitude negligible, but structurally **no forward pass is
   reproducible for the same subject at the same weights without external seed control**.
2. **TransConv `F.dropout(p=0.5)` is live in train mode**, and the spec's suggested BN-eval
   isolation does **not** remove it because `trans_conv.bns` are **LayerNorm, not BatchNorm**
   (mode-invariant ⇒ that call is a no-op). Only the injected-λ variant is pure Beta.

⚠ **Measurement hazard (disclosed and corrected):** train-mode forwards mutate BN
`running_mean`/`running_var` (momentum 0.1), silently shifting every later eval measurement
(a first pass gave pos 0.8395 instead of the true 0.5439 after 800 prior train forwards). All
reported numbers restore the state_dict before every draw.

## 15–16. paper_intent λ-asymmetry and the matched-λ rescue

**Measured (arm A, N=256, eval regime):** λ_orig = **1.000e-04 exactly**, λ_aug = **0.5051 ±
0.0034**. Exact pooled decomposition `h = X_topo + λ·X_atte`:

| quantity | original view | augmented view |
|---|---|---|
| ‖X_atte‖/‖X_topo‖ (raw) | 37.07 | 37.07 |
| **λ·‖X_atte‖/‖X_topo‖** | **0.0037** (pure topology) | **14.57** (attention-dominated) |
| cos(X_atte_orig, X_atte_aug) | **1.000000 exactly** | — |

⚠ **STRUCTURAL FINDING (verifier-confirmed by forward hook: `torch.equal(x_trans_orig,
x_trans_aug) = True`, max|diff| 0.000e+00):** `TA_encoder.py:361` calls
`self.trans_conv(x, batch)` and `TransConv.forward` never receives `edge_index` or
`edge_weight` — **the attention branch is structurally blind to the PCC×gate augmentation.**
The two views therefore share a bit-identical X_atte component. This is by construction, not
a defect, but it is load-bearing for interpreting every retrieval number below.

**Matched-λ isolation (identical graph, identical gate via `gate_override`):**

| arm | λ_orig | λ_aug | pos cos | margin | top-1 (N=256, chance 0.0039) | binomial p |
|---|---|---|---|---|---|---|
| **A — normal paper_intent** | 1e-4 | 0.5051 | 0.6288 | +0.00504 | **0.0117** (3/256) | **0.0799 n.s.** |
| **B g=1.00 (attention off both)** | 1e-4 | 1e-4 | 0.9832 | **+0.11919** | **0.3125** (80/256) | 6.7e-126 |
| B g=0.4949 (attention matched, present) | 0.5051 | 0.5051 | 0.99997 | +0.00300 | 0.9961 | <1e-300 |
| **C — authors_release (attention discarded)** | n/a | n/a | 0.9845 | +0.10631 | **0.3555** (91/256) | 4.4e-149 |

Replicated across 4 init seeds and both feature modes. Arm C uses **bit-identical weights**
(38/38 state_dict keys) and the same gate, so two independent mechanisms for removing
attention agree to within 4.3 points.

### ⚠ Verifier corrections (adversarial pass — these override the first-pass claims)

| claim | status | verifier evidence |
|---|---|---|
| λ-asymmetry destroys positives (**unqualified**) | **UNPROVEN** | Survives only in the **eval-BN regime**. Toggling *only* the BatchNorm1d `.training` flag (λ held deterministic) collapses ‖X_atte‖/‖X_topo‖ **37.02 → 1.07** and lifts arm-A top-1 0.0117 → 0.2500. **BatchNorm is an unruled-out confound the first pass never tested.** |
| mechanism is λ asymmetry, not the gate | **SURVIVES** | Strengthened by the control the first pass omitted: [gate OFF, λ matched] top-1 = **256/256 = 1.0000**; [gate ON, λ matched] 77/256 (p=9.8e-120); [gate OFF, λ asymmetric] collapses. |
| alternative: the gate perturbation itself is the cause | **REFUTED** | Perturbation is large (‖ΔW‖/‖W‖ = 0.586, cos 0.861) yet [gate ON, λ matched] still retrieves 77/256 vs chance 1/256. |
| alternative: attention norm dominance is the cause | **REFUTED as sole cause; confirmed as amplifier** | Counterfactual k-sweep: k=0 → 0.2969, k=0.25 → 0.1211, k=1 → 0.0234, k=2 → 0.0117. |
| alternative: untrained encoder is at chance for *any* view pair | **REFUTED** | Same random init: gate-off + matched λ = **256/256**; X_topo channel alone 98/256 (p=2.6e-164). |
| "positives not separable from negatives" | **REFUTED as too strong** | Top-1 is at chance (3/256, p=0.080) **but** own-subject median rank is **84–90 vs chance 128**, sign-test **p=6.4e-05** — a faint but statistically real residual signal. |
| MATCHED_LAMBDA_RESCUES_POSITIVE_IDENTITY = YES | **SURVIVES** on the unconfounded arms (B g=1.00, C) | Independently reproduced: 77/256 p=9.8e-120, margin +0.1321 (26× arm A). |
| the 99.61% rescue is a shared-component artifact | **SURVIVES, strengthened** | X_atte bit-identity verified directly by forward hook, not inferred. |
| reported losses 3.4446 alongside N=256 retrieval | **REFUTED as an inconsistent pairing** | `calc_loss` at N=256 is log(255)≈5.54; 3.4446 = log(31)+0.011, i.e. a B=32 loss. The table paired cohort-level retrieval with a batch-level loss without saying so. |
| correlations (r ≈ +0.232, p=1.8e-04) | **UNPROVEN** | Signs replicate, magnitudes **25–40% smaller** and one-to-two orders weaker: +0.1801 (p=0.0038). Also, the first three "drivers" are **one measurement, not three** (γ_aug ≡ gate mean, r=+1.000000; γ_aug vs |Δλ| r=−1.000000). |
| scope: all numbers at random initialization | **SURVIVES — upgraded to load-bearing** | Untrained BN running stats are *exactly what makes* ‖X_topo‖ small and hence produce the 37× ratio. The headline eval-regime collapse is inseparable from the epoch-0 condition. |

**⚠ Loss-is-not-a-health-proxy (verifier, substantive):** arm A loss **5.5556** with top-1
0.0117 vs arm B g=mid loss **5.5264** with top-1 **1.0000** — **0.5% apart in loss across a
98.8-point retrieval gap.** The contrastive loss value alone cannot be used to judge
representation health across these arms.

## 17. paper_intent vs paper_printed (bitwise-identical weights, one shared gate)

State_dicts verified identical (38/38 tensors, max diff 0.000e+00); feeding the intent encoder
γ'=1−γ reproduces the printed encoder exactly (max|Δh_orig| = 0.000e+00), proving **λ is the
only difference**.

| regime | metric | paper_intent | paper_printed |
|---|---|---|---|
| **eval / Phase 1 / downstream** | top-1 (N=100, chance 0.01) | 0.04 (p=0.018) | **0.80** (p=4.4e-140) |
| | pos cos / mean-neg | 0.6288 / 0.6238 | 0.999742 / 0.996887 |
| | model-param grad norm | 5.1699 | 0.1455 (**35.5× smaller**) |
| **train / Phase 2 (the only surviving model gradient)** | top-1 | 0.41 | 0.36 |
| | grad-norm ratio | — | **1.23×** (converged) |

⚠ The two phases give **opposite readings** of the profile contrast; the 2×2 isolation shows
**BatchNorm mode, not the profile, is dominant**. paper_printed's eval-regime advantage is
substantially the shared bit-identical X_atte (λ_orig≈1 makes the original view 99.4%
attention). Also asymmetric under Beta sampling: intent 0.040→0.068 while printed
**collapses 0.800→0.020**. Scale confound noted: ‖z‖ 4.81 vs 235.92 (49×), so the raw
gradient ratio 164.6× becomes 2.70× against the unit-normalized embedding.

## 18–19. Phase-mode mismatch (BLOCKER) and fresh per-phase noise

Literal trace: Phase 1 = `view_learner.train()` + `model.eval()` (training.py:176-178);
Phase 2 = `model.train()` + `view_learner.eval()` (271-273) — **statement-for-statement the
authors' release** (GraSTIACL.py:158-160, 204-205).

With identical batch, identical weights and the gate pinned bit-identical (max|Δgate| = 0.000000):

| quantity | Phase-1 mode | Phase-2 mode |
|---|---|---|
| contrastive loss at the **same parameters** | **1.475 ± 0.044** | **0.310 ± 0.127** (ratio **4.76×**) |
| off-diagonal similarity structure | Spearman **r = 0.350** between the two | |
| hardest-negative identity agreement | **20%** (chance 14.3%) | |
| **cos(grad_Φ Phase-1 mode, grad_Φ Phase-2 mode)** | **0.232 ± 0.219**, range **[−0.164, +0.646]** | |

Decomposition: **BatchNorm batch-statistics dominates** (Δloss −1.222 of −1.250; relL2 z_orig
0.822 vs 0.00008 for λ and 0.00003 for dropout); Beta-λ is large on the representation
(relL2 z_aug 0.449, λ_aug sd 0.362 spanning [0.00015, 0.99998]) but **near-cancels in the
loss** (+0.100); TransConv dropout contributes relL2 0.195 / −0.054; the GCN path's
`drop_ratio=0.0` contributes nothing.

**The adversary Φ ascends a surface reading 1.475 while the model Θ descends a surface reading
0.310 at identical parameters and an identical augmentation, and their Φ-gradients are nearly
orthogonal and sometimes anti-aligned.** This breaks the min-max coupling, not merely the
numeric level ⇒ **STAGE7/8 BLOCKER**.

### ⚠ Verifier pass on §18–19 (independent re-measurement, fresh build per forward)

| claim | status | verifier evidence |
|---|---|---|
| PHASE1_PHASE2_SEE_SAME_CONTRASTIVE_PROBLEM = NO | **SURVIVES — strengthened** | The first pass gave no reference distribution. The verifier supplied the missing **within-Phase-2 null**: cross-mode \|ΔL\| **1.7590** vs within-Phase-2 null **0.1522** = **11.56×** at pristine BN, still **2.54×** at fully saturated BN. Off-diagonal structure: cross 0.350 vs **within-P2 0.9193 ± 0.0358**; hardest-negative agreement cross 0.200 vs **within 0.708**. Sign the same in 15/15 configurations at pristine BN. |
| STAGE7_8_BLOCKER = YES | **SURVIVES** | Effect exceeds Phase-2's own noise at **every** BN state built. |
| "the gap never closes and never changes sign" | **REFUTED** | Tested on one seed/batch originally. Over 5 seeds × 3 batches the gap **attenuates and loses sign-consistency as BN warms**: pristine −1.759 ± 0.292 (sign-consistent TRUE) → warm −0.608 ± 0.601, range [−1.735, **+0.053**] (FALSE) → saturated −0.343 ± 0.407 (FALSE). **The blocker is real but shrinks as BN running stats track the data.** |
| dominant term is dropout, not Beta-λ | **REFUTED** | Isolated: λ **0.18313** vs dropout **0.01446** at pristine BN (**12.67×**), λ 0.10145 vs 0.04349 saturated (2.33×); λ > dropout in **5/5 seeds at both BN states**. |
| PHASE_MODE_DOMINANT_TERM = BATCHNORM | **SURVIVES** | Largest \|ΔL\| in **24/24** rows of a profile × feature × seed × BN-state sweep. |
| cos(grad_Φ) = 0.232 ± 0.219 | **seed-specific** | Verifier's own exact grad_Φ at seed 42: **+0.4708 ± 0.2111, range [−0.1255, +0.6783]** — same sd, same negative tail, different centre. Both agree the cross-mode gradient is far from aligned. |
| "relL2 z_orig 0.82241" | **UNPROVEN** | Does not reproduce under any normalization convention tried (verifier gets 40–76 or 0.987). The **98%-of-loss-shift** part *does* reproduce (BN isolated ΔL −1.7280 of a −1.7590 total = 98.2%). |
| BN-buffer restore is required | **SURVIVES** | Order dependence is large: on a pristine model eval loss is 1.967060 (bit-identical over 5 fresh builds); after K train-mode forward pairs it drifts to 1.946 (K=1), 1.894 (K=20), **1.040 (K=100)** with running_var mean 1.0000 → 0.0013. |

⚠ **grad_Φ pathway (measured, ownership deferred to Stage 8):** the contrastive gradient into
the view learner is **ToyNet-only** — `encoder` ‖g‖ = 0, `net` (ToyNet) ‖g‖ = 0.01514,
`mlp_edge_model` ‖g‖ = 0. Recorded as a measurement; **no ownership claim here**.

**§19 fresh noise:** both phases draw independent logistic noise (194/285), an exact
authors-release match. With logits fixed: correlation(gate_p1, gate_p2) = **0.0438 ± 0.0036**,
mean|Δgate| 0.329, augmented adjacency relL2 **0.707** — but per-subject γ_aug barely moves
(mean|Δ| 0.0035) so the loss shifts only 0.064 ± 0.052 on a level of 1.484.
**Material at the view/graph level, an order of magnitude below the mode mismatch at the loss level.**

## 20. Batch-size dependence — PASS

Loss tracks **log(B−1)** to within +0.0063…+0.0138 at every B ∈ {2,4,8,16,32,64} (raw
magnitudes are therefore **not** comparable across B, exactly as the B−1-negative structure
predicts). Top-1 stays at the 1/B chance baseline throughout. Everything finite at every B;
exp arguments confined to [2.357, 3.880] against the analytic bound [−5, 5]; minimum
denominator 1.69e+01. B=1 unreachable (guard, §4). *B=2 caveat: the discriminability z-score
is nan (one negative ⇒ undefined sd); read the raw cosines there.*

## 21. Temperature — PROJECT/AUTHORS choice, ranking-invariant

HEAD: `calc_loss(x, x_aug, temperature=0.2, sym=True)` — **identical signature to the authors'
release**; all three production call sites pass no temperature; no CLI flag exposes it.
**Ranking is τ-invariant** — per-row *and* global argsort identical at τ ∈ {0.1,0.2,0.5,1.0}
(s→s/τ is strictly monotone; confirmed, not assumed); top-1 unchanged at every τ. What τ does
change: loss magnitude (corr B=16 swings −7.018 → +1.725) and gradient magnitude (≈1/τ on
well-conditioned embeddings; sub-linear where the softmax saturates).
**TAU_02_STATUS = AUTHORS_RELEASE_SUPPORTED_NOT_PAPER_SPECIFIED.**

## 22. NT-Xent control — denominator changes magnitude, not direction

Exact relation: `dL/dS_GraSTI = dL/dS_NTXent / (1 − p_ii)` (max residual 1.5e-08) — a per-row
**positive rescale**. Gradient **direction essentially unchanged**: cosine **0.99849–1.00000**
across all five embedding sets. Gradient **magnitude** rescaled by 1/(1−p_ii), measured
**1.03× to 22.20×** — i.e. GraSTI's excluded positive *amplifies* the update for rows whose
positive already dominates. Diagnostic control only; the objective is **not** replaced.

## 23–25. Memory bank, supervised mode, projection vs downstream

- **Memory bank:** `calc_regloss(z, aug, memory)` exists in both our repo and the authors'
  release with **zero call sites** in either training loop → **DEAD/EXPERIMENTAL RELEASE
  CODE**. Paper Eq.20 uses in-batch negatives only. **NOT wired in.** A-GCL's memory bank
  belongs to A-GCL and is not evidence GraSTI requires one.
- **Supervised mode:** `contrastive_mode='supervised'` reaches `calc_loss_supervised(...,
  labels, ...)` **only in Phase 2**; Phase 1 (the adversary) always stays self-supervised.
  Default is `self_supervised`. Classified **ABIDE PROJECT ABLATION**, not paper GraSTI.
- **Projection vs downstream:** contrastive optimization consumes the **projected z**;
  evaluation reads `model.encoder` → `xpool`, the **pre-projection** embedding — the same
  split as the authors' release. Note `get_embeddings` uses `encoder.eval()` with
  `gamma_eval=1.0`, i.e. **exactly the untrained-BN eval regime** that §8/§17 show is
  degenerate at initialization — making that caveat load-bearing for downstream evaluation.
- **model.net (ToyNet/VIB) receives ZERO gradient from the contrastive loss** (0/10 tensors,
  ‖g‖=0.000000) — the expected consequence of `calc_loss` consuming only encoder→proj_head.
  Recorded so it is not misread later. **Ownership is Stage 8.**

## 26. Deferred to Stage 8 (recorded, NOT resolved)

Paper Eq. 21 reads `min_Ψ max_Ω I_R` — minimization on the TAE (Ψ) and maximization on the
projection head (Ω) — which conflicts with parts of the released training code. Optimizer
membership, phase signs, `model.net`/`view.net` ownership and dead parameter groups are all
**out of scope here**.

## FINAL BLOCK

```
PAPER_POSITIVE_PAIR                     = SAME_SUBJECT_ORIG_AUG (verified from Eq.20 directly)
CURRENT_POSITIVE_PAIR_ALIGNMENT         = PASS (off-diagonal movement exactly 0; orig_ID_order
                                          == aug_ID_order EXACTLY at B=8 and B=32)
CURRENT_NEGATIVE_SET                    = exactly B-1 in-batch AUGMENTED graphs j != i; own
                                          positive excluded; no duplicates; no out-of-batch;
                                          NO label filtering (bit-identical under y permute/zero)
CURRENT_LOSS_MATCHES_AUTHORS_RELEASE    = YES (bit-exact, 0.000e+00, 66 cases)
CURRENT_LOSS_MATCHES_PRINTED_EQ20_EXACTLY = NO (exactly two: tau=0.2 + symmetric extension;
                                          with both removed, equal to 2.4e-06)
CURRENT_LOSS_SYMMETRIC                  = YES (sym=True; authors' default, not printed Eq.20)
PAPER_EQ20_TEMPERATURE_SPECIFIED        = NO
CURRENT_TEMPERATURE                     = 0.2
NEGATIVE_LOSS_VALUES_VALID              = YES (range [-2/tau+ln(B-1), +2/tau+ln(B-1)];
                                          constructed -9.99905 vs analytic min -10.0)
ZERO_NORM_SAFETY                        = CONCERN (no zeros observed, min ||z|| 1.3979; but
                                          calc_loss is unguarded -> silent nan -> 6784 params
                                          nan-poisoned; structural exact-zero route exists)
PRE_PROJECTION_TOP1                     = 0.00105 (1/956; chance 0.001046; p=0.632)
POST_PROJECTION_TOP1                    = 0.00314 (3/956; chance 0.001046; p=0.080)
                                          -> projection PRESERVES; collapse is upstream
THETA_LOCAL_CONTRASTIVE_STEP            = PASS (loss down 23/24; 0/150 sign violations)
                                          CAVEAT: 83.5% non-specific in the eval geometry
GATE_LOGIT_ADVERSARIAL_DIRECTION        = PASS (6/6 raise the loss, up to +2.443 nats;
                                          ~97% via W_aug; redistribution, not net retention)
GATE_NOISE_EFFECT                       = small in the eval regime (loss sd 7.42e-04) but
                                          DOMINANT under train-mode BN (marginal 2.04x Beta)
LAMBDA_NOISE_EFFECT                     = dominant in the eval regime (loss sd 5.89e-02,
                                          6291x the gate); U-shaped Beta drives attention
                                          fully on/off at random; near-cancels in the loss
                                          under train-mode BN
STOCHASTIC_VARIANCE_EXCEEDS_BETWEEN_SUBJECT_SIGNAL = YES (0.1075 vs 0.0718-0.0814, 1.3-1.5x)
PAPER_INTENT_LAMBDA_ASYMMETRY           = CONCERN (real and large in the eval-BN regime:
                                          lambda*||X_atte||/||X_topo|| 0.0037 vs 14.57;
                                          but VERIFIER-QUALIFIED -- under train-mode BN the
                                          ratio collapses 37.02 -> 1.07 and the effect
                                          largely disappears. NOT established as a
                                          regime-independent statement.)
MATCHED_LAMBDA_RESCUES_POSITIVE_IDENTITY = YES (unconfounded arms: top-1 0.0117 -> 0.3125
                                          (p=6.7e-126) and 0.3555; margin 26x. The 99.61%
                                          arm is substantially a shared-X_atte artifact.)
PAPER_PRINTED_POSITIVE_COMPATIBILITY    = near-degenerate but identifiable positives in the
                                          eval regime (top-1 0.80, pos cos 0.9997) with a
                                          35.5x SUPPRESSED gradient; collapses to 0.020 under
                                          Beta sampling; indistinguishable from intent in Phase 2
PAPER_INTENT_POSITIVE_COMPATIBILITY     = structurally asymmetric positives, near-chance in
                                          the eval/Phase-1/embedding regime; fully compatible
                                          in Phase 2
PHASE1_PHASE2_REPRESENTATION_MATCH      = FAIL  ** STAGE7/8 BLOCKER **
                                          (loss 1.475 vs 0.310 at identical params, 4.76x;
                                          off-diagonal Spearman 0.350 vs a within-Phase-2 null
                                          of 0.9193; cross-mode |dL| 11.56x the within-Phase-2
                                          null at pristine BN and still 2.54x when BN is fully
                                          saturated; cos(grad_Phi) far from aligned with a
                                          negative tail in both independent runs;
                                          BatchNorm-dominated in 24/24 sweep rows.
                                          VERIFIER QUALIFICATION: the gap ATTENUATES and loses
                                          sign-consistency as BN running stats warm
                                          (-1.759 pristine -> -0.343 saturated), so the blocker
                                          is largest at epoch 0 and shrinks during training --
                                          it does not vanish.)
FRESH_PHASE_NOISE_MATERIAL              = YES at the view/graph level (gate corr 0.044,
                                          W_aug relL2 0.707); secondary at the loss level
                                          (0.064 on 1.484). AUTHORS_RELEASE_MATCH.
BATCH_SIZE_STABILITY                    = PASS (finite at every B; loss ~ log(B-1); B=1
                                          guarded at training.py:162)
TAU_02_STATUS                           = AUTHORS_RELEASE_SUPPORTED_NOT_PAPER_SPECIFIED
MEMORY_BANK_REQUIRED_BY_GRASTI          = NO (calc_regloss: 0 call sites in BOTH codebases)
SUPERVISED_CONTRASTIVE_STATUS           = PROJECT_ABLATION
STAGE7_CONFIRMED_LOSS_BUG               = NO
                                          (formula bit-exact to the release; pairing exact;
                                          negatives exact; gradient signs correct in 0/150
                                          violations; negative values mathematically valid)
STAGE7_CONFIRMED_VIEW_COMPATIBILITY_BUG = YES
                                          1. Phase-1/Phase-2 mode mismatch (BLOCKER):
                                             the two players optimize different surfaces.
                                          2. X_atte is STRUCTURALLY BLIND to the augmentation
                                             (trans_conv never receives edge_index/edge_weight)
                                             -> the "positive pair" shares a bit-identical
                                             component; cos(X_atte_orig, X_atte_aug) = 1.000000
                                          3. paper_intent lambda asymmetry (regime-qualified)
SAFE_TO_FREEZE_STAGE7                   = NO
                                          (the Phase-mode blocker must be resolved in Stage 8;
                                          zero-norm guard is an open CONCERN. Verification
                                          status: 2 of 3 adversarial verifiers complete
                                          (lambda-asymmetry, phase-mismatch) -- both applied
                                          above; the loss-correctness verifier was still
                                          running at write time, though its targets are the
                                          bit-exact/structural claims of sections 3-6 which
                                          rest on exact equalities.)
```

## Confirmed issues (evidence-backed) — no fixes applied

1. **Phase-1/Phase-2 mode mismatch — BLOCKER.** Stage 8 owns the correction (it is an
   optimizer/training-loop matter). Not fixed here.
2. **X_atte blind to the augmentation.** Structural: `TransConv.forward` has no edge input.
   Consequence: the augmented "view" differs from the original only in X_topo and in λ.
3. **paper_intent λ asymmetry** — real in the eval regime, **regime-qualified** by the verifier.
4. **Unguarded cosine denominator** in `calc_loss` — CONCERN; fail-loud guard proposed only.
5. **ToyNet VIB resamples in eval mode** (not gated on `self.training`) — no forward is
   reproducible without external seed control.
6. **`drop_last=False` + certain batch sizes** make the B=1 assert reachable mid-epoch.

Per spec §20/§26 **no production code was modified** and no fix is proposed beyond the
fail-loud zero-norm guard, because the decisive issues are training-loop/ownership matters
that Stage 8 owns.
