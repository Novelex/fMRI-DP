# STAGE 8E — COLLAPSE MECHANISM AUDIT

Stage 8D produced trained evidence that the representation degeneration persists to
epoch 30 in every arm. Stage 8E answers one question:

> Is that failure primarily caused by (1) λ asymmetry between the two contrastive
> views, (2) Binary-Concrete gate noise, (3) KLD dominance of the Φ objective,
> (4) GCN2 oversmoothing, or (5) an interaction?

Stage 8E does **not** implement a production fix. Accuracy is secondary evidence only
and is never used to select an epoch or a configuration.

---

## 1. Provenance

| check | result |
|---|---|
| repository | `/users/3171356m/muhammad/GraSTIACL` (the real repo) |
| HEAD at Stage-8E start | `8b4dc17` |
| `git ls-remote origin main` | matches HEAD |
| commits ahead / behind origin | 0 / 0 |
| working tree at start | clean |
| modules asserted to load from repo root | `unsupervised/training.py`, `datasets/Dataset.py` |
| **PROVENANCE_PASS** | **YES** |

Stage-8E additions: `unsupervised/training.py` (`lambda_pairing_mode`, experimental,
default `production`), `tests/test_stage8e_lambda_pairing.py` (23/23 pass),
`stage8e/`. Stage-8C regression re-run: **18/18 pass**. No Stage 0–8D artefact was
edited or deleted; the only Stage-8D file touched is an appended amendment (§3).

`production` is proven **bit-for-bit** identical to the pre-8E code in *both*
`phase_state_mode`s — losses, Θ weights and Φ weights all differ by exactly 0.0.

---

## 2. Paper vs authors' release vs this implementation — λ semantics

| item | paper (He et al. 2026) | authors' released code | this implementation |
|---|---|---|---|
| Eq. 19 fusion `X_update = X_topo + λ·X_atte` | printed | attention branch is computed, `F.normalize`d, then **DISCARDED** — it never reaches fusion | fusion is executed |
| Eq. 18 λ | **`PRINTED_FORM`**: the paper prints the Beta *function* integral `B(γ, 1−γ)`, symbol order (γ, 1−γ) | `np.random.beta(fin_reg, 1-fin_reg)` exists but is **never called** in fusion | `paper_printed` → `Beta(γ, 1−γ)`; `paper_intent` → `Beta(1−γ, γ)` |
| γ, original view | retention ratio; the original view drops nothing ⇒ γ = 1 | unused | γ_orig = 1.0 |
| γ, augmented view | retained-edge ratio after Eq. 12 | unused | per-subject mean of the Eq. 12 gate |
| must the two views share λ? | **the paper does not say either way** | n/a | **they do not** — this is the defect isolated below |

**`ADOPTED_PAPER_INTENT_INTERPRETATION`** (kept strictly separate from `PRINTED_FORM`):
the active profile `paper_intent` samples `Beta(1−γ, γ)`, justified by the Mixup-style
prose and by the authors' drop-rate-first argument order. This is **not** a claim that
`Beta(1−γ, γ)` is Eq. 18's printed parameter order — the printed order is (γ, 1−γ) and
is preserved separately as the `paper_printed` profile.

Under `paper_intent`, γ = 1 ⇒ λ = 1 − clamp(1, 1e−4, 1−1e−4) = **1e−4** (the clamp
floor). Measured γ_aug ≈ 0.5096 ⇒ λ_aug ≈ 0.49.

**The λ asymmetry is an artefact of this reproduction enabling a fusion the authors'
released code disables.** Fusing is paper-faithful (Eq. 19 is printed); fusing the two
views with *different* λ is not paper-mandated.

---

## 3. Stage-8D numerical corrections

Both corrections are also appended to `docs/STAGE8D_TRAINED_TRAJECTORY_RESULTS.md`
under "Amendment after Stage 8E verification". No Stage-8D history was deleted.

### 3a. "All 8 arms CL = 3.43–3.46" — **WITHDRAWN, the claim is false**

Recomputed from every recorded epoch of all 8 arms (248 rows), not just the
checkpointed epochs:

| | claimed | actual |
|---|---|---|
| rows inside 3.43–3.46 | all | **95 / 248 = 38.3 %** |
| observed range | 3.43–3.46 | **2.089 – 3.980** |

Per-arm CL range and how often each arm sits below log(31) = 3.4340:

| arm | CL min | CL max | rows < log(31) | min CL excess vs log(32) |
|---|---|---|---|---|
| arm01 consistent 42 | 3.3958 | 3.5665 | 3 / 31 | −0.0699 |
| arm02 consistent 7 | 3.4175 | 3.5854 | 8 / 31 | −0.0483 |
| arm03 consistent 2024 | 3.3816 | 3.5972 | 4 / 31 | −0.0842 |
| arm04 legacy 42 | 2.5164 | 3.9803 | 15 / 31 | **−0.9493** |
| arm05 legacy 7 | **2.0891** | 3.7531 | 13 / 31 | **−1.3766** |
| arm06 legacy 2024 | 2.4495 | 3.6482 | 20 / 31 | **−1.0163** |
| arm07 consistent reg0 | 3.4122 | 3.5237 | 4 / 31 | −0.0535 |
| arm08 consistent reg1 | 3.4270 | 3.5157 | 1 / 31 | −0.0387 |

### 3b. Consistent reproducible collapse vs legacy transient separation

The two families behave **qualitatively differently** and Stage 8D conflated them:

* **`consistent` — reproducible collapse.** Best CL excess is only −0.084 nats, best
  margin +0.019, best top-1 0.083, uniformity never below −0.53. These arms never
  meaningfully leave the null regime. They do dip below log(31) in 20 / 155 rows, so
  `CONSISTENT_ALWAYS_AT_OR_ABOVE_LOG31 = NO` **literally**, but never by more than
  0.084 nats — the dips are noise, not separation.
* **`legacy` — transient separation that does not persist.** CL excess reaches −1.377,
  margin +0.440, top-1 0.604, uniformity −2.59. These arms genuinely escape the null
  regime and then return to it by epoch 30.

### 3c. The Stage-8D measurement surface was eval-mode only

Every Stage-8D representation metric came from an **eval-mode probe**. The Phase-2
objective is optimised in **train** mode. Therefore Stage-8D's metrics diagnose
downstream / evaluation geometry and **do not by themselves prove that the exact
train-mode Phase-2 objective remained at the null value.** §4 shows they in fact
did not.

---

## 4. Train-mode vs eval-mode contrastive loss

All six Stage-8D main arms at epoch 30, fixed 96-subject probe, same gate draw, BN
buffers snapshotted and restored so the probe cannot perturb anything.
Null: CL = 3.4657, top-1 = 0.03125, posRank = 16.5.

| arm | TRAIN CL | TRAIN top-1 | TRAIN posRank | EVAL CL | EVAL top-1 | EVAL posRank |
|---|---|---|---|---|---|---|
| arm01 consistent 42 | 3.2576 | 0.0938 | 12.24 | 3.4541 | 0.0312 | 16.32 |
| arm02 consistent 7 | 3.1644 | 0.0521 | 9.98 | 3.4626 | 0.0104 | 16.40 |
| arm03 consistent 2024 | 3.2796 | 0.0521 | 11.58 | 3.4838 | 0.0312 | 16.39 |
| **arm04 legacy 42** | **0.1133** | **0.9479** | **1.09** | 3.4365 | **0.0312** | **16.47** |
| **arm05 legacy 7** | **−0.0901** | **0.9375** | **1.07** | 3.2683 | 0.4792 | 2.94 |
| arm06 legacy 2024 | 3.0457 | 0.1042 | 9.09 | 3.4312 | 0.0729 | 14.51 |

Cross-checked against the Phase-2 `model_loss` recorded during training, so the train
column is the objective actually descended.

1. **arm04 learns near-perfect same-subject identity (train top-1 0.948, posRank 1.09)
   and it is entirely invisible on the eval surface** (top-1 0.0312 = the exact
   analytic chance rate 1/32; posRank 16.47 = the exact null 16.5).
2. The three `consistent` arms barely optimise even in train mode (top-1 0.05–0.09).
   The Stage-8C forward-state correction, right on its own terms, makes the
   *train-mode* optimisation markedly worse. Reported as measured; Stage 8E does not
   revise the Stage-8C policy.

**The optimiser is not stuck at the null.** No Stage-8E statement calls it stuck.

---

## 5. Exact Φ objective decomposition (REG included)

Executable Φ objective, maximised: `view_loss = CL − 2·CE − 0.003·KLD − 0.2·REG`.

**ΔJ_Φ attribution, epoch 0 → 30, full objective including REG:**

| term | share of ΔJ_Φ | direction |
|---|---|---|
| KLD | **98.8 %** | dominates |
| CE | — | **worsens** (+0.108) |
| CL | — | barely moves (+0.003) |
| REG | small — **included, not omitted** | — |

**Gradient decomposition into `view.net` (Φ, 203 354 parameters), arm01, each term
weighted by its coefficient in `view_loss`:**

| epoch | CL | CE | KLD | REG | cos(CL, CE) | cos(CL, KLD) | cos(CE, KLD) |
|---|---|---|---|---|---|---|---|
| 0 | 5.0 % | 17.9 % | **72.7 %** | 4.4 % | +0.336 | +0.052 | +0.301 |
| 30 | **3.2 %** | **55.1 %** | **36.9 %** | 4.8 % | **−0.690** | **−0.707** | +0.663 |

`cos(term, total)` at epoch 30: CE +0.942, KLD +0.874, CL +0.761, REG −0.176.

At epoch 30 the contrastive term supplies **3.2 %** of Φ's gradient norm and points
**against** the 92 % supplied by CE + KLD. Φ is not driven by the contrastive
objective.

---

## 6. Gate stochastic-vs-learned variance

Variance decomposition of the Eq. 12 gate **logit** into the learned part
`edge_logits` and the injected Gumbel-logistic noise η (measured std(η) = 1.8086 vs
the logistic distribution's π/√3 = 1.8138):

| epoch | std(edge_logits) | **STOCH_FRAC** = Var(η) / Var(η + logits) |
|---|---|---|
| 0 | 0.31 | **0.9618** |
| 5 | ~1.5 | **0.7063** |
| 30 | ~1.7–2.0 | **0.8558** |

The learned signal grows but never dominates the injected noise. This is a variance
decomposition of the logit — it is **not** inferred from the gate's standard
deviation, which cannot separate learned spread from injected noise.

---

## 7. λ distributions actually realised (arm01, epoch 30, `paper_intent`)

| view | γ | λ mean | λ sd | P(λ<0.05) | P(λ>0.95) | shape |
|---|---|---|---|---|---|---|
| ORIGINAL, eval | 1.0 | 0.00010 | 0.00000 | **1.000** | 0.000 | degenerate at the clamp floor |
| ORIGINAL, train | 1.0 | 0.00000 | 0.00000 | **1.000** | 0.000 | degenerate |
| AUGMENTED, eval | 0.5096 | 0.49038 | 0.00678 | 0.000 | 0.000 | concentrated (deterministic expectation) |
| AUGMENTED, **train** | 0.5096 | 0.49081 | **0.35179** | 0.143 | 0.136 | **U-shaped**, q01 = 0.000, q99 = 1.000 |

**Realised branch mixtures at the same weights:**

| view | ‖X_topo‖ | ‖λ·X_atte‖ | ratio | eRank(fused) |
|---|---|---|---|---|
| ORIGINAL | 1.5017 | 0.000419 | **0.000279** | 2.51 |
| AUGMENTED | 1.5983 | 2.0540 | **1.2851** | 12.81 |

cos(X_topo, X_atte) = −0.1955; eRank(X_topo) = 2.41; eRank(X_atte) = 12.84.

The two views run through **different effective branch mixtures** — a factor of
≈4 600 in the attention-to-topology ratio, landing at effective ranks 2.51 vs 12.81.
(Different effective branch mixtures of the *same* encoder — not different
architectures.)

---

## 8. Hardest-negative compatibility, and the λ / BatchNorm factorial

### 8a. Single checkpoint, four pairings

Same trained checkpoint, same gate draw, same weights, B = 32. Only the γ each view is
encoded with changes. Null: top-1 = 0.03125, posRank = 16.5.

| pairing | pos | max-neg | pos − maxneg | **posRank** | **top-1** |
|---|---|---|---|---|---|
| **A — production** (orig γ=1, aug γ=mean gate) | 0.7923 | 0.8479 | −0.0556 | **16.53** | **0.031** |
| B — shared γ = mean(gate) | 0.9793 | 0.9974 | −0.0181 | **4.38** | **0.531** |
| C — shared γ = 1 (topology only) | 0.9838 | 0.9998 | −0.0160 | 14.69 | 0.125 |
| D — shared γ = 0.5 | 0.9794 | — | — | **4.25** | **0.594** |

Production sits at **exactly** the analytic null on both scale-free measures.

### 8b. The 2 × 2 factorial — λ mismatch vs the BatchNorm train/eval gap

Two candidate blockers of eval-time identity, each removable independently at
**frozen weights**: remove the BN gap → measure in train state; remove the λ mismatch
→ measure under the matched pairing.

top-1 / posRank, epoch 30, no weight ever changed:

| arm | state | production | matched | balanced (γ=0.5) | topology-only (γ=1) |
|---|---|---|---|---|---|
| arm01 consistent 42 | **EVAL** | 0.0312 / 16.32 | **0.6562 / 5.23** | 0.6562 / 5.20 | 0.0938 / 14.16 |
| | TRAIN | 0.0938 / 12.24 | 0.1146 / 10.72 | 0.1146 / 10.70 | 0.1042 / 10.81 |
| arm02 consistent 7 | **EVAL** | 0.0104 / 16.40 | **0.6667 / 5.86** | 0.6562 / 5.82 | 0.0833 / 15.36 |
| | TRAIN | 0.0521 / 9.98 | 0.0833 / 11.76 | 0.0625 / 11.78 | 0.0729 / 12.48 |
| arm03 consistent 2024 | **EVAL** | 0.0312 / 16.39 | **0.3333 / 9.78** | 0.3333 / 9.70 | 0.0625 / 16.19 |
| | TRAIN | 0.0521 / 11.58 | 0.0729 / 11.38 | 0.0938 / 11.38 | 0.0729 / 13.67 |
| arm04 legacy 42 | **EVAL** | 0.0312 / 16.47 | **1.0000 / 1.00** | 1.0000 / 1.00 | 0.3125 / 2.42 |
| | TRAIN | 0.9479 / 1.09 | 0.9167 / 1.23 | 0.9167 / 1.21 | 0.9479 / 1.05 |
| arm05 legacy 7 | **EVAL** | 0.4792 / 2.94 | **0.8125 / 1.36** | 0.8125 / 1.35 | 0.6979 / 1.80 |
| | TRAIN | 0.9375 / 1.07 | 0.8958 / 1.10 | 0.8958 / 1.11 | 0.9271 / 1.08 |
| arm06 legacy 2024 | **EVAL** | 0.0729 / 14.51 | **0.1771 / 14.17** | 0.1771 / 14.19 | 0.1354 / 14.47 |
| | TRAIN | 0.1042 / 9.09 | 0.1250 / 9.72 | 0.1042 / 9.71 | 0.1146 / 9.00 |

As a 2 × 2 on the eval surface (the only surface any downstream consumer sees):

| arm | neither removed | BN gap removed only | **λ mismatch removed only** | both removed |
|---|---|---|---|---|
| arm01 | 0.0312 | 0.0938 | **0.6562** | 0.1146 |
| arm02 | 0.0104 | 0.0521 | **0.6667** | 0.0833 |
| arm03 | 0.0312 | 0.0521 | **0.3333** | 0.0729 |
| arm04 | 0.0312 | 0.9479 | **1.0000** | 0.9167 |
| arm05 | 0.4792 | 0.9375 | **0.8125** | 0.8958 |
| arm06 | 0.0729 | 0.1042 | **0.1771** | 0.1250 |

* **Removing the λ mismatch alone improves eval identity in 6 / 6 arms** (×2.4 to ×64);
  in arm04 it takes the exact null to **perfect identity (1.0000, posRank 1.00)**.
* Removing the BatchNorm gap alone rescues only the two arms that already had a
  train-mode solution and does nothing for the three `consistent` arms. A real second
  defect, but a **conditional** one.
* The two are **not additive**: for `consistent` arms, removing both is worse than
  removing λ alone — those arms have no train-surface solution to transfer.
* Topology-only (γ = 1 in both views) recovers far less than matched. **The problem is
  not that attention is harmful; it is that the two views must share λ.** `balanced`
  (γ = 0.5, a value from neither view) equals `matched`, confirming the operative
  variable is *equality* of λ, not any particular value.

**LAMBDA_MISMATCH_BREAKS_POSITIVE_IDENTITY = YES.**

---

## 9. Six-arm layer localisation

Subject-level entropy effective rank through seven stages, eval state, 96 fixed
subjects. `FIRST_COMPRESSION_STAGE` = first stage whose rank falls >1.0 below its
epoch-0 value.

| arm | ep | R0 | GCN1 | GCN2 | X_atte | FUSION | POOL | PROJ |
|---|---|---|---|---|---|---|---|---|
| arm01 consistent 42 | 0 | 2.97 | 5.52 | 5.48 | 11.57 | 5.49 | 5.49 | 8.18 |
| | 10 | 2.97 | 7.41 | 6.62 | 12.28 | 6.64 | 6.64 | 5.83 |
| | 30 | 2.97 | 4.96 | **2.73** | 8.51 | 2.76 | 2.76 | 2.77 |
| arm02 consistent 7 | 30 | 2.97 | 6.25 | **3.12** | 8.51 | 3.20 | 3.20 | 3.07 |
| arm03 consistent 2024 | 30 | 2.97 | 6.29 | **4.12** | 7.47 | 4.16 | 4.16 | 3.62 |
| arm04 legacy 42 | 30 | 2.97 | **3.05** | 1.37 | 12.00 | 1.47 | 1.47 | 1.25 |
| arm05 legacy 7 | 30 | 2.97 | 9.41 | 9.34 | 10.79 | 9.34 | 9.34 | 8.68 |
| arm06 legacy 2024 | 30 | 2.97 | 8.23 | 7.88 | 9.72 | 7.88 | 7.88 | 8.81 |

| arm | FIRST_COMPRESSION_STAGE |
|---|---|
| arm01 consistent 42 | GCN2 |
| arm02 consistent 7 | GCN2 |
| arm03 consistent 2024 | GCN2 |
| arm04 legacy 42 | **GCN1** |
| arm05 legacy 7 | **NONE** |
| arm06 legacy 2024 | **NONE** |

**GCN2_COMPRESSION_REPLICATES = PARTIAL (3 / 6).** All three replicating arms are
`consistent`; two `legacy` arms show no compression and in fact *increase* rank by
epoch 30 (9.34, 7.88 vs 5.01, 5.73 at epoch 0). GCN2 is **not** called causal.

---

## 10. GCN2 subcomponent localisation

Rank change per sub-operation of GCN2 (`BN(propagate(W·r2) + bias)`), epoch 30:

| sub-operation | mean Δ effective rank (6 arms) |
|---|---|
| feature transform `W·r2` | −0.40 |
| **propagation `S·(W·r2)`** | **−1.32** |
| bias | 0.00 |
| BatchNorm | +0.11 |

Per-arm propagation Δ: −1.99, −2.41, −2.70, −1.76 in the four compressing arms;
**+0.80, +0.16** in the two non-compressing legacy arms. At epoch 0 propagation
*raises* rank (+0.23 to +0.39) in every arm.

**GCN2_COMPRESSION_DOMINANT_SOURCE = PROPAGATION.** The propagation operator only
becomes rank-destroying once the learned features align with its dominant direction;
it is not intrinsically rank-destroying at initialisation.

---

## 11. Spectral analysis of the signed-safe propagation operator

Singular values (preferred; eigenvalue moduli agree) of the row-normalised
signed-safe operator `S`:

| quantity | value |
|---|---|
| spectral radius | 0.9979 |
| second |eigenvalue| | 0.2294 |
| **spectral gap** | **0.7684** |
| top singular values | 0.9979, 0.2294, 0.1419, 0.1146, 0.1006 |
| eRank(S) | 10.85 |
| eRank(S·S) | **1.73** |
| eRank(X) | 1.64 |
| eRank(S·X) | 1.86 |
| eRank(S²·X) | 1.37 |

A very large spectral gap means `Sᵏ` converges to a rank-1 projector quickly — a
genuine oversmoothing *capacity*. But on the actual node features the sequence is
1.64 → 1.86 → 1.37: the first application *raises* rank, so the fall is not monotone
in depth on real data.

**DENSE_GCN_OVERSMOOTHING_SUPPORTED = PARTIAL.**

---

## 12. Collapsed-state escape gradients

Contrastive gradient norms into each module at the epoch-30 checkpoint (train state,
B = 32):

| arm | GCN1 | BN1 | GCN2 | BN2 | attention | projection |
|---|---|---|---|---|---|---|
| arm01 consistent 42 | 1.105 | 0.565 | 1.083 | 0.479 | 1.480 | 1.012 |
| arm02 consistent 7 | 1.910 | 0.434 | 1.554 | 0.358 | 0.672 | 0.888 |
| arm03 consistent 2024 | 1.660 | 0.470 | 1.071 | 0.288 | 0.616 | 1.069 |

Every module carries an O(1) contrastive gradient. Nothing is saturated or dead.

**COLLAPSED_STATE_HAS_NEAR_ZERO_ESCAPE_GRADIENT = NO.**

---

## 13. Causal experimental arms

Exactly one factor changes per arm. No arm changes dropout, GCN depth, REG, learning
rate, ALFF, signedness, projection or temperature.

| arm | change vs E0 | everything else |
|---|---|---|
| E0 baseline | none (production) | frozen |
| E1 matched λ | `lambda_pairing_mode=matched` — both views use the augmented γ | frozen |
| E2 topology only | `lambda_pairing_mode=attention_off` — both views γ=1 ⇒ λ = 1e−4, the clamp floor (**not literally 0**) | frozen |
| E3 fixed half λ | `lambda_pairing_mode=balanced` — both views γ=0.5 ⇒ λ = 0.5 | frozen |
| E4 KLD 0.001 | `kld_lambda` 0.003 → 0.001 | frozen |
| E5 batch 128 | `batch_size` 32 → 128 | frozen |

All arms: `phase_state_mode=consistent`, seed 42, 30 epochs, `alff_new_z`,
`tae_profile=paper_intent`, `reg_lambda=0.2`, `ce_lambda=2.0`, lr 5e−4.

Every arm is probed on four surfaces per epoch — {eval, train} × {its own pairing,
the fixed `production` pairing} — because E1–E3 change the ruler as well as the
representation. E5 is probed at B = 128 against its own null (log 128 = 4.8520,
chance top-1 = 1/128, null posRank = 64.5).

---

## 14. Rescue criteria (pre-registered before any arm ran)

Written to `stage8e/PREREGISTERED_CRITERIA.md`, sha256 `34e0da03…babacd6`, committed
at `f709a14` — **before** the first arm launched.

An arm counts as a **representation rescue** only if, compared with E0:

1. sustained improvement in `CL_EXCESS` (= CL − log B) at **≥ 2 late checkpoints**;
2. positive-minus-**hardest**-negative improves materially;
3. positive rank improves;
4. uniformity stays meaningfully below 0 rather than collapsing to ≈0;
5. subject effective rank does not collapse to the E0 regime;
6. the effect is **not a one-checkpoint transient**.

Rescue is **never** declared from positive cosine alone.

**Secondary evidence** (never used for selection): the locked LinearSVC probe —
`LinearSVC(dual=False, fit_intercept=True, max_iter=10000)`,
`StratifiedKFold(5, shuffle=True, random_state=42)` with the **same fold objects
reused for every arm and epoch**, at epochs 0 / 10 / 30, balanced accuracy.
Raw baselines: FC ≈ 0.663, ALFF ≈ 0.591, FC+ALFF ≈ 0.657.

**Replication rule**, also pre-registered: if one arm rescues, replicate **only that
arm** at seeds 7 and 2024 for the same 30 epochs, then set `RESCUE_REPLICATES`.
If no arm rescues, run no further modifications.

---

## 15. Results

All eight arms reached epoch 30 and reported `COMPLETED`. No arm went non-finite.
Provenance: every arm records its git commit (`29f51a72`), the sha256 of
`unsupervised/training.py`, `datasets/Dataset.py` and `PREREGISTERED_CRITERIA.md`,
its exact command line and its host.

### 15a. Verdicts on the six governing criteria

Late = epochs ≥ 20 (11 checkpoints per arm). Surface = `eval_own` (the arm's own
pairing, eval state).

| arm | C1 CL_excess | C2 pos−hardest-neg | C3 pos rank | C4 uniformity | C5 subject rank | C6 not transient | **verdict** |
|---|---|---|---|---|---|---|---|
| E1 matched λ | ✅ 11/11 | ✅ +0.0824 | ✅ 1.90 vs 16.37 | ✅ −0.374 | ✅ 4.34 vs 1.79 | ✅ 11/11 | **RESCUE (6/6)** |
| E3 fixed half λ | ✅ 11/11 | ✅ +0.0722 | ✅ 3.10 vs 16.37 | ✅ −0.383 | ✅ 4.39 vs 1.79 | ✅ 11/11 | **RESCUE (6/6)** |
| E2 topology only | ❌ 0/11 | ✅ +0.0514 | ✅ 14.28 vs 16.37 | ❌ −0.0001 | ❌ 2.06 vs 1.79 | ✅ 11/11 | PARTIAL (3/6) |
| E4 KLD 0.001 | ❌ 0/11 | ❌ +0.0138 | ❌ 16.43 vs 16.37 | ❌ −0.0008 | ❌ 2.04 | ❌ 0/11 | **NO_RESCUE (0/6)** |
| E5 batch 128 | ❌ 0/11 | ✅ +0.0400 | ❌ 55.84 vs null 64.5 | ❌ −0.0163 | ✅ 7.65 | ❌ 0/11 | **NO_RESCUE (2/6)** |
| R1 seed 7, matched | ✅ 11/11 | ✅ +0.0435 | ✅ 7.43 vs 16.37 | ✅ −0.261 | ✅ 5.33 | ✅ 11/11 | **RESCUE (6/6)** |
| R2 seed 2024, matched | ✅ 11/11 | ✅ +0.0755 | ✅ 1.86 vs 16.37 | ✅ −0.424 | ✅ 4.35 | ✅ 11/11 | **RESCUE (6/6)** |

### 15b. Trajectories (eval, own pairing). Null: CL 3.4657, top-1 0.0312, posRank 16.5

| arm | metric | ep 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|
| **E0 baseline** | CL | 3.440 | 3.432 | 3.441 | 3.534 | 3.463 | 3.489 | 3.458 |
| | top-1 | 0.073 | 0.031 | 0.042 | 0.031 | 0.031 | 0.031 | 0.031 |
| | posRank | 13.28 | 14.91 | 15.92 | 16.50 | 16.27 | 16.48 | 16.48 |
| | uniformity | −0.530 | −0.018 | −0.014 | −0.000 | −0.001 | −0.002 | −0.000 |
| | subj rank | 8.18 | 5.81 | 4.53 | 1.65 | 2.62 | 1.64 | 2.46 |
| **E1 matched λ** | CL | 3.417 | 3.423 | 3.398 | 3.224 | 3.097 | 3.055 | **2.840** |
| | top-1 | 1.000 | 0.375 | 0.500 | 0.969 | 0.875 | 0.979 | **0.969** |
| | posRank | 1.00 | 12.67 | 8.74 | 1.62 | 2.79 | 1.14 | **1.32** |
| | uniformity | −0.014 | −0.028 | −0.035 | −0.175 | −0.299 | −0.300 | **−0.504** |
| | subj rank | 9.97 | 12.09 | 10.10 | 6.22 | 4.80 | 4.54 | 4.01 |
| **E3 fixed half λ** | CL | 3.417 | 3.421 | 3.397 | 3.245 | 3.199 | 3.115 | **2.834** |
| | top-1 | 1.000 | 0.302 | 0.469 | 0.562 | 0.656 | 0.750 | **0.896** |
| | posRank | 1.00 | 13.16 | 10.89 | 7.59 | 6.03 | 5.00 | **1.77** |
| **R2 seed 2024** | CL | 3.414 | 3.457 | 3.420 | 3.393 | 3.249 | 2.919 | **2.698** |
| | top-1 | 1.000 | 0.292 | 0.229 | 0.615 | 0.844 | 0.958 | **1.000** |
| | posRank | 1.00 | 13.48 | 13.75 | 7.58 | 3.53 | 1.32 | **1.00** |
| **R1 seed 7** | CL | 3.387 | 3.453 | 3.431 | 3.438 | 3.297 | 3.138 | 3.177 |
| | top-1 | 0.990 | 0.521 | 0.229 | 0.354 | 0.542 | 0.448 | 0.271 |
| | posRank | 1.01 | 9.84 | 13.66 | 11.85 | 8.49 | 7.99 | 10.59 |

Two things must be said plainly about these trajectories:

* **The epoch-0 value is not evidence of learning.** Under matched λ, top-1 is already
  1.000 at random initialisation — the untrained encoder trivially maps the same
  subject to nearby points when both views use the same branch mixture. Training then
  *destroys* it (0.29–0.52 by epoch 5–10) and rebuilds it.
* **The late-epoch state is nevertheless genuinely better than epoch 0**, and that is
  what C1 and C4 measure: E1's CL excess goes from −0.049 (ep 0) to **−0.626** (ep 30)
  and its uniformity from −0.014 to **−0.504**. Under production pairing the same
  training produces CL excess −0.008 and uniformity −0.000. So training adds real
  structure under matched λ and adds essentially nothing under mismatched λ.
* R1 (seed 7) is the weakest rescue and is not monotone (top-1 peaks 0.542 at ep 20,
  falls to 0.271 at ep 30). It still satisfies all six criteria, but the seed-to-seed
  spread is large and is reported rather than averaged away.

### 15c. Non-rescuing arms

| arm | CL excess (late) | top-1 (null) | posRank (null) | uniformity | subj rank |
|---|---|---|---|---|---|
| E2 topology only | +0.0231 | 0.1259 (0.0312) | 14.28 (16.5) | −0.0001 | 2.06 |
| E4 KLD 0.001 | −0.0058 | 0.0341 (0.0312) | 16.43 (16.5) | −0.0008 | 2.04 |
| E5 batch 128 | −0.0249 | 0.0111 (0.0078) | 55.84 (64.5) | −0.0163 | 7.65 |

**E4 is the direct test of the KLD-dominance hypothesis and it fails 0/6.** Reducing
`kld_lambda` by 3× leaves every identity measure at the null. **E5 is the direct test
of the batch-size hypothesis and it fails 2/6** — a larger batch raises subject rank
(7.65) but leaves posRank at 55.84 against its own null of 64.5.

**E2 is the crucial control.** Removing attention from both views (γ=1 ⇒ λ=1e−4)
improves the hardest-negative margin and positive rank a little but leaves CL excess
*positive* (+0.023, i.e. worse than the null) and uniformity at −0.0001. So the rescue
is **not** "switch attention off"; it is "encode both views with the same λ". E3
confirms this from the other side: γ = 0.5, a value belonging to neither view, rescues
just as well as matching the augmented view's own γ.

### 15d. The common-yardstick check (`eval_prod`)

Every arm was also probed under the fixed `production` pairing, so that arms which
change the encoding cannot appear to improve merely by changing the ruler:

| arm | eval_own verdict | eval_prod verdict | classification |
|---|---|---|---|
| E1 matched λ | RESCUE (6/6) | **NO_RESCUE (0/6)** | `RESCUED_ENCODING_PAIRING` |
| E3 fixed half λ | RESCUE (6/6) | **NO_RESCUE (0/6)** | `RESCUED_ENCODING_PAIRING` |
| R1 / R2 | RESCUE (6/6) | NO_RESCUE (1/6) | `RESCUED_ENCODING_PAIRING` |

**This is a material limitation and it is not softened here.** Training with matched λ
does not produce an encoder that survives being *read out* with mismatched λ. The
rescue lies in encoding the two views consistently — which is how the model would be
used — not in a fundamentally more robust encoder. The pre-declared label for this
outcome is `RESCUED_ENCODING_PAIRING`, not `RESCUED_REPRESENTATION`.

### 15e. Secondary evidence — locked LinearSVC probe

Balanced accuracy, `LinearSVC(dual=False, fit_intercept=True, max_iter=10000)`,
`StratifiedKFold(5, shuffle=True, random_state=42)` with identical fold objects
across every arm and epoch. Never used to select anything.

| arm | epoch 0 | epoch 10 | epoch 30 |
|---|---|---|---|
| E0 baseline | 0.4760 | 0.4967 | 0.5014 |
| E1 matched λ | 0.4760 | 0.4839 | **0.5398** |
| E2 topology only | 0.4760 | 0.4954 | 0.5104 |
| E3 fixed half λ | 0.4760 | 0.5007 | 0.5275 |
| E4 KLD 0.001 | 0.4760 | 0.5174 | 0.5187 |
| E5 batch 128 | 0.4760 | 0.4993 | 0.5171 |
| R1 seed 7 | 0.4922 | 0.4943 | 0.5103 |
| R2 seed 2024 | 0.4965 | 0.5345 | 0.5095 |

Reference baselines: **FC ≈ 0.663, ALFF ≈ 0.591, FC+ALFF ≈ 0.657** (fold sd ≈ 0.02–0.04).

**The secondary evidence does not support a classification benefit.** E1's 0.5398 is
the best value but sits about one fold-sd above E0's 0.5014, and **neither replication
seed reproduces it** (0.5103, 0.5095). Every arm remains far below the raw FC baseline.
**Same-subject identity is rescued; diagnostic class information is not.**

---

## 16. Final interpretation

**The primary mechanism is λ asymmetry between the two contrastive views, and it is
confirmed causally, not just correlationally.**

Three independent lines of evidence agree:

1. **Frozen-weight intervention (§8).** Changing only which γ each view is encoded
   with — no training, no weight change, no data change — moves eval-time identity
   from the *exact analytic null* (top-1 0.0312 vs 1/32 = 0.03125; posRank 16.53 vs
   (B+1)/2 = 16.5) to 0.33–1.00, in **6 / 6** independently trained Stage-8D arms. In
   arm04 it reaches perfect identity (top-1 1.0000, posRank 1.00).
2. **Trained causal arms (§15).** E1 (matched λ) and E3 (fixed λ = 0.5) satisfy all
   six governing criteria; the two competing hypotheses fail their own direct tests —
   **E4 (KLD ÷3) scores 0/6** and **E5 (batch ×4) scores 2/6**.
3. **Replication (§15a).** Both pre-registered seeds, 7 and 2024, also score 6/6.
   **RESCUE_REPLICATES_MULTI_SEED = YES**, with an honest caveat that seed 7 is the
   weakest and non-monotone.

**Why it happens.** This reproduction executes Eq. 19's fusion, which the authors'
released code disables by discarding the attention branch. Under `paper_intent`,
γ_orig = 1 gives λ_orig = 1e−4 while γ_aug ≈ 0.51 gives λ_aug ≈ 0.49. The positive
pair is therefore encoded through branch mixtures whose attention-to-topology ratios
are 0.000279 and 1.2851 — a factor of ≈4 600 — landing at effective ranks 2.51 and
12.81. The InfoNCE positive is not a view of the same object in the same space, so it
cannot beat the hardest negative, and the loss sits at log B by construction. The
paper never states that the two views must share λ; it also never states that they
must not. **Fusing is paper-faithful; fusing the two views with different λ is a
reproduction choice, and it is the one that fails.**

**What the other candidates actually are.**

* **KLD/CE dominance of Φ is real but is not the primary cause.** At epoch 30 the
  contrastive term supplies 3.2 % of Φ's gradient and is anti-aligned with CE
  (cos −0.690) and KLD (cos −0.707). That explains why the augmenter is not steered by
  the contrastive objective — but E4 tested it directly and scored 0/6.
* **Gate noise is a contributor, not the cause.** STOCH_FRAC stays at 0.71–0.96; the
  learned logit never dominates η. No arm isolated it, so it is not ruled in or out as
  a *sufficient* cause; it is ruled out as the *primary* one by E1/E3 rescuing without
  touching it.
* **GCN2 oversmoothing is a consequence, not a cause.** It replicates in only 3 / 6
  arms — all `consistent` — while two `legacy` arms *increase* rank to 9.34 and 7.88.
  Where it occurs, propagation dominates (−1.32) over the feature transform (−0.40)
  and BatchNorm (+0.11), and the operator's spectral gap (0.7684) gives it genuine
  oversmoothing capacity. But a mechanism absent in a third of the arms cannot be the
  cause of a failure present in all of them.
* **A second, independent defect exists: the BatchNorm train/eval transfer gap.**
  arm04 has train top-1 0.948 and eval top-1 0.031. Removing that gap alone rescues
  only the two arms that already had a train-mode solution and does nothing for the
  three `consistent` arms. It is real, conditional, and **not** addressed by Stage 8E.

**What was NOT achieved, stated plainly.**

* The rescue is `RESCUED_ENCODING_PAIRING`, not `RESCUED_REPRESENTATION`: matched-λ
  training does not yield an encoder robust to mismatched read-out (0/6 on `eval_prod`).
* **Classification does not improve.** The best arm reaches balanced accuracy 0.5398
  against raw baselines of FC 0.663 / ALFF 0.591 / FC+ALFF 0.657, and neither
  replication seed reproduces even that. Rescuing same-subject identity did **not**
  rescue diagnostic signal. Any expectation that fixing λ will by itself raise accuracy
  is not supported by this evidence.
* The three `consistent` arms' failure to optimise even in *train* mode (§4) is
  measured but not explained. Stage 8E does not revise the Stage-8C policy.

**Production is unchanged.** `lambda_pairing_mode` defaults to `production` and is
proven bit-for-bit identical to the pre-8E code in both `phase_state_mode`s. Stage 8E
was authorised to diagnose, not to fix, and it has not fixed anything. The evidence
supports a specific, narrow change — encode both contrastive views with the same γ —
but making that change is a separate decision that requires its own authorisation.

---

## Verdict block

```text
PROVENANCE_PASS =
YES

STAGE8D_ALL_ARMS_3_43_TO_3_46 =
NO          (38.3% of 248 rows; true range 2.089-3.980)

CONSISTENT_ALWAYS_AT_OR_ABOVE_LOG31 =
NO          (20/155 rows dip below log 31, but never by more than 0.084 nats)

LEGACY_TRANSIENTLY_ESCAPES_NULL_REGIME =
YES         (CL excess to -1.377, margin +0.440, top1 0.604, uniformity -2.59)

TRAIN_CL_STUCK_AT_NULL =
NO          (arm04 0.1133, arm05 -0.0901; E0 train CL excess -0.317)

EVAL_CL_STUCK_AT_NULL =
PARTIAL     (E0 excess -0.0021 = stuck; arm05 3.2683 and E1 2.840 = not stuck)

TRAIN_EVAL_CL_BEHAVIOR_DIFFERENT =
YES         (arm04 train top1 0.948 / eval top1 0.031)

KLD_EXPLAINS_MOST_OF_J_DESCENT =
YES         (98.8% of dJ_Phi, REG included)

CE_IMPROVES_UNDER_PHI_OPTIMIZATION =
NO          (CE worsens by +0.108)

CL_ADVERSARIAL_COMPONENT_CHANGES_MATERIALLY =
NO          (+0.003 over epochs 0->30; 3.2% of Phi's gradient at epoch 30)

GATE_STOCHASTIC_FRACTION_E0 =
0.9618
GATE_STOCHASTIC_FRACTION_E5 =
0.7063
GATE_STOCHASTIC_FRACTION_E30 =
0.8558

GATE_IS_MOSTLY_NOISE_AT_E30 =
YES

ORIGINAL_VIEW_ATTENTION_EFFECTIVELY_SUPPRESSED =
YES         (lambda_orig = 1e-4; ||lam*X_atte||/||X_topo|| = 0.000279)

AUGMENTED_LAMBDA_U_SHAPED =
YES         (train mode: sd 0.352, q01 0.000, q99 1.000, P(<.05) 0.143, P(>.95) 0.136)

LAMBDA_MISMATCH_BREAKS_POSITIVE_IDENTITY =
YES         (frozen weights: posRank 16.53 vs null 16.5 -> 4.38 when gamma is shared;
             improves eval identity in 6/6 Stage-8D arms)

GCN2_COMPRESSION_REPLICATES =
PARTIAL     (3/6 arms, all 'consistent'; two legacy arms INCREASE rank)

GCN2_COMPRESSION_DOMINANT_SOURCE =
PROPAGATION (-1.32 vs feature transform -0.40, BatchNorm +0.11)

DENSE_GCN_OVERSMOOTHING_SUPPORTED =
PARTIAL     (spectral gap 0.7684, eRank(S) 10.85 -> eRank(S@S) 1.73, but
             non-monotone on real features: 1.64 -> 1.86 -> 1.37)

COLLAPSED_STATE_HAS_NEAR_ZERO_ESCAPE_GRADIENT =
NO          (O(1) contrastive gradient in every module: 0.29-1.91)

PRIMARY_SUPPORTED_MECHANISM =
LAMBDA_MISMATCH

PRIMARY_MECHANISM_CONFIDENCE =
HIGH

E0_BASELINE =
Pinned at the null. Late-epoch eval: CL 3.4636 (excess -0.0021 vs log 32), top1 0.0360
(chance 0.03125), posRank 16.37 (null 16.5), pos-minus-hardest-neg -0.0873,
uniformity -0.0019, subject effective rank 1.79. Late-epoch TRAIN: CL 3.1488
(excess -0.3169), top1 0.0862, posRank 9.95, uniformity -1.5265, rank 15.35.

E1_MATCHED_LAMBDA =
RESCUE      (6/6; top1 0.9157, posRank 1.90, CL excess -0.473, uniformity -0.374,
             subject rank 4.34, sustained at 11/11 late checkpoints)

E2_TOPOLOGY_ONLY =
NO_RESCUE   (3/6 -- PARTIAL. CL excess remains POSITIVE at +0.023; uniformity -0.0001;
             subject rank 2.06. This is the control that rules out "turn attention off")

E3_FIXED_HALF_LAMBDA =
RESCUE      (6/6; top1 0.8172, posRank 3.10, CL excess -0.462, uniformity -0.383)

E4_KLD_0001 =
NO_RESCUE   (0/6 -- the direct test of the KLD-dominance hypothesis)

E5_BATCH128 =
NO_RESCUE   (2/6; posRank 55.84 against its own null of 64.5)

BEST_MECHANISTIC_ARM =
E1_matched_lambda

RESCUE_REPLICATES_MULTI_SEED =
YES         (seed 7 = 6/6, seed 2024 = 6/6; seed 7 is the weakest and non-monotone)

SAFE_TO_CHANGE_PRODUCTION =
NO

SAFE_TO_BEGIN_FULL_NESTED_CV =
NO

NEED_RANDOM_ARCHITECTURE_SEARCH =
NO
```

### Why SAFE_TO_CHANGE_PRODUCTION = NO despite a replicated 6/6 rescue

Three reasons, none of which is a doubt about the mechanism:

1. Stage 8E was authorised to diagnose, not to fix. Implementing the change is a
   separate decision.
2. The rescue is `RESCUED_ENCODING_PAIRING`, not `RESCUED_REPRESENTATION` — every
   rescuing arm scores 0–1/6 on the fixed `production` yardstick.
3. **Classification does not improve.** Best arm 0.5398 balanced accuracy vs E0 0.5014,
   not reproduced by either replication seed (0.5103, 0.5095), against raw baselines of
   FC 0.663 / ALFF 0.591 / FC+ALFF 0.657.

A production change would also have to decide what the *original* view's γ should be,
which is a modelling decision the paper does not settle — and it should be taken
together with the second, independent defect Stage 8E identified but did not address:
the BatchNorm train/eval transfer gap.
