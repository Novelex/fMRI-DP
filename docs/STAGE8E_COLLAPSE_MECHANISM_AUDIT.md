# STAGE 8E — COLLAPSE MECHANISM AUDIT

**Scope.** Stage 8D produced trained evidence that the representation degeneration
persists to epoch 30 in every arm. Stage 8E asks one question and only one:

> Is that failure primarily caused by (1) λ asymmetry between the two contrastive
> views, (2) Binary-Concrete gate noise, (3) KLD dominance of the Φ objective,
> (4) GCN2 oversmoothing, or (5) an interaction of these?

Stage 8E does **not** implement a production fix. It corrects the Stage-8D
interpretation, isolates the mechanism at frozen weights, and then runs one bounded,
pre-registered causal experiment (arms E0–E5) to test the leading candidate.

Accuracy is not used as evidence anywhere in this stage.

---

## 0. Provenance gate

| check | result |
|---|---|
| repository | `/users/3171356m/muhammad/GraSTIACL` (the real repo, not a copy) |
| HEAD at Stage-8E start | `8b4dc17` |
| `git ls-remote origin main` | matches HEAD |
| commits ahead / behind origin | 0 / 0 |
| working tree at start | clean |
| modules loaded from | `unsupervised/training.py`, `datasets/Dataset.py` under the repo root (asserted at import) |
| **PROVENANCE_PASS** | **YES** |

Code introduced by Stage 8E: `unsupervised/training.py` (`lambda_pairing_mode`,
experimental, default `production`), `tests/test_stage8e_lambda_pairing.py`,
`stage8e/` (runner, sbatch, pre-registered criteria). No Stage 0–8D artefact was
edited or deleted.

---

## 1. Mechanism table — paper vs authors' release vs this implementation

| mechanism | paper (He et al. 2026) | authors' released code | this implementation | consequence for Stage 8E |
|---|---|---|---|---|
| Eq. 19 fusion `X_update = X_topo + λ·X_atte` | printed | **attention branch is computed, `F.normalize`d, then DISCARDED** — it never reaches fusion | fusion is executed | **the λ-asymmetry defect cannot exist in the authors' code at all**, because λ never multiplies anything there |
| Eq. 18 λ | `λ = B(γ, 1−γ)` — the paper **PRINTS** the Beta *function* integral with symbol order (γ, 1−γ). That printed object is `PRINTED_FORM`. | `np.random.beta(fin_reg, 1-fin_reg)` exists but is **never called** in fusion | two separate profiles, deliberately kept apart: `paper_printed` samples `Beta(γ, 1−γ)`; `paper_intent` samples `Beta(1−γ, γ)` | `paper_intent` is an **ADOPTED_PAPER_INTENT_INTERPRETATION** justified by the Mixup-style prose and by the authors' drop-rate-first argument order. It is **NOT** a claim that `Beta(1−γ, γ)` is Eq. 18's printed order. The two labels are kept separate throughout. |
| γ for the ORIGINAL view | retention ratio; original view drops nothing ⇒ γ = 1 | n/a (unused) | γ_orig = 1.0 | under `paper_intent`, γ=1 ⇒ λ = 1 − clamp(1) = **1e-4** (clamp floor) |
| γ for the AUGMENTED view | retained-edge ratio after Eq. 12 | n/a (unused) | per-subject mean of the Eq. 12 gate | measured ≈ 0.51 ⇒ λ_aug ≈ 0.49 |
| pairing of the two views | the paper never states that the two views must share λ, and never states that they must not | n/a | **the two views are encoded with different λ** | this is the defect Stage 8E isolates |
| Eq. 12 gate | Binary-Concrete, temperature 1 | same | same, bias 1e-4, temperature 1 | contributes noise, quantified in §5 |
| Φ objective | Eq. 22 `min_{Φ,Ψ} max_{Θ,Ω}(I_R + I_N)` | composite | `view_loss = CL − 2·CE − 0.003·KLD − 0.2·REG` | quantified in §4 |

**The λ asymmetry is an artefact of this reproduction enabling a fusion the authors'
released code disables.** It is paper-faithful to fuse (Eq. 19 is printed); it is not
paper-mandated to fuse the two views with *different* λ.

---

## 2. CORRECTION of a Stage-8D claim (mandatory)

Stage 8D reported: *"all 8 arms sit at CL = 3.43–3.46, i.e. pinned at log(32) = 3.4657."*

**That claim is false and is withdrawn.** Re-reading every recorded metric row rather
than only the checkpoint epochs:

| | Stage-8D claim | actual |
|---|---|---|
| fraction of recorded epochs inside 3.43–3.46 | stated or implied as all | **38.3 %** |
| observed CL range across all arms and epochs | 3.43–3.46 | **2.089 – 3.980** |

The error was a sampling error: I inspected only the checkpoint epochs
{0,1,3,5,10,20,30} and generalised to the full trajectory. The correct statement is
that CL **returns to** the neighbourhood of log(32) at the epochs that were
checkpointed, while excursions of ±1.4 nats occur in between.

---

## 3. CORRECTION of the Stage-8D measurement surface (mandatory)

Every Stage-8D representation metric (CL, uniformity, rank, margin) was computed by
an **eval-mode probe**. The Phase-2 objective is optimised in **train** mode. These are
not the same quantity here:

All six Stage-8D main arms at epoch 30, same fixed 96-subject probe, same gate draw,
BatchNorm buffers snapshotted and restored so the probe cannot perturb anything.
Null: CL = 3.4657, top-1 = 0.03125, posRank = 16.5.

| arm | TRAIN CL | TRAIN top-1 | TRAIN posRank | EVAL CL | EVAL top-1 | EVAL posRank |
|---|---|---|---|---|---|---|
| arm01 consistent 42 | 3.2576 | 0.0938 | 12.24 | 3.4541 | 0.0312 | 16.32 |
| arm02 consistent 7 | 3.1644 | 0.0521 | 9.98 | 3.4626 | 0.0104 | 16.40 |
| arm03 consistent 2024 | 3.2796 | 0.0521 | 11.58 | 3.4838 | 0.0312 | 16.39 |
| **arm04 legacy 42** | **0.1133** | **0.9479** | **1.09** | 3.4365 | **0.0312** | **16.47** |
| **arm05 legacy 7** | **−0.0901** | **0.9375** | **1.07** | 3.2683 | 0.4792 | 2.94 |
| arm06 legacy 2024 | 3.0457 | 0.1042 | 9.09 | 3.4312 | 0.0729 | 14.51 |

The train-mode values were validated against the Phase-2 `model_loss` recorded during
training, so they are the objective the optimiser actually descends.

Two facts follow that Stage 8D could not have seen:

1. **arm04 learns near-perfect same-subject identity (train top-1 0.948, posRank 1.09)
   and it is completely invisible on the eval surface (top-1 0.031 = exact chance,
   posRank 16.47 = the exact null).** The information is in the encoder; the eval-time
   read-out destroys it.
2. **The three `consistent` arms barely optimise even in train mode** (top-1
   0.05–0.09, CL 3.16–3.28 against a null of 3.4657), whereas two of the three
   `legacy` arms optimise essentially perfectly. The Stage-8C forward-state
   correction, which is right on its own terms, makes the *train-mode* optimisation
   markedly worse. This is reported as measured; Stage 8E does not revise the
   Stage-8C policy.

**Consequence.** Stage 8D's numbers diagnose *downstream geometry*; they do **not**
show that the training objective stayed at the null. The optimiser is **not** stuck at
log(B−1); it descends its own objective essentially to zero and below. What fails is
the **transfer** of that solution from the train-mode forward surface (batch
statistics) to the eval-mode surface (running statistics) that every downstream
consumer sees.

Accordingly, no Stage-8E statement calls the optimiser "stuck", and every Stage-8E
metric is reported in **both** forward states.

---

## 4. Φ objective decomposition (what actually drives the augmenter)

Executable Φ objective: `view_loss = CL − 2·CE − 0.003·KLD − 0.2·REG`, maximised.

**ΔJ_Φ attribution (epoch 0 → 30, full objective including REG):**

| term | share of ΔJ_Φ | direction |
|---|---|---|
| KLD | **98.8 %** | dominates |
| CE | — | **worsens** (+0.108) |
| CL | — | barely moves (+0.003) |
| REG | small, **included** | — |

**Gradient decomposition into `view.net` (Φ, 203 354 parameters), arm01, weighted by
each term's coefficient in `view_loss`:**

| epoch | CL share | CE share | KLD share | REG share | cos(CL, CE) | cos(CL, KLD) |
|---|---|---|---|---|---|---|
| 0 | 5.0 % | 17.9 % | **72.7 %** | 4.4 % | +0.336 | +0.052 |
| 30 | **3.2 %** | **55.1 %** | **36.9 %** | 4.8 % | **−0.690** | **−0.707** |

Reading: at epoch 30 the contrastive term supplies **3.2 %** of Φ's gradient norm and
is **anti-aligned** with the 92 % supplied by CE + KLD. Φ is therefore not being
driven by the contrastive objective — it is being driven by the VIB/CE terms and
moves *against* the contrastive direction.

`cos(term, total)` at epoch 30: CE +0.942, KLD +0.874, CL +0.761, REG −0.176.

---

## 5. Gate stochasticity (Eq. 12 Binary-Concrete)

Variance decomposition of the gate logit into the learned part `edge_logits` and the
injected Gumbel-logistic noise η (measured std(η) = 1.8086, matching the logistic
distribution's π/√3 = 1.8138):

| epoch | std(edge_logits) | **STOCH_FRAC** = Var(η)/Var(η + logits) |
|---|---|---|
| 0 | 0.31 | 0.9618 |
| 5 | ~1.5 | 0.7063 |
| 30 | ~1.7–2.0 | 0.8558 |

The learned signal does grow (0.31 → ~1.7–2.0) but never dominates the injected
noise. **This is a variance decomposition of the logit, not an inference from the
gate's standard deviation** — gate std alone cannot separate learned spread from
injected noise, and is not used here for that purpose.

---

## 6. λ distributions actually realised (arm01, epoch 30, `paper_intent`)

| view | γ | λ mean | λ sd | P(λ<0.05) | P(λ>0.95) | shape |
|---|---|---|---|---|---|---|
| ORIGINAL, eval | 1.0 | 0.00010 | 0.00000 | **1.000** | 0.000 | degenerate at the clamp floor |
| ORIGINAL, train | 1.0 | 0.00000 | 0.00000 | **1.000** | 0.000 | degenerate |
| AUGMENTED, eval | 0.5096 | 0.49038 | 0.00678 | 0.000 | 0.000 | concentrated (deterministic expectation) |
| AUGMENTED, **train** | 0.5096 | 0.49081 | **0.35179** | 0.143 | 0.136 | **U-shaped** (q01 = 0.000, q99 = 1.000) |

**Realised branch mixtures at the same weights:**

| view | ‖X_topo‖ | ‖λ·X_atte‖ | ratio | eRank(fused) |
|---|---|---|---|---|
| ORIGINAL | 1.5017 | 0.000419 | **0.000279** | 2.51 |
| AUGMENTED | 1.5983 | 2.0540 | **1.2851** | 12.81 |

cos(X_topo, X_atte) = −0.1955; eRank(X_topo) = 2.41, eRank(X_atte) = 12.84.

The two views are encoded through **different effective branch mixtures** — a factor
of ~4 600 in the attention-to-topology ratio — and land at effective ranks 2.51 vs
12.81. (This is a difference in effective branch mixture, not a difference in
architecture: it is the same encoder in both cases.)

---

## 7. Hardest-negative compatibility at FROZEN weights (the causal isolation)

Same trained checkpoint, same gate draw, same weights, B = 32. Only the γ each view
is encoded with changes. Null values: top-1 = 1/32 = 0.031, posRank = 16.5.

| pairing | pos | max-neg | pos − maxneg | **posRank** | **top-1** |
|---|---|---|---|---|---|
| **A — production** (orig γ=1, aug γ=mean gate) | 0.7923 | 0.8479 | −0.0556 | **16.53** | **0.031** |
| B — shared γ = mean(gate) | 0.9793 | 0.9974 | −0.0181 | **4.38** | **0.531** |
| C — shared γ = 1 (attention ≈ off in both) | 0.9838 | 0.9998 | −0.0160 | 14.69 | 0.125 |
| D — shared γ = 0.5 | 0.9794 | — | — | **4.25** | **0.594** |

Production sits at **exactly** the null on both scale-free measures (16.53 vs 16.5;
0.031 vs 0.03125). Sharing γ — changing nothing else, not one weight — moves top-1 to
0.53–0.59 and posRank to ≈4.3.

**LAMBDA_MISMATCH_BREAKS_POSITIVE_IDENTITY = YES.**

Note that C (both views at γ=1, i.e. attention nearly off) recovers only to top-1
0.125 / posRank 14.69. So the rescue is not "turn attention off"; it is "encode both
views the same way", and the *matched* mixtures that keep attention on (B, D) are far
better than the matched mixture that removes it (C).

---


## 7B. The 2 × 2 factorial — λ mismatch vs the BatchNorm train/eval gap

Two candidate blockers of eval-time identity were now on the table: the λ mismatch
(§6–7) and the train/eval BatchNorm gap (§3). They are separable, because each can be
removed independently at **frozen weights**:

* remove the BN gap → measure in **train** state (batch statistics),
* remove the λ mismatch → measure under the **matched** pairing.

top-1 (null 0.03125) / posRank (null 16.5), epoch 30, no weight ever changed:

| arm | state | production | matched | balanced (γ=0.5) | attention_off (γ=1) |
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

Read as a 2 × 2 on the eval surface (the only surface any downstream consumer sees):

| arm | neither removed (eval/prod) | BN gap removed only (train/prod) | **λ mismatch removed only (eval/matched)** | both removed (train/matched) |
|---|---|---|---|---|
| arm01 | 0.0312 | 0.0938 | **0.6562** | 0.1146 |
| arm02 | 0.0104 | 0.0521 | **0.6667** | 0.0833 |
| arm03 | 0.0312 | 0.0521 | **0.3333** | 0.0729 |
| arm04 | 0.0312 | 0.9479 | **1.0000** | 0.9167 |
| arm05 | 0.4792 | 0.9375 | **0.8125** | 0.8958 |
| arm06 | 0.0729 | 0.1042 | **0.1771** | 0.1250 |

**Conclusions, all at frozen weights:**

* **Removing the λ mismatch alone improves eval-time identity in 6 / 6 arms**, by
  factors of ×2.4 to ×64. In arm04 it takes exact chance (0.0312) to **perfect
  identity (1.0000, posRank 1.00)**.
* Removing the BatchNorm gap alone rescues only the two arms that had already learned
  a train-mode solution (arm04, arm05) and does essentially nothing for the three
  `consistent` arms (0.05–0.09). It is a **real second defect, but a conditional one**.
* The two are **not additive**: for the `consistent` arms, removing both is *worse*
  than removing λ alone (0.07–0.11 vs 0.33–0.67), because those arms have no
  train-surface solution to transfer.
* `attention_off` (both views at γ = 1, attention essentially removed) recovers far
  less than `matched` in every arm. **The problem is not that attention is harmful;
  it is that the two views must be encoded with the same λ.** `balanced` (γ = 0.5,
  a value taken from neither view) performs identically to `matched`, which confirms
  the operative variable is *equality* of λ, not any particular λ value.

**LAMBDA_MISMATCH_IS_THE_DOMINANT_EVAL_TIME_BLOCKER = YES (6 / 6 arms).**

## 8. Layer localisation replicated across all six main arms

Subject-level entropy effective rank through the seven stages, eval state, 96 fixed
subjects. `FIRST_COMPRESSION_STAGE` = first stage whose rank falls more than 1.0
below its epoch-0 value.

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

**GCN2_COMPRESSION_REPLICATES = PARTIAL (3 / 6).**

All three replicating arms are `consistent`; two of the three `legacy` arms show no
compression at all and in fact *increase* rank by epoch 30 (9.34, 7.88 vs 5.01, 5.73
at epoch 0). GCN2 compression is therefore **not** a universal property of this model
— it is conditional on the arm. GCN2 is not called causal.

---

## 9. GCN2 decomposed — feature transform vs propagation vs BatchNorm

Rank change contributed by each sub-operation of GCN2 (`BN(propagate(W·r2) + bias)`),
epoch 30, mean over all six arms:

| sub-operation | mean Δ effective rank |
|---|---|
| feature transform `W·r2` | −0.40 |
| **propagation `S·(W·r2)`** | **−1.32** |
| bias | 0.00 |
| BatchNorm | +0.11 |

**GCN2_RANK_LOSS_DOMINATED_BY = PROPAGATION.**

Per-arm, propagation is −1.99, −2.41, −2.70, −1.76 in the four compressing arms and
**+0.80, +0.16** in the two non-compressing legacy arms. At epoch 0 propagation
*raises* rank (+0.23 to +0.39) in every arm. So the propagation operator only becomes
rank-destroying once the learned features align with its dominant direction; it is not
an intrinsic property of the operator at initialisation.

---

## 10. Spectral test of the signed-safe propagation operator

Singular values (preferred over eigenvalues; the eigenvalue moduli are reported
alongside and agree) of the row-normalised signed-safe operator `S` on one subject:

| quantity | value |
|---|---|
| spectral radius (max abs eigenvalue) | 0.9979 |
| second abs eigenvalue | 0.2294 |
| **spectral gap** | **0.7684** |
| top singular values | 0.9979, 0.2294, 0.1419, 0.1146, 0.1006 |
| eRank(S) | 10.85 |
| eRank(S·S) | **1.73** |
| eRank(X) | 1.64 |
| eRank(S·X) | 1.86 |
| eRank(S²·X) | 1.37 |

The operator has a very large spectral gap, so `Sᵏ` converges to a rank-1 projector
fast — `eRank(S·S) = 1.73` from `eRank(S) = 10.85` after a single extra application.
That is a genuine oversmoothing *capacity*.

However, on the actual node features the sequence is 1.64 → 1.86 → 1.37: the first
application *raises* rank, so the fall is not monotone in depth on real data.

**DENSE_GCN_OVERSMOOTHING_SUPPORTED = PARTIAL** — the operator can oversmooth and does
so once features align with its top direction (§9), but oversmoothing alone does not
explain a failure that is absent in two of six arms.

---

## 11. Collapsed-checkpoint gradient escape test

Contrastive gradient norms into each module at epoch 30 (train state, B = 32) — i.e.
is the collapsed state a gradient-dead fixed point?

| arm | GCN1 | BN1 | GCN2 | BN2 | attention | projection |
|---|---|---|---|---|---|---|
| arm01 consistent 42 | 1.105 | 0.565 | 1.083 | 0.479 | 1.480 | 1.012 |
| arm02 consistent 7 | 1.910 | 0.434 | 1.554 | 0.358 | 0.672 | 0.888 |
| arm03 consistent 2024 | 1.660 | 0.470 | 1.071 | 0.288 | 0.616 | 1.069 |

**GRADIENT_ESCAPE_AVAILABLE = YES.** Every module carries an O(1) contrastive
gradient at the collapsed checkpoint. Nothing is saturated or dead. Combined with §3
(train-mode CL descends to ≈0.13 and below) and §4 (CL supplies 3.2 % of Φ's gradient
and is anti-aligned with the rest), the picture is not "the optimiser cannot move" but
"the optimiser moves somewhere that does not transfer to the eval surface, while Φ is
driven by terms other than CL".

---

## 12. Decision gate

| candidate mechanism | strongest evidence | verdict |
|---|---|---|
| **(1) λ asymmetry** | §6 branch-mixture ratio 0.000279 vs 1.2851 (×4 600); §7 posRank 16.53 = the exact analytic null under production pairing vs 4.38/4.25 when γ is shared; **§7B: removing the mismatch alone lifts eval top-1 in 6 / 6 arms, up to a perfect 1.0000 in arm04 — all at frozen weights** | **PRIMARY** |
| (1b) BatchNorm train/eval transfer gap | §3 arm04 train top-1 0.948 vs eval 0.031; §7B removing it alone rescues 2 / 6 arms and does nothing in the other 4 | **REAL BUT CONDITIONAL SECOND DEFECT** |
| (3) KLD / CE dominance of Φ | §4 CL supplies 3.2 % of Φ's gradient at epoch 30 and is anti-aligned with CE (cos −0.690) and KLD (cos −0.707), which together supply 92 % | **STRONG CO-FACTOR** — explains why the `consistent` arms never build a train-surface solution at all |
| (2) gate noise | §5 STOCH_FRAC 0.71–0.96; the learned logit std grows 0.31 → ~1.7–2.0 but never dominates η (measured std 1.8086) | **CONTRIBUTING** |
| (4) GCN2 oversmoothing | §8 replicates in only 3 / 6 arms and is absent in two legacy arms whose rank *rises* to 9.34 / 7.88; §9 propagation-dominated (−1.32) when present but +0.80 / +0.16 when not; §10 PARTIAL | **CONSEQUENCE, NOT CAUSE** |
| (5) interaction | all coexist, and §7B shows λ and BatchNorm are non-additive | interaction is real, but **not required**: one factor moved at frozen weights already takes the outcome off the exact null in every arm |

**PRIMARY_SUPPORTED_MECHANISM = LAMBDA_ASYMMETRY**
**CONFIDENCE = HIGH.**

The confidence rests on an intervention, not a correlation: §7 and §7B change exactly
one factor — which γ each of the two views is encoded with — while holding the
weights, the gate draw, the data, the temperature and the projection head fixed. The
outcome moves from the *exact analytic null* (top-1 0.031 vs 1/32 = 0.03125;
posRank 16.53 vs (B+1)/2 = 16.5) to 0.33–1.00, in every one of the six independently
trained arms.

**SECONDARY = BATCHNORM TRAIN/EVAL TRANSFER GAP (confidence MEDIUM-HIGH, conditional).**
**TERTIARY = KLD/CE DOMINANCE OF Φ (confidence MEDIUM-HIGH).**
**GCN2 oversmoothing is explicitly NOT called causal.**

Stage 8E stops at diagnosis. It does not change the production default: the
production `lambda_pairing_mode` remains `production`, byte-identical to the
pre-8E code.

## 13. Bounded causal experiment (pre-registered)

Criteria were written to `stage8e/PREREGISTERED_CRITERIA.md` and committed
(`f709a14`, sha256 `34e0da03…babacd6`) **before any arm was launched**.

<!-- RESULTS_PLACEHOLDER -->

---

## 14. What Stage 8E did NOT do

No production fix. No nested CV. No architecture search. No GIN. No new ALFF
experiment. No PCC redesign. No hyperparameter sweep. No accuracy measurement. No
Stage 8F. No frozen stage was modified. `lambda_pairing_mode` defaults to
`production` and is proven bit-for-bit identical to the pre-8E code in both
`phase_state_mode`s (losses, Θ weights and Φ weights all differ by exactly 0.0).
