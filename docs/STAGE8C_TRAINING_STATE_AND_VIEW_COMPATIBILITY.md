# STAGE 8C — TRAINING-STATE POLICY AND VIEW COMPATIBILITY

Date: 2026-08-19 · Stages 1–6 frozen · Stage 7 complete · Stage 8A superseded where corrected
by 8B · Stage 8B accepted at `e656d63` · **First stage since Stage 6E to change production
code.** No accuracy, no CV, no hyperparameter search, no reopened ALFF/PCC/Mij/TAE/loss.

---

## 1. Provenance

```
pwd / toplevel   /users/3171356m/muhammad/GraSTIACL        branch  main
HEAD             e656d63…  ==  origin/main  (Stage 8B present on both)
status           clean, 0 modified tracked files
```
All seven Stage-8B source SHA256s **unchanged**; `data_alff_new_z.pt` cache hash
`392b3b18…9e414e` unchanged. Every production module asserted to resolve under the repo
(harness raises otherwise). Because nothing moved, the expensive Stage-8B content audit was
not repeated. `RUNTIME_REPO_CORRECT = YES`.

## 2. Eq. 15 / Θ-side note (one cheap check; Stage 4 NOT reopened, `mij_source` unchanged)

With M_ij fixed by raw ALFF, `I_N = 2·CE + 0.003·KLD` has **no gradient path into Θ at all**:

| target | ∂I_N/∂· | tensors with a path |
|---|---|---|
| `model.encoder` | **0.000000e+00** | 0/18 (autograd returns `None`) |
| `model.proj_head` | **0.000000e+00** | 0/4 |
| `model.net` | **0.000000e+00** | 0/10 |
| *control* `view.net` | 7.225045e-01 | 10/10 |
| *control* ∂CL/∂`model.encoder` | 5.579412e+00 | — (machinery works) |

Finite differences confirm the zero is exact: a scale-1.0 Gaussian on **every** model parameter
leaves CE and KLD **bit-identical** while moving CL by −2.4e−02. M_ij is provably inert to
learned embeddings (`edge_prod` bitwise equals `Σ x[:,:3]_src·x[:,:3]_dst`; max|Δ| = 0 exactly
under perturbations that move `edge_logits` by 1.3e+05).

**Crucially this is not caused by our `mij_source='alff'` choice.** Rebuilding M_ij with the
authors' released semantics (view-encoder `node_emb` dot, target `.detach()`ed) *also* gives
∂I_N/∂model = 0. Only a **double counterfactual** — M_ij from the *model* encoder **and**
dropping the production `.detach()` — creates any Θ path, and it is tiny (2.69e−03, 0.37 % of
the concurrent view-net norm).

```
EQ15_THETA_SIDE_WITH_ALFF_MIJ = ONE_SIDED
EQ15_MIJ_PAPER_CONTRADICTION  = YES   (internal to the paper: Eq.4 defines M_ij over
    parameter-free raw ALFF while Eq.15 asks for max_Theta over I_N. mij_source='alff'
    INHERITS the contradiction rather than creating it. Recorded, NOT fixed here.)
```

## 3. Cross-phase decomposition (Problem B) — reproduced, then factorially decomposed

24 paired repetitions, identical weights/subjects/PCC/ALFF/edge-logits/gate, BN snapshotted and
restored around every call (max weight drift over the whole run = **0**):

| metric | Phase-1 semantics | Phase-2 semantics | gap |
|---|---|---|---|
| CL | 1.9572 | 0.2875 | **−1.6698** |
| margin | 0.00475 | 0.4110 | +0.4062 |
| top-1 (chance 0.125) | 0.208 | 0.828 | +0.620 |
| rel L2 of the **original** representation | — | — | **37.65** |

Within-phase SD (independent redraws): 0.00137 (Phase 1), 0.179 (Phase 2) ⇒
`CROSS_TO_WITHIN_RATIO` = **9.33** against the noisier phase, 1219 against Phase 1. Zero
distribution overlap across 192 draws/phase. `CURRENT_PHASES_SAME_FORWARD_DISTRIBUTION = NO`.

**2⁴ factorial** (bn × λ × dropout × gate-realization, 10 seeds/cell, 3 batch compositions):
BatchNorm carries **92.6 %** of the CL effect mass and **100 %** of the original-representation
mass; λ, dropout and gate-realization are minor (gate-realization < 0.6 % everywhere).
Dropout owns the attention branch (99.9 % of its effect mass), BN owns the topology branch —
clean separation, consistent with `trans_conv.bns` being LayerNorm.

Null controls pass: **CE, REG, KLD and γ_aug have exactly zero cross-phase gap**, and
`ΔJ = −ΔCL` exactly, so the entire cross-phase movement of Φ's objective is the contrastive term.

⚠ **Verifier scoping (accepted):** BN dominance is a *state-scoped* claim. With buffers
populated by W = 60/200 forward-only passes, BN's share of several metrics collapses
(positive cosine 74.8 % → 0.9 % → 45.9 %) and the CL gap decays −1.70 → −1.46 (W=60) → −1.14
(W=200) → ~69 % surviving at W=400. So the gap **shrinks but does not vanish** with calibration;
the first agent's "85 % survives" was right only at W=60.

## 4. Within-view decomposition (Problem A) — kept separate from §3

128 subjects, one gate realization reused, audited under **both** `bn='eval'` and `bn='train'`:

- The Stage-7 norm gap **replicates** (‖G′‖/‖G‖ = 16.62 at h, 25.09 at z) **but is
  regime-conditional**: 1.07 under `bn='train'`, and it collapses to 1.06 after a *single*
  forward-only warm pass. The 16–25× figure is an artifact of running stats still at their
  init values, not an architectural property.
- `VIEW_NORM_GAP_PRIMARY_CAUSE = LAMBDA`, dominant in **every** regime (so not a BN artifact),
  with GRAPH a genuine secondary contributor once BN matures and interaction 7–18 %. A verifier
  strengthened this with an exactly-additive split (h is affine in λ; residual ~2e−07).
- Structural driver: `TransConv.forward(x, batch)` takes no edge weights, so **X_atte is
  bit-identical between G and G′** — the PCC×gate perturbation reaches only the topology branch,
  while γ_orig = 1 pins λ_orig at the 1e−4 clamp floor and λ_aug ≈ 0.504.
- **Interpretation rule honored:** the loss is cosine-based and scale-invariant, so the norm
  ratio alone proves nothing. Measured jointly, ‖∂CL/∂h‖ scales as ~1/‖h‖ exactly as required.
  (A verifier **refuted** the first agent's "22.8× tracks 16.6×" figure — the correct ratio is
  10.1–10.9 across four conventions — while leaving the conclusion intact.)

`POSITIVE_PAIR_DISTRIBUTION_SHIFT = MATERIAL` at realistic states, EXTREME only un-warmed.

## 5. BatchNorm buffer audit

Census: exactly **four** stateful normalizers — `model.encoder.bns[0..1]` and
`view_learner.encoder.bns[0..1]`, all `BatchNorm1d(32)`. `trans_conv.bns` are LayerNorm (zero
buffers, verified bit-identical train vs eval); ToyNet is pure Linear/LeakyReLU. Nothing else
can drift.

```
PHASE1_CURRENT_MUTATES_MODEL_BN_BUFFERS = NO    (model is in eval under legacy)
PHASE1_CURRENT_MUTATES_VIEW_BN_BUFFERS  = YES   (+1 num_batches_tracked / batch)
PHASE2_CURRENT_MUTATES_MODEL_BN_BUFFERS = YES   (+2 / batch: original AND augmented forward)
PHASE2_CURRENT_MUTATES_VIEW_BN_BUFFERS  = NO
CROSS_NET_BN_CONTAMINATION_IN_LEGACY    = NONE  (ownership is already clean — by accident)
```

Drift is fast and consumed downstream: `running_var` collapses geometrically toward the true
activation variance, and a later **eval-mode** CL measurement moves 1.966 → 1.446 (varied
batches) / 1.104 (same batch) by K=100. `embedding_evaluation.py:199` calls `encoder.eval()`,
so those drifted buffers are exactly what the downstream representation path reads.

**Restore policy proven** (three ways): output bit-identical (0 ULP) to the unrestored
train-like forward; non-owner `bn_delta` exactly 0.0 including int64 `num_batches_tracked`;
and a fresh eval forward on the restored net equals a pristine build. Mechanism: corrupting all
buffers leaves **train-mode** output bit-identical, so restoration is free.

⚠ **Verifier-caught implementation constraint (would have shipped a crash):** the restore must
run **after `backward()`**. Placing it between forward and backward — where the forward-only
proof placed it — raises `RuntimeError: one of the variables needed for gradient computation has
been modified by an inplace operation`, because native `batch_norm` saves running stats as
backward inputs. *I hit this myself in the §12 probe and it confirmed the constraint.*

## 6. Four forward-state policies

| policy | matched-draw J gap | gap/SD (indep.) | Φ-grad cosine | non-owner BN mutation |
|---|---|---|---|---|
| **P0 legacy** | **+1.9336** | 16.7× | **0.314** | none (accidentally safe) |
| P1 both-eval | 0.00000 | 0.31 | 1.00000 | none |
| P2 both-train naive | 0.00000 | 0.26 | 1.00000 | **6.17 L1/call, 61.65 cumulative** |
| **P3 consistent, ownership-safe** | **0.00000** | 0.26 | 1.00000 | **0 (restored)** |

BN alone accounts for **98.3 %** of P0's matched-draw gap. P1 is rejected: it is not the
train-like distribution and CL is nearly signal-free there (‖∂(−CL)/∂Φ‖ = 0.032 vs 2.076).
P2 ≡ P3 on every loss/gradient metric; they differ *only* in that P2 silently mutates the frozen
player's buffers (shifting its eval CL by −0.0323 with `max|ΔΘ| = 0`).

⚠ Verifier notes accepted: the "0.00000 / 1.00000" figures are **tautological** (with no step
between phases and matched draws the two forwards are literally the same function) and are
labelled as such; the substantive evidence is that P3's cross-phase relative L2 of the original
representation is **exactly 0** vs 60.65 under P0, and |CL gap| falls 1.72 → 0.097 while
within-phase SD rises only 1.5× — so the improvement is **not** denominator inflation
(hypothesis explicitly disproved). Verifier also refuted "P3 introduces no new effect": it
raises Phase-1 objective noise ~**250×** and Φ-gradient noise ~**27–35×**, and the pre-existing
`max_norm=5.0` clip now fires on some batches. Both costs are documented in the code.

## 7. Real Adam step (SGD gradient norms do NOT transfer)

`ADAM_STEP_1_IS_SIGN_LIKE = YES`: cos(Δ, −sign g) = **0.9999**, `max|Δᵢ|` = lr **exactly**,
‖Δ‖ = lr·√P for every objective regardless of ‖g‖ (0.032 … 2.148). So update-share statements
must come from the measured displacement, never from gradient-norm ratios.

| | ΔJ (8 seeds) | cos(Δ, paper composite) | cos(Δ, pure-CL) |
|---|---|---|---|
| P0 | −0.0553 ± 0.0070, **8/8 < 0** | +0.909 | **+0.041** (wrong sign 3/8) |
| P3 | −0.3898 ± 0.217, **8/8 < 0** | +0.992 | **+0.669** |

⚠ Verifier **refuted** framing the P0 smallness as a *policy* property: it is a
BatchNorm-at-init artifact — the raw cos(g_FULL, g_CL) goes −0.093 (P0@init) → **+0.759**
(P0 warmed) → +0.988 (P3). So warming BN, not the policy alone, restores the CL component.

## 8. Selected correction

```
SELECTED_STAGE8C_POLICY = P3   (implemented as --phase_state_mode consistent)
WHY  = (1) both phases share one train-like forward DISTRIBUTION, so Phi and Theta optimize
           the same representation game; (2) a phase may mutate only the state -- parameters
           AND persistent buffers -- of the player it owns, enforced structurally instead of
           by accident; (3) Phi's paper-consistent composite stays coherent (cos 0.9996 at
           epoch 5); (4) Theta still gets a normal train-like forward, unlike P1;
           (5) the positive pair becomes MORE recognizable, not less; (6) minimal deviation --
           the phase structure, losses, weights and optimizer membership are untouched;
           (7) reproducible and unit-tested.
PAPER_ALIGNMENT = REASONED_ADAPTATION
    (the paper specifies no module train/eval discipline; the authors' release uses the same
     legacy arrangement, so this departs from the release on a point the paper is silent about)
```

## 9. Implementation and tests

`--phase_state_mode {legacy,consistent}` in `unsupervised/training.py` plus both parsers.
Default **legacy** everywhere — no silent default change. Helpers `_bn_buffer_snapshot` /
`_bn_buffer_restore` carry the after-backward constraint in their docstrings.

**Tests 18/18** (`tests/test_stage8c_phase_state_mode.py`):
snapshot/restore incl. int64 `num_batches_tracked`; **legacy reproduces the pre-change file
extracted from git BIT-FOR-BIT** (that code is deterministic at this batch size — its own
run-to-run noise is `0.000e+00` — so the bar was exact equality, not a tolerance); BN ownership
witnessed by `num_batches_tracked` (+2 model / +1 view per batch in *both* modes, i.e. the
non-owner's forwards are rolled back under `consistent`); no non-owner **parameter** moves with
the view lr zeroed; the two modes give materially different Phase-1 objectives, both finite.

## 10. Five-epoch mechanism sanity run (`consistent`, `alff_new_z`, frozen hyperparameters)

1025 s total (~3.4 min/epoch), batch 32, `drop_last=True`. No tuning, no accuracy.

| ckpt | CL | J (Φ obj) | gate mean | gate std | q05 | q95 | subj eRank | pos cos | margin | ‖G′‖/‖G‖ | finite |
|---|---|---|---|---|---|---|---|---|---|---|---|
| epoch0 | 3.4499 | 9.410 | 0.4949 | 0.2911 | 0.0471 | 0.9504 | **8.36** | 0.6235 | +0.00411 | **23.85** | ✓ |
| epoch1 | 3.4764 | 6.404 | 0.4890 | 0.3196 | 0.0064 | 0.9637 | 6.88 | 0.6186 | +0.00186 | 1.34 | ✓ |
| epoch3 | 3.4403 | 0.707 | 0.5113 | 0.3165 | 0.0231 | 0.9753 | 5.28 | 0.7650 | +0.00029 | 1.62 | ✓ |
| epoch5 | 3.4357 | **0.431** | 0.5103 | 0.3157 | 0.0232 | 0.9743 | **5.27** | **0.8147** | +0.00165 | 2.41 | ✓ |

Model BN buffers calibrate as intended: `running_var` 1.000 → 0.354 → 1.377 → 1.355,
`num_batches_tracked` 0 → 58 → 174 → 290.

Training-loop scalars: `|gΦ|` 3.53 → 1.90, `|gΘ|` 5.41 → 2.67, max update-to-weight ratio
3.97e+10 (epoch 1) → 0.28 (epoch 5). *The epoch-1 figure is an artifact of biases initialized to
exactly 0 (a ratio against ~0), not divergence — it falls monotonically and everything stays
finite.*

## 11. Epoch-5 trained-weight probes

| | ‖g_CL‖ | ‖g_CE‖ₛ | ‖g_REG‖ₛ | ‖g_KLD‖ₛ | cos(u_FULL, u_CL) | cos(u_FULL, u_J) | clip fired |
|---|---|---|---|---|---|---|---|
| epoch0 | 1.938 | 0.127 | 0.028 | 0.484 | **0.9116** | 0.9998 | 0/4 |
| epoch5 | 4.896 | 1.348 | 0.118 | 3.923 | **0.6249** | 0.9996 | **2/4** |

The contrastive gradient **grows 2.5×** with training and the update stays strongly aligned with
the pure-adversarial CL direction (0.62–0.91) — against **0.041** measured under legacy at init.
Φ's composite remains collinear with the paper objective throughout (0.9996). The clip firing
2/4 at epoch 5 is the pre-existing guard doing its job, exactly as the verifier predicted.

```
GATE_LEARNS                     = YES   (std 0.2911 -> 0.3157, +8.5%; q05 0.0471 -> 0.0232 and
                                         q95 0.9504 -> 0.9743, i.e. the gate becomes more
                                         decisive at both tails; mean 0.4949 -> 0.5103)
GATE_STD_CHANGES                = YES
POSITIVE_IDENTITY_IMPROVES      = INCONCLUSIVE / MIXED
    positive cosine improves strongly (0.6235 -> 0.8147) but the positive-minus-negative
    MARGIN does not (+0.00411 -> +0.00165): positives and negatives are rising together,
    i.e. the embedding is becoming more anisotropic rather than more discriminative.
SUBJECT_EFFECTIVE_RANK_COLLAPSES = PARTIAL -- 8.36 -> 5.27 (-37%), a real decline but not a
                                   collapse to ~1. Recorded as CONCERN.
VIEW_NORM_RATIO_STABILIZES      = YES   (23.85 -> 1.34-2.41; the Stage-7 norm gap was indeed
                                         an un-warmed-BatchNorm artifact and it disappears
                                         within one epoch of real training)
FULL_OBJECTIVE_REMAINS_FINITE   = YES   (finite at every checkpoint; J 9.410 -> 0.431)
```

## FINAL VERDICT BLOCK

```
STAGE8B_ACCEPTED = YES
EQ15_THETA_SIDE_WITH_ALFF_MIJ = ONE_SIDED
EQ15_MIJ_PAPER_CONTRADICTION = YES
CURRENT_CROSS_PHASE_STATE_SHIFT = EXTREME
    (CL gap -1.67 at 9.3x the noisier phase's own SD; zero distribution overlap over 192
     draws/phase; Phi-gradient cosine 0.31; original-representation rel L2 37.65)
PRIMARY_CROSS_PHASE_CAUSE = BN
    (92.6% of CL effect mass, 100% of original-representation mass, 98.3% of the P0 J gap
     -- state-scoped: the share falls as buffers calibrate, ~67-69% of the gap surviving
     200-400 calibration passes)
VIEW_NORM_GAP_REPLICATES = YES  (16.62 at h / 25.09 at z, eval regime -- REGIME-CONDITIONAL:
     1.07 under bn='train' and 1.06 after a single warm pass)
VIEW_NORM_GAP_PRIMARY_CAUSE = LAMBDA  (dominant in every regime; GRAPH secondary once BN
     matures; INTERACTION 7-18%)
POSITIVE_PAIR_DISTRIBUTION_SHIFT = MATERIAL  (EXTREME only in the un-warmed eval regime)
BEST_PHASE_STATE_POLICY = P3 (--phase_state_mode consistent)
NONOWNER_BN_BUFFER_MUTATION_PREVENTED = YES  (bn_delta exactly 0.0 incl. int64
     num_batches_tracked; output bit-identical, 0 ULP; restore placed AFTER backward)
ADAM_FULL_UPDATE_PAPER_OBJECTIVE_COHERENT = YES  (dJ < 0 on 8/8 seeds under BOTH policies;
     cos(update, paper composite) 0.909 (P0) / 0.992 (P3); 0.9996 at epoch 5)
ADAM_CL_COMPONENT_MATERIALLY_PRESENT = YES under P3 (cos 0.669 at init, 0.625 at epoch 5)
     -- SMALL under P0 (0.041), but a verifier established that is a BatchNorm-at-init
     artifact, not a policy property (raw cos goes -0.093 -> +0.759 when P0's BN is warmed)
SAFE_TO_IMPLEMENT_STAGE8C = YES
STAGE8C_IMPLEMENTED_AS_EXPLICIT_MODE = YES  (--phase_state_mode {legacy,consistent})
LEGACY_MODE_PRESERVED = YES  (bit-for-bit vs the pre-change file from git; default unchanged
     in train_one_epoch and in both parsers)
FIVE_EPOCH_SANITY_COMPLETED = YES  (1025 s; checkpoints at epochs 0,1,3,5)
GATE_LEARNS = YES
POSITIVE_IDENTITY_IMPROVES = INCONCLUSIVE
     (positive cosine 0.6235 -> 0.8147 but margin +0.00411 -> +0.00165)
SUBJECT_REPRESENTATION_HEALTH = CONCERN
     (subject effective rank 8.36 -> 5.27, -37%, while the margin fails to improve; nothing
      is non-finite and nothing collapses to rank ~1)
STAGE8C_BLOCKER_REMAINING = NONE for the training-state axis -- the cross-phase shift is
     removed, ownership is enforced, and the objective is coherent. The OPEN item is
     representation health under training (rank decline + flat margin), which is trained
     evidence, not initialization evidence.
SAFE_TO_BEGIN_LOCKED_DEVELOPMENT_EXPERIMENTS = YES
NO_NEW_STAGE_FROM_INIT_ONLY_EVIDENCE = TRUE
```

## Confirmed issues only

1. **Legacy's two phases score the objective on different surfaces** — fixed behind an explicit
   opt-in mode; legacy preserved bit-for-bit.
2. **Eq. 15's Θ side is not executable** with the paper's own parameter-free M_ij. A paper-level
   contradiction; **not** an artifact of our choices; not fixed here.
3. **Representation health under training is a CONCERN** — subject effective rank falls 37 % over
   5 epochs while the positive margin fails to improve. This is *trained* evidence and is the
   only thing that could justify a future structural stage; **I have not opened one.**
4. **P3 costs noise** (~250× Phase-1 objective, ~27–35× Φ-gradient) and makes the existing clip
   fire. Documented in-code, accepted deliberately.

**Stage 8C stops here. No Stage 8D opened. No full CV launched.**
