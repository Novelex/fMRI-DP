# STAGE 8A — PARAMETER / OPTIMIZER OWNERSHIP AUDIT

Date: 2026-08-18 · Stages 1–6 frozen, Stage 7 complete · **DIAGNOSTIC ONLY: zero production
edits, no epochs, no accuracy** · Features `alff_new_z`, profile `paper_intent`,
signed_edges=True, one fixed real batch (B=8, dataset indices 0,17,…,119), ce_lambda=2.0,
reg_lambda=0.2, kld_lambda=0.003.

**Method — fidelity by instrumentation, not replication.** A first attempt to *replicate* the
two phases drifted from the real RNG stream (view_loss −11.0401 vs the real −11.0291) and was
discarded. Every number below comes from running the **REAL
`unsupervised.training.train_one_epoch`** with the two optimizer *objects* wrapped, so that at
the exact moment each `.step()` is called we capture per-group gradient norms and the
parameter delta that step produces. Nothing in the repo was modified.

## 1. Paper ownership (read directly from `docs/grastical.pdf`)

The paper defines **four** parameter sets, not two:

| symbol | paper definition | direction in Eq. 22 |
|---|---|---|
| Φ | "the network parameters encoding the adjacency matrix A" (Eq. 15); elsewhere "the augmenter GNN and MLP" | **min** I_R+I_N |
| Ψ | "the learnable parameters of the TAE" (Eq. 21) | **min** |
| Θ | "the network parameters encoding the node vectors" (Eq. 15) | **max** |
| Ω | "the learnable parameters of the projection head" (Eq. 21) | **max** |

Eq. 21 `min_Ψ max_Ω I_R` · Eq. 22 `min_{Φ,Ψ} max_{Θ,Ω} (I_R + I_N)`. Since `L = −I_R`,
maximizing I_R ⇒ **descending** the contrastive loss, minimizing I_R ⇒ **ascending** it.

⚠ **PAPER-INTERNAL INCONSISTENCY (recorded, not resolved):** the TAE *is* the node-vector
encoder, so Eq. 15 calls that module **Θ (maximize)** while Eq. 21 calls it **Ψ (minimize)**.
The same physical module is assigned opposite optimization directions. No implementation can
satisfy both readings simultaneously.

## 2. Authors' released ownership (source-derived, `original_grastiacl_full/GraSTIACL.py`)

```python
x, _, edge_logits = model(...)                      # <-- edge_logits comes from MODEL.net
_, mu, std, edge_prod = view_learner(...)           # <-- view_learner's own logits DISCARDED
gate_inputs = (gate_inputs + edge_logits) / temperature
batch_aug_edge_weight = batch.edge_weight * torch.sigmoid(gate_inputs).squeeze()
view_loss = model.calc_loss(x, x_aug) - args.kld_lambda * kld_loss
(-view_loss).backward(); view_optimizer.step()
```

⚠ **In the authors' release the augmentation gate is produced by `model.net`, i.e. by the Θ
side.** The view learner's first return value is discarded. Consequently `model.calc_loss(x,
x_aug)` has **no gradient path into `view_learner` at all** — both `x` and `x_aug` come from
`model`, and the gate comes from `model.net`. Their `view_optimizer` therefore optimizes only
`−kld_lambda·KLD`, i.e. **their Φ merely minimizes the KL divergence and is not an adversary
on the contrastive objective**; meanwhile the actual augmenter (`model.net`) sits in
`model_optimizer` and is trained to **minimize** the contrastive loss. This is our project's
documented, deliberate deviation: we route the gate through `view.net` instead
(`training.py:188-199`, comment "view_learner IS the augmenter (Eq. 3's Phi)").

## 3. Ownership table — built from `data_ptr()` object identity, never from names

Optimizer membership: `model_optimizer = Adam(model.parameters())`,
`view_optimizer = Adam(view_learner.parameters())` (`training.py:118-119`).

| parameter group | #params | paper owner | authors-release owner | our owner | in model_opt | in view_opt | shared object? | P1 grad | P1 stepped | P2 grad | P2 stepped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `model.encoder` (TAE) | 4,672 | **Θ max / Ψ min (conflict)** | Θ | Θ | **18/18** | 0 | no | **5.6477** | **NO** | **5.6847** | **YES** (Δ 3.406e-02) |
| `model.net` (ToyNet/VIB) | 203,354 | Φ (authors' augmenter) | Θ (produces the gate) | Θ | **10/10** | 0 | no | **0.0000** | NO | **0.0000** | **NO** (Δ 0) |
| `model.proj_head` | 2,112 | **Ω max** | Ω | Ω | **4/4** | 0 | no | 0.3774 | **NO** | **2.5706** | **YES** (Δ 2.298e-02) |
| `view.encoder` (augmenter GNN) | 4,672 | **Φ min** | Φ | Φ | 0 | **18/18** | no | **0.0000** | NO | **0.0000** | **NO** (Δ 0) |
| `view.net` (ToyNet/VIB) | 203,354 | Φ min | Φ (KLD only) | **Φ — produces the gate** | 0 | **10/10** | no | **0.7244** | **YES** (Δ 2.254e-01) | **2.9729** | **NO** |
| `view.mlp_edge_model` (augmenter MLP) | 4,225 | **Φ min** | Φ | Φ | 0 | **4/4** | no | **0.0000** | NO | **0.0000** | **NO** (Δ 0) |

Sub-group breakdown (same measurement, finer split):

| sub-group | #params | P1 grad | P1 Δ | P2 grad | P2 Δ |
|---|---|---|---|---|---|
| `model.encoder` / GCN (convs) | 1,120 | 0.3725 | 0 | 5.5155 | 1.673e-02 |
| `model.encoder` / **BatchNorm affine** | 128 | **5.6288** | 0 | 1.0908 | 5.657e-03 |
| `model.encoder` / TAE attention | 3,424 | 0.2735 | 0 | 0.8398 | 2.912e-02 |
| `view.encoder` / GCN (convs) | 1,120 | 0.0000 | 0 | 0.0000 | 0 |
| `view.encoder` / BatchNorm affine | 128 | 0.0000 | 0 | 0.0000 | 0 |
| `view.encoder` / TAE attention | 3,424 | 0.0000 | 0 | 0.0000 | 0 |

Note: in Phase 1 the model's gradient is **overwhelmingly on the BatchNorm affine parameters**
(5.6288 of a 5.6477 group norm) — consistent with Stage 7's finding that BN dominates the
phase geometry at initialization.

## 4. Why three groups are dead — mechanism, traced in source

- **`model.net`**: both phases call `model(..., dyn_weight=None)` (`training.py:183, 215, 275,
  292`), and `GInfoMinMax.forward` returns `z, node_emb` *before* touching `self.net` when
  `dyn_weight is None`. The module is therefore **never executed**. 203,354 parameters.
- **`view.encoder`**: `ViewLearner.forward` computes `_, node_emb = self.encoder(...)` but
  `node_emb` is used **only in the `dyn_weight is None` branch**. Training always passes
  `batch.dyn_weight` (not None), so the encoder runs and its output is **discarded**. 4,672
  parameters. (Compute is spent; gradient is zero.)
- **`view.mlp_edge_model`**: used **only** in the `dyn_weight is None` branch — never in
  training. 4,225 parameters.

**Total dead: 212,251 of 422,389 parameters = 50.25%.** (For scale: the paper reports the main
model at "about 422k parameters", which our 422,389 matches.)

Effective optimized capacity: **Θ = 6,784** (`model.encoder` + `model.proj_head`) versus
**Φ = 203,354** (`view.net`) — a **30× asymmetry**, independently confirming the Stage-3
measurement.

## 5. Gradient-without-step and cross-phase accumulation (event trace)

Captured event order from the real epoch:
`zero_grad(view) → step(view) → zero_grad(model) → step(model)`.

- **Phase 1 computes a full gradient for Θ and throws it away.** At `model_optimizer.zero_grad()`
  (start of Phase 2) the model side still carries `model.encoder` grad **5.6477** and
  `model.proj_head` grad **0.3774** from Phase 1's `(-view_loss).backward()`. These are cleared
  and never applied. Wasted backward compute, **not** a correctness fault.
- **Phase 2 accumulates gradient into Φ that is never applied.** `model_loss.backward()` puts
  grad **2.9729** on `view.net` (the gate is differentiable), on top of Phase 1's residual
  0.7244. `view_optimizer.step()` is not called in Phase 2, and the next batch's
  `view_optimizer.zero_grad()` clears it. **Verified safe** — the accumulation never reaches a
  step — but it means the Phase-1 gradient-clip (`max_norm=5.0`) operates on a tensor that will
  later be contaminated and discarded.
- **Shared / double-optimized parameters: NONE.** `len(model_opt_ptrs ∩ view_opt_ptrs) = 0`.
  The two optimizers partition the parameter space exactly.

## 6. Real optimization direction — the decisive test

Probe design: Φ reaches the contrastive loss **only** through the gate, so the Phase-1 probe
recomputes the gate from the updated `view.net` with the model held fixed. Θ does not affect
the gate, so the Phase-2 probe holds the gate **fixed** and restores pre-epoch BatchNorm
buffers, isolating the Adam step from BN drift. **All probe forwards are seeded** — an earlier
unseeded version was dominated by Beta-λ/dropout noise (Stage-7 measured train-mode loss sd
0.09–0.26) and its numbers were discarded.

8 seeds (42, 7, 2024, 123, 5, 11, 77, 314):

| surface | phase | intended | measured | mean ΔL | range |
|---|---|---|---|---|---|
| **train** (the surface Phase 2's gradient is computed on) | **Phase 1** | ASCENT | **DOWN 7/8** | **−0.070025** | [−0.16081, +0.00073] |
| **train** | **Phase 2** | DESCENT | **DOWN 8/8** ✓ | −0.062615 | [−0.08884, −0.05225] |
| eval (deterministic, Phase-1 module mode) | Phase 1 | ASCENT | down 6/8 | −0.000052 | [−0.00028, +0.00043] |
| eval | Phase 2 | DESCENT | down 4/8 | −0.000320 | [−0.00569, +0.00191] |

**On the train surface both players move the contrastive loss DOWN. The adversary is not
adversarial.** On the eval surface Phase 1's effect is ~10⁻⁵ — negligible in either direction.

### Mechanism — measured, not inferred

`view_loss = contrastive − 2.0·CE − 0.2·REG − 0.003·KLD`, and Phase 1 **maximizes** it. The
gradient of each *coefficient-weighted* term with respect to `view.net` (5 seeds):

| term (as weighted in view_loss) | ‖∂/∂ view.net‖ | ratio vs contrastive |
|---|---|---|
| contrastive | **0.012518** | 1× |
| 0.2 · REG | 0.034213 | 2.7× |
| 2.0 · CE | 0.156422 | **12.5×** |
| **0.003 · KLD** | **0.606995** | **48.5×** |

The contrastive term — the one the adversary exists to maximize — is the **smallest**
contributor to Φ's gradient by a factor of 48. Phase 1 is, in practice, a KLD minimizer with a
CE minimizer attached; its effect on the contrastive objective is incidental and, measured,
usually negative. Note the KLD coefficient is *already* the small 0.003 — the raw KLD gradient
is ~200× the contrastive one.

### Code-level sign (what the source literally does)

```
view_loss  = model.calc_loss(x, x_aug) - ce_lambda*ce_loss - reg_lambda*reg - kld_lambda*kld_loss
(-view_loss).backward() ; view_optimizer.step()      # = ASCENT on view_loss      -> correct
model_loss = model.calc_loss(x, x_aug)
model_loss.backward()   ; model_optimizer.step()     # = DESCENT on model_loss    -> correct
```

**The literal signs are correct.** The failure is not a flipped sign; it is that
`view_loss ≠ contrastive loss`, and the three subtracted regularizers dominate Φ's gradient so
thoroughly that the realized direction on the contrastive objective inverts.

## FINAL VERDICT BLOCK

```
PAPER_THETA_OWNER = node-vector encoder (the TAE), MAXIMIZES I_R  (Eq. 15)
PAPER_PHI_OWNER   = adjacency/augmenter network ("augmenter GNN and MLP"), MINIMIZES I_R
                    (Eq. 15/22)
                    PLUS two further sets the spec's Theta/Phi dichotomy does not cover:
                      Psi = TAE parameters, MINIMIZES I_R (Eq. 21)
                      Omega = projection head, MAXIMIZES I_R (Eq. 21)
                    ** PAPER-INTERNAL CONFLICT: the TAE is simultaneously Theta (max) and
                       Psi (min). Unsatisfiable as printed. Recorded, NOT resolved here. **

AUTHORS_THETA_OWNER = model.{encoder, net, proj_head} via model_optimizer, minimizing
                      calc_loss + ce_lambda*ce_loss. CRUCIALLY their gate is produced by
                      model.net, so THE AUGMENTER LIVES INSIDE THETA and is trained to
                      MINIMIZE the contrastive loss.
AUTHORS_PHI_OWNER   = view_learner.{encoder, net, mlp_edge_model} via view_optimizer, but
                      calc_loss has NO gradient path into view_learner in their code, so
                      their Phi optimizes ONLY -kld_lambda*KLD. Not an adversary.

OUR_MODEL_OPTIMIZER_GROUPS = model.encoder (4,672) + model.net (203,354) + model.proj_head
                             (2,112) = 210,138 params, 32 tensors
OUR_VIEW_OPTIMIZER_GROUPS  = view.encoder (4,672) + view.net (203,354) + view.mlp_edge_model
                             (4,225) = 212,251 params, 32 tensors

SHARED_PARAMETERS = NONE
    (|model_opt data_ptr set ∩ view_opt data_ptr set| = 0; the partition is exact)

DOUBLE_OPTIMIZED_PARAMETERS = NONE

GRADIENT_WITHOUT_STEP_PARAMETERS =
    model.encoder    grad 5.6477 in Phase 1, never stepped in Phase 1 (cleared by
                     model_optimizer.zero_grad() at Phase-2 entry)
    model.proj_head  grad 0.3774 in Phase 1, same fate
    view.net         grad 2.9729 in Phase 2, never stepped in Phase 2 (cleared by
                     view_optimizer.zero_grad() at the next batch's Phase-1 entry)
    -> all three are computed-and-discarded. Wasted compute; verified NOT to leak into any
       optimizer step.

STEPPED_WITHOUT_INTENDED_OWNERSHIP =
    model.encoder (the TAE). Under Eq. 21's literal assignment the TAE is Psi and must
    MINIMIZE I_R (i.e. ASCEND the contrastive loss); our code steps it inside
    model_optimizer, which DESCENDS the contrastive loss (the Theta role of Eq. 15).
    This follows directly from the paper-internal conflict above and cannot be called an
    implementation error without first choosing a reading.

DEAD_OPTIMIZER_PARAMETERS =
    model.net            203,354 params — module never executed (dyn_weight=None in all four
                         model() call sites)
    view.encoder           4,672 params — executed, output node_emb DISCARDED in the
                         dyn_weight-not-None branch
    view.mlp_edge_model    4,225 params — only reachable in the dyn_weight-is-None branch
    TOTAL 212,251 of 422,389 = 50.25% of all parameters are in an optimizer, receive exactly
    zero gradient, and never move. Effective Theta = 6,784 vs effective Phi = 203,354 (30x).
    NOTE: the paper's Phi is "the augmenter GNN and MLP" — in our implementation BOTH the
    augmenter GNN (view.encoder) and the augmenter MLP (view.mlp_edge_model) are dead; only
    view.net (ToyNet/VIB) is alive.

PHASE1_REAL_DIRECTION = MIXED
    Code-level: correct ASCENT on view_loss. Realized on the CONTRASTIVE objective: DOWN in
    7/8 seeds on the train surface (mean -0.0700) and 6/8 on the eval surface (mean -5.2e-05).
    Cause measured: the contrastive term contributes only 0.0125 of Phi's gradient versus
    0.607 for the KLD term (48.5x) and 0.156 for the CE term (12.5x).

PHASE2_REAL_DIRECTION = DESCENT
    Correct and consistent: DOWN 8/8 on the train surface (mean -0.0626, range
    [-0.0888, -0.0522]). On the eval surface the movement is ~3e-04 and sign-inconsistent
    (4/8), which is the Stage-7 phase-mode mismatch showing through, not a Phase-2 defect.

STAGE8A_CONFIRMED_OWNERSHIP_BUG = YES
    50.25% of optimizer-registered parameters are dead, including BOTH modules the paper
    names as Phi ("the augmenter GNN and MLP"). The augmenter that actually acts is
    view.net, which the paper does not describe in that role.

STAGE8A_CONFIRMED_SIGN_BUG = YES
    Not a literal sign error — the backward/step signs are correct. The min-max game is
    nevertheless broken in effect: on the surface Phase 2 optimizes, BOTH phases move the
    contrastive loss DOWN (7/8 and 8/8). The adversary does not oppose the model because
    its objective is dominated 48:1 by the KLD regularizer.

SAFE_TO_CONTINUE_TO_STAGE8B = YES
    Ownership is now fully mapped with object identity, the dead set is enumerated exactly,
    and both failure mechanisms are measured rather than inferred. Stage 8B has what it
    needs. It must NOT begin before this evidence is reviewed.
```

## What Stage 8B will have to decide (listed, NOT actioned)

1. Whether Φ should be `view.net` (our routing) or `model.net` (authors' routing), and what to
   do with the paper's dead "augmenter GNN and MLP".
2. Whether the adversary's objective should be the contrastive term alone, or whether the
   KLD/CE/REG coefficients should be rebalanced so the contrastive term is not swamped 48:1.
3. Which reading of the TAE's role (Eq. 15 Θ-max vs Eq. 21 Ψ-min) the project adopts.
4. The Stage-7 phase-mode blocker, which is entangled with all of the above.

No production code was modified. No epochs were run. No accuracy was computed.
