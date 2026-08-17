# STAGE 5 — TAENCODER / GAMMA / BETA / LAMBDA / ATTENTION FUSION AUDIT

Date: 2026-08-17 · Stages 1-4 frozen · Seed 42 · Read-only (zero production edits) · No epochs, no accuracy.

## 1. Paper vs authors vs ours

| quantity | paper | authors' released code | our code | verdict |
|---|---|---|---|---|
| X_topo source | GCN(X, A) (Eq. 17) | GCN, then `F.normalize(x)` | GCN, un-normalized (deliberate) | ours closer to Eq. 17 (no normalize in paper) |
| X_atte source | Eq. 16 Transformer/Graphormer attention | DIFFormer-style linear attention (uncited), computed then DISCARDED | softmax(QK^T/√d)V, per-subject blocks | ours matches Eq. 16; authors neither |
| attention equation | softmax attention | no softmax anywhere | rows sum to 1.0 (verified) | ours PASS |
| attention subject isolation | — | unbatched only | verified exact | PASS |
| gamma meaning | "strength of RETAINED edges … average of all elements in the ADJACENCY matrix" → retention | epoch-level `np.random.beta(fin_reg, 1-fin_reg)` computed but unused in fusion | per-subject; see call-site table | paper = retention |
| gamma original graph | adjacency initialized all-ones → 1 | n/a (fusion dead) | `paper_literal`: FALLBACK per-subject mean(gcn_w) = signal strength; `baseline`: 1.0 | **current paper_literal deviates** |
| gamma augmented graph | mean of gate/adjacency elements | n/a | per-subject mean(gate) | ours paper-aligned (Stage 2) |
| Beta parameters | λ ~ B(γ, 1−γ) (Eq. 18) | never executed in fusion | literal: B(γ,1−γ); reversed: B(1−γ,γ) | both offered; see contradiction |
| expected λ | E=γ (equation) vs E=1−γ (prose) | n/a | matches each convention (sampled-verified) | **equation and prose CONTRADICT** |
| λ train/eval | reparameterized sample | n/a | train: sampled; eval: deterministic expectation | PROJECT_CORRECTION (paper silent, authors n/a) |
| λ granularity | per graph | n/a | per subject [B], `lambda[batch]` per node, isolation verified | PASS |
| X_atte normalization | none in Eq. 19 | normalized (then discarded) | **L2-normalized before fusion** | PROJECT_STABILIZATION |
| X_topo normalization | none | normalized | un-normalized | ours matches paper |
| Eq. 19 fusion | X_topo + λ·X_atte | **X_update = normalize(X_topo); attention DISCARDED** (fusion line commented) | X_topo + λ·‖X_topo‖·normalize(X_atte) | **neither matches the equation literally** |
| extra scaling | none | none (no fusion) | ‖X_topo‖ multiplier | PROJECT_STABILIZATION |
| authors fusion enabled | — | **NO** | default YES (`enable_attention_mix`) | documented discrepancy |

## 2-4. X_topo / X_atte / isolation (measured)

X_topo [180,32] finite; node norms 0.007-0.417 (median 0.349). Attention: softmax rows sum
to 1.0 exactly; A-alone == A-batched; A bitwise-stable when B is radically altered. PASS.

## 6. Gamma call-site table (traced)

`gamma_mode='paper_literal'` (all campaign arms): Phase1/Phase2 ORIGINAL and eval →
`gamma=None` → **FALLBACK = per-subject mean(gcn_w)** (signal strength); AUGMENTED →
explicit per-subject mean(gate). `baseline`/`literal_beta`: original/eval gamma=1.0.
Nested-CV identical (shared loop). No unintended fallback beyond the designed one — but the
designed fallback itself is a deviation from paper retention (below).

## 7. signal_strength diagnostic

Two subjects: mean(W)=0.4672/0.2366 vs mean(|W|)=0.4682/0.2865. All 956: mean(W)=0.3387,
mean(|W|)=0.3592, diff 0.0205. Under signed edges the executable fallback computes
**mean(W)** while historical comments reference mean(|W|) — a real comment/code divergence
(deferred; no patch). **SIGNAL_STRENGTH_GAMMA_STATUS = PROJECT_ABLATION** (paper's γ is
retention from the adjacency, not edge-weight magnitude).

## 8. Beta table (theoretical + 20k-sample verified)

γ ∈ {0.1,0.25,0.5,0.75,0.9}: literal B(γ,1−γ) → E[λ]=γ exactly (sampled 0.0999-0.9028);
reversed B(1−γ,γ) → E[λ]=1−γ. Eq. 18's symbol order = literal (attention grows WITH
retention). Prose ("as connectivity diminishes … focus more on global information") =
reversed. **EQUATION AND PROSE CONTRADICT; both recorded; not resolved by intuition.**
Authors' executable code: no Beta reaches fusion at all.

## 9. paper_literal profile verdict

**CURRENT_PAPER_LITERAL_IS_ACTUALLY_LITERAL = NO.** Its Beta order is Eq.18-literal, but its
original-view γ is the signal-strength fallback (mean of gcn_w), not the paper's retention
(=1 for the initialized all-ones adjacency / unaugmented view). A true paper-equation
profile = {γ_orig=1, γ_aug=mean(gate), λ~B(γ,1−γ), fusion X_topo+λ·X_atte}.

## 10-11. λ granularity + stochasticity (measured)

γ [B] → λ [B] → per-node via `lambda[batch]`; changing only B's γ leaves A bit-stable while
B changes. Train: seeded-reproducible, seed-sensitive. Eval: exactly repeatable.
Deterministic eval classified: paper-unspecified, authors-n/a → PROJECT_CORRECTION (kept).

## 12-13. Eq. 19 — the material finding

```
PAPER:    X_update = X_topo + lambda * X_atte                      (raw X_atte, ||~4.18||)
AUTHORS:  X_update = normalize(X_topo)          (attention discarded; fusion commented out)
OURS:     X_update = X_topo + lambda * ||X_topo|| * normalize(X_atte)
```
Numerical comparison, same X_topo/X_atte/λ, two real subjects:
**ours vs paper: relative difference 0.978, cosine similarity 0.211** — the extra operations
are MATERIAL, not cosmetic. (Mechanism: raw X_atte norms ≈ 4.18 vs X_topo ≈ 0.35, so the
literal equation is attention-dominated; our rescaling makes attention proportional to each
node's own topology magnitude.) paper vs authors-executable: cosine −0.096 (unrelated).
Classification: normalize(X_atte) + ‖X_topo‖ multiplier = **PROJECT_STABILIZATION**
(documented rationale in-code), **EQ19_CURRENT_MATCHES_PAPER = NO**.

## 14-15. Extreme γ + clamp

Literal/reversed sampled means match theory across γ ∈ {0.01…0.99}, all finite. Clamp
[1e-4, 1−1e-4] is numerically required (Beta(0,1)/B(1,0) undefined). γ_orig=1 → 0.9999 →
literal λ≈1 (full attention), reversed λ≈1e-4 (none): the clamp is benign for literal,
decisive for reversed.

## 16-18. Profiles, gradients, batch composition (measured)

AUTHORS_LITERAL: attention does NOT affect the final embedding. PAPER: it must. OUR
default: it does (mix ON). Gradients: mix ON → attention branch 2372.4 (live); mix OFF →
exactly 0 (genuinely disconnected). Batch composition (eval, deterministic λ): A alone ==
A batched, before and after fusion. PASS.

## 20. Proposed profile separation (PROPOSAL ONLY — not implemented)

- `PAPER_EQUATION`: γ=retention (orig 1, aug mean(gate)); λ~B(γ,1−γ); fusion X_topo+λ·X_atte (raw).
- `PAPER_PROSE_CORRECTED`: same but B(1−γ,γ) — only if the prose reading is chosen.
- `AUTHORS_LITERAL`: mix OFF (already exists via replicate/enable_attention_mix).
- `CURRENT_ABIDE`: today's stabilized behavior, renamed away from "paper_literal"
  (the name overstates; it is Beta-order-literal only).
Awaiting decision; zero code changed in Stage 5.

## Final block

```
PAPER_GAMMA_MEANING = retention -- "strength of retained edges", average of the ADJACENCY matrix elements
ORIGINAL_GAMMA_PAPER = 1 (adjacency initialized all-ones; unaugmented view)
AUGMENTED_GAMMA_PAPER = mean of gate/adjacency elements, per graph
CURRENT_ORIGINAL_GAMMA = per-subject mean(gcn_w) via fallback (paper_literal/signal_strength); 1.0 (baseline/literal_beta)
CURRENT_AUGMENTED_GAMMA = per-subject mean(gate)  [paper-aligned]
SIGNAL_STRENGTH_GAMMA_STATUS = PROJECT_ABLATION (+ comment/code mean(|W|)-vs-mean(W) divergence under signed edges, diff 0.0205 cohort-wide)
PAPER_BETA_ORDER = B(gamma, 1-gamma)   (Eq. 18 symbol order)
PAPER_BETA_EXPECTED_LAMBDA = gamma
PAPER_PROSE_BETA_ORDER = CONTRADICTORY (prose requires E[lambda] = 1-gamma)
AUTHORS_BETA_ORDER = NOT EXECUTED in fusion (fusion commented out; epoch-level beta unused)
CURRENT_LITERAL_BETA_ORDER = B(gamma, 1-gamma)
CURRENT_REVERSED_BETA_ORDER = B(1-gamma, gamma)
CURRENT_PAPER_LITERAL_ACTUALLY_LITERAL = NO (Beta order literal; gamma_orig is signal strength, not retention=1)
PAPER_EQ19 = X_topo + lambda * X_atte
AUTHORS_EQ19_EXECUTABLE = normalize(X_topo); attention discarded
CURRENT_EQ19 = X_topo + lambda * ||X_topo|| * normalize(X_atte)
CURRENT_EXTRA_XNORM = YES
CURRENT_XATTE_NORMALIZED = YES
EQ19_CURRENT_MATCHES_PAPER = NO (material: rel diff 0.978, cosine 0.211; classified PROJECT_STABILIZATION)
ATTENTION_SUBJECT_ISOLATION = PASS
LAMBDA_SUBJECT_ISOLATION = PASS
TRAIN_EVAL_DETERMINISM = PASS
ALL_STAGE5_INTERNAL_TESTS = PASS (13/13)
SAFE_TO_FREEZE_STAGE5 = NO (two confirmed paper-mismatches recorded -- Eq.19 extra operations
    and the paper_literal gamma_orig misnomer -- pending the Section-20 profile decision)
```
