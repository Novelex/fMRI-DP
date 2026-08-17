# STAGE 4 — M_ij / VIEW-LEARNER SIMILARITY TARGET AUDIT

Date: 2026-08-17 · Stages 1-3 frozen · Seed 42 · Read-only (zero production edits) · No epochs, no accuracy.

## 1. Paper vs authors vs ours

| quantity | paper | authors' released code | our code | verdict |
|---|---|---|---|---|
| M_ij feature source | "dot product of the i-th and j-th **nodal feature vectors** v_i, v_j passed through a sigmoid"; "node feature vectors, **composed of ALFF features from three frequency bands**" | **learned encoder embeddings** (`node_emb[src]*node_emb[dst]`) | raw ALFF slice `x[:, :3]` | **PAPER != AUTHORS; ours paper-aligned** |
| M_ij dimensionality | d=3 (Sec. 4.1) | emb_dim (learned) | 3 | ours matches |
| dot product | v_i^T v_j | emb dot | `sum(x_src*x_dst, dim=1)` (bitwise-verified) | ours matches |
| sigmoid location | once, in M's definition | once, in CE step | once, in training.py (`edge_prod_sig`) — verified exactly once | MATCH |
| edge ordering | — | same row-major loop | verified all 16,200 positions | PASS |
| CE prediction | Eq. 12's A = σ((noise+logits)/τ) — the reparameterized sampled adjacency | σ((logits+noise)/τ) | identical | **MATCH (literal Eq. 12+13)** |
| CE target | M_ij | σ(emb-dot).detach() | σ(ALFF-dot).detach() | ours paper-aligned |
| noise involvement | inherent to Eq. 12's reparameterization | in prediction only | in prediction only; M noise-free (bitwise) | MATCH |
| detach | M is input-derived, non-learnable | detached | detached | conceptually correct |
| gradient recipient | augmenter (Φ) | via edge_logits → their net | CE-alone: **view.net only** (0.085 norm; all five other modules zero; x.grad None) | measured |

## 2. Paper formula (established)

`M_ij = σ(v_i^T v_j)` with v = the 3-band ALFF nodal feature vector — explicit in the
paper's Eq. 4 prose. **PAPER_V_SOURCE = ALFF.** Authors' embedding-based M contradicts this;
we do NOT follow the authors here.

## 3-6. Executable-path, sigmoid-once, order, symmetry — all verified

edge_prod == `sum(x[src,:3]*x[dst,:3])` bitwise · alff_paper: `x[:,:3] == x` · M applied
sigmoid exactly once · all 16,200 edge positions aligned `(p//90, p%90)` · M symmetric ·
diagonal `M_ii == σ(||v_i||²)` exact.

## 7. Full-956 M distribution (alff_paper)

```
RAW dot:  min 0.0000  max 2.9708  mean 0.2423  std 0.2332
          pct 1/5/25/50/75/95/99: 0.007/0.026/0.089/0.171/0.312/0.687/1.157
M:        min 0.5000  max 0.9512  mean 0.5586  std 0.0536
          pct: 0.5016/0.5065/0.5221/0.5425/0.5774/0.6653/0.7608
M<0.5: 0.0000%   M~0.5: 0.09%   M>0.5: 99.91%   M>0.75: 1.10%   M>0.90: 0.011%
saturation M~0: 0   M~1: 0
diagonal mean 0.5758 · off-diagonal mean 0.5588
```

**The one-sided M distribution is a deterministic consequence of this project's frozen
alff_paper preprocessing ([0,1], A-GCL-aligned) combined with GraSTI Eq. 4.** GraSTI-ACL
specifies three-band ALFF node features but does not publicly specify the
preprocessing/normalization used to create the released precomputed norm_matrix. Therefore
the numerical M distribution of the authors' experiments cannot be established. CE-target
implication for THIS project's profile: the target distribution is one-sided (>=0.5),
narrow (std 0.054). Documented; no change triggered; Stage 1 not reopened.

## 8. Legacy-ALFF comparison (diagnostic only)

Legacy z-scored ALFF gives two-sided targets (49.3% of M below 0.5) vs alff_paper's 0.0%.
Explanatory only; Stage-1 ALFF frozen; no preprocessing change without new evidence.

## 9. mij_source

Paper-aligned setting: **mij_source='alff'** — the default in BOTH main and nested parsers
(verified) and passed explicitly by campaign scripts. `alff_pcc` = **PROJECT_ABLATION**
(deliberate, documented in-code), retained, never the paper profile.

## 11-12. CE prediction vs paper; noise

Current: `BCE(σ((logits+noise)/τ), σ(ALFF-dot).detach())`. Paper: Eq. 12 defines A as the
reparameterized noisy adjacency; Eq. 13's L_CE(A_ij, M_ij) compares that soft adjacency to
M. Current form matches **under the learned-Bernoulli-logit interpretation of Eq. 12**,
which is also what the authors' released executable code does. Note the notation ambiguity:
Eq. 12 writes `log(W_ij/(1−W_ij))`, but raw signed PCC cannot literally serve as a
Bernoulli probability. The public implementation instead uses learned edge logits in this
position. This is treated as a notation ambiguity, not a Stage-4 implementation error.
M bitwise noise-invariant; CE varies with the draw (1.0117 vs 1.0067) — inherent to the
paper's own reparameterization trick, not an error.

## 13. CE-alone gradient graph (measured)

view.net 0.0850 (10/10) · view.encoder 0 · view.mlp 0 · model.encoder 0 · model.net 0 ·
synthetic x-probe: **x.grad is None** (target detached, prediction x-free). CE's local
graph touches exactly the augmenter net. Min-max ownership deferred to Stage 8.

## 14-15. Independence + isolation

M bitwise-unchanged under static-PCC and dynamic-PCC perturbation; changes under ALFF
perturbation; batch isolation exact (A unchanged when B altered; A alone == A batched).

## 16. Self-edges

Eq. 9/13's 1/N² double sum covers all i,j → diagonal included by notation; prose silent →
**AMBIGUOUS in prose, notation-supported**. Read-only CE diagnostic: with 1.007541 vs
without 1.007190 (diff 0.00035, negligible). Self-edges retained.

## 17. Numerical safety

All 956 subjects: raw dots and M finite, M in [0.5, 0.9512], zero saturation, BCE inputs
finite. No sanitization used.

## Final block

```
PAPER_MIJ_FORMULA = sigmoid(v_i^T v_j), v = 3-band ALFF nodal feature vector
PAPER_V_SOURCE = ALFF
CURRENT_MIJ_SOURCE = raw ALFF slice x[:, :3] (mij_source='alff')
CURRENT_MIJ_SIGMOID_COUNT = 1
MIJ_EDGE_ORDER = PASS
MIJ_SYMMETRY = PASS
MIJ_BATCH_ISOLATION = PASS
MIJ_STATIC_PCC_INDEPENDENT = PASS
MIJ_DYNAMIC_PCC_INDEPENDENT = PASS
MIJ_ALFF_DEPENDENT = PASS
MIJ_RANGE_REAL_DATA = [0.5000, 0.9512]
MIJ_MEAN_REAL_DATA = 0.5586
MIJ_DIAGONAL_MEAN = 0.5758
MIJ_OFFDIAGONAL_MEAN = 0.5588
PAPER_ALIGNED_MIJ_SOURCE_SETTING = alff
ALFF_PCC_MIJ_STATUS = PROJECT_ABLATION
AUTHORS_MIJ_MATCHES_PAPER = NO (learned embeddings vs paper's explicit ALFF)
CURRENT_CE_PREDICTION = sigmoid((edge_logits + logistic_noise)/temperature)
PAPER_CE_PREDICTION = Eq.12's A -- the reparameterized (noisy) soft adjacency
CURRENT_CE_MATCHES_PAPER = YES under the learned-Bernoulli-logit interpretation of Eq.12,
    which matches the authors' released executable code
CE_NOISE_DEPENDENT = YES (inherent to the paper's reparameterization)
MIJ_TARGET_DETACHED = YES (conceptually correct: input-derived, non-learnable)
CE_GRADIENT_RECIPIENTS = view.net only (measured; all others zero, x.grad None)
SELF_EDGES_IN_PAPER_CE = AMBIGUOUS (notation includes; prose silent; retained)
GLOBAL_PHI_THETA_OWNERSHIP_CERTIFIED_IN_STAGE4 = NO
DEFERRED_TO_STAGE8 = YES
ALL_STAGE4_INTERNAL_TESTS = PASS (20/20)
SAFE_TO_FREEZE_STAGE4 = YES
```

No confirmed error in our code → zero edits. Proposed minimal fix: none.
Recorded for later stages: the one-sided M target distribution (Stage 8 training-dynamics
context); authors-vs-paper M source discrepancy (write-up material).
