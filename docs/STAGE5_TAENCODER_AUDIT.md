# STAGE 5 — TAENCODER / GAMMA / BETA / LAMBDA / ATTENTION FUSION AUDIT

Date: 2026-08-17 · Stages 1-4 frozen · Seed 42 · Read-only (zero production edits) · No epochs, no accuracy.

## 1. Paper vs authors vs ours

| quantity | paper | authors' released code | our code | verdict |
|---|---|---|---|---|
| X_topo source | GCN(X, A) (Eq. 17) | GCN, then `F.normalize(x)` | GCN, un-normalized (deliberate) | ours closer to Eq. 17 (no normalize in paper) |
| X_atte source | Eq. 16 Transformer/Graphormer attention | DIFFormer-style linear attention (uncited), computed then DISCARDED | softmax(QK^T/√d)V, per-subject blocks | ours = PAPER-CONSISTENT Transformer interpretation (Eq. 16 does not uniquely specify the exact Q/K/V operator); authors' operator is neither cited nor softmax |
| attention equation | Transformer principle (operator not uniquely specified) | no softmax anywhere | rows sum to 1.0 (verified) | ours internally consistent softmax; paper-consistent, not uniquely paper-exact |
| attention subject isolation | — | unbatched only | verified exact | PASS |
| gamma meaning | "strength of RETAINED edges … average of all elements in the ADJACENCY matrix" → retention | epoch-level `np.random.beta(fin_reg, 1-fin_reg)` computed but unused in fusion | per-subject; see call-site table | paper = retention |
| gamma original graph | adjacency initialized all-ones → 1 | n/a (fusion dead) | `paper_literal`: FALLBACK per-subject mean(gcn_w) = signal strength; `baseline`: 1.0 | **current paper_literal deviates** |
| gamma augmented graph | mean of gate/adjacency elements | n/a | per-subject mean(gate) | ours paper-aligned (Stage 2) |
| Beta parameters | Eq. 18 PRINTS the mathematical Beta FUNCTION integral B(x,y) with printed parameter order (γ, 1−γ); sampling from a Beta DISTRIBUTION is an implementation INFERENCE (supported by the Mixup-style prose and the authors' own `np.random.beta` call) | never executed in fusion | "literal": B(γ,1−γ); "reversed": B(1−γ,γ) | printed order = (γ,1−γ) — call it PAPER_PRINTED_PARAMETER_ORDER, not "literal Eq.18 distribution" |
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

γ ∈ {0.1,0.25,0.5,0.75,0.9}: B(γ,1−γ) → E[λ]=γ exactly (sampled 0.0999-0.9028);
B(1−γ,γ) → E[λ]=1−γ.

CORRECTION (Eq. 18 nuance): the paper literally prints the mathematical Beta
FUNCTION integral B(x,y) while describing Mixup-style randomness. Interpreting λ as a
sample from a Beta DISTRIBUTION is therefore an **implementation inference**, supported
by the Mixup text and the authors' own `np.random.beta` call — B(γ,1−γ) is the
**PAPER_PRINTED_PARAMETER_ORDER**, not a "literal Eq.18 distribution". The PRINTED
parameter order (γ,1−γ) implies attention grows WITH retention; the prose ("as
connectivity diminishes … focus more on global information") implies the opposite,
E[λ]=1−γ — and the authors' unused `np.random.beta(fin_reg, 1-fin_reg)` puts the DROP
rate (1−γ) first, siding with the prose. **PRINTED ORDER AND PROSE CONTRADICT; both
recorded; not resolved by intuition.** Authors' executable code: no Beta reaches
fusion at all.

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

CORRECTION (authors-literal): `enable_attention_mix=False` is **NOT** authors-literal.
The released TAEncoder executes `x_trans = F.normalize(x_trans)` **and
`x = F.normalize(x)`** before the commented-out fusion, then pools the NORMALIZED
GCN branch; our mix-off path returns the RAW (un-normalized) X_topo. Same "attention
discarded" property, different final representation.
`CURRENT_OLD_MIX_OFF_WAS_AUTHORS_LITERAL = NO` — a true `authors_release` profile
must normalize both branches and discard attention.

## 20. Proposed profile separation (PROPOSAL ONLY — not implemented)

- `paper_printed`: γ=retention (orig 1, aug mean(gate)); λ~B(γ,1−γ) (printed parameter
  order); fusion X_topo+λ·X_atte (raw).
- `paper_intent`: same but B(1−γ,γ) — the prose reading, additionally supported by the
  authors' unused `np.random.beta(fin_reg, 1-fin_reg)` (drop rate first).
- `authors_release`: normalize BOTH branches, discard attention, pool normalized X_topo.
  (NOT the old mix-off path, which returned raw X_topo — see the correction above.)
- `abide_stable_legacy`: today's stabilized behavior, renamed away from "paper_literal"
  (the name overstated; it was printed-parameter-order only, with signal-strength γ_orig).
Implemented in the FINAL PROFILE IMPLEMENTATION section below.

## Audit-phase block (historical -- superseded by FINAL PROFILE IMPLEMENTATION below)

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

---

# STAGE 5 — FINAL PROFILE IMPLEMENTATION (2026-08-17)

Baseline: `46940f5` (+ doc corrections `387fd67`). Seed 42 diagnostics, B=2 real
subjects (+ per-probe fresh encoders). NO epochs, NO accuracy. Stages 1-4 untouched.

## The four explicit profiles (`--tae_profile`, main + nested, one shared builder)

| profile | gamma (orig / aug) | lambda | fusion | eval lambda |
|---|---|---|---|---|
| `paper_printed` | 1.0 explicit / mean(gate) | ~Beta(γ,1−γ) — PRINTED parameter order | EXACTLY `X_topo + λ·X_atte` (raw X_atte; no normalize, no ‖x‖ multiplier) | γ_safe (γ=1 → 0.9999: attention ~fully present) |
| `paper_intent` | 1.0 explicit / mean(gate) | ~Beta(1−γ,γ) — prose + authors' drop-rate-first `np.random.beta` | same printed Eq. 19 | 1−γ_safe (γ=1 → 1e-4: attention ~absent on original views) |
| `authors_release` | unused | none reaches fusion | `normalize(X_topo)`; attention computed, normalized, DISCARDED | n/a (deterministic) |
| `abide_stable_legacy` | old `gamma_mode` semantics incl. signal-strength fallback | old `beta_convention` | old stabilized `X_topo + λ·‖X_topo‖·normalize(X_atte)` | old semantics |

Contracts: paper profiles REQUIRE explicit finite gamma — `gamma=None` and non-finite
gamma both RAISE (no signal-strength fallback, no `nan_to_num`; both remain
legacy-only). Epsilon clamp [1e-4, 1−1e-4] applied ONLY to make Beta shape parameters
valid. Training samples λ per subject; evaluation uses the deterministic expectation
(project reproducibility correction — the paper does not specify eval-time sampling).
`paper_printed`/`paper_intent` + `enable_attention_mix=False` raises (contradiction).
Rename: retired `--gamma_mode paper_literal` → `legacy_signal_literal` (identical
behavior; the old name overstated paper fidelity); the retired name FAILS LOUDLY with
a rename message; all 11 campaign/nested scripts updated. Nested CV: `--tae_profile`
recorded in result JSON; filename tag appended ONLY for non-legacy profiles, so all
old saved-run filenames and behavior are unchanged.

## Test results — 63/63 PASS (+ authors cross-check)

- **paper_printed / paper_intent** (19 checks each): builder returns γ_orig=1.0
  explicitly; augmented γ == per-subject mean(gate); eval λ exact (γ=0.2 →
  printed 0.2 / intent 0.8; γ=0.8 → printed 0.8 / intent 0.2); output BITWISE ==
  `X_topo + λ·X_atte` via seeded mirror-forward re-execution (proves Beta parameters
  AND that no normalize(X_atte)/‖x‖ multiplier enters fusion — the legacy-style
  variant provably differs); γ=None RAISES; NaN γ RAISES; train-mode bitwise mirror
  with Beta(γ_safe,1−γ_safe) resp. Beta(1−γ_safe,γ_safe); per-subject λ [B] finite in
  [0,1]; backward finite; attention branch gradient live; subject isolation; eval
  repeatable bitwise.
- **authors_release** (9 checks): node representation BITWISE == `F.normalize(X_topo)`;
  pooled matches; node norms all 1 (unlike old mix-off raw X_topo); scrambling ALL
  attention parameters leaves output bitwise unchanged; attention parameters receive
  ZERO gradient; γ ∈ {0.1, 0.9, None} outputs bitwise identical; isolation; finite.
- **authors cross-check**: our `authors_release` vs the authors' OWN released
  TA_encoder.py, instantiated from their repo clone in its own process, identical
  weights (state_dict transfer: 0 missing / 0 unexpected), single real subject,
  |PCC|-clamped edges, their `add_self_loops=True` set to False post-construction
  (our edge_index already carries explicit self-loops — the one harmonization
  architecture requires): **pooled and node outputs BITWISE EQUAL (max diff 0.0)**.
  Finding en route: the release does not import as shipped — `convs/__init__.py`
  requires `gine_conv.py`, absent from the release (stub-shimmed in the harness;
  same public-source provenance pattern as the broken released ToyNet forward).
- **abide_stable_legacy** (7 checks): pristine-`46940f5` worktree vs post-refactor
  code, same probe script, per-probe fresh same-seed encoders, deep-copied state
  dicts: constructed weights (24 tensors incl. BN stats), eval explicit-γ, eval
  γ=None fallback, seeded train-mode forward, `get_embeddings`, and the old
  mix-off path — **all BITWISE EQUAL**. (A first comparison failed spuriously:
  the capture had saved `state_dict()` by reference and its own train-mode probe
  mutated the saved BN running stats; the v2 methodology above eliminates this.)
- **guards + integration** (6 checks): retired name fails loudly; contradiction
  guard; both parsers expose all four profiles; both entry points thread
  `tae_profile=args.tae_profile` into the SAME shared `build_model_and_view_learner`;
  encoders of BOTH networks (Θ model + Φ view) carry the requested profile.
- **shared-path forward/backward** (3 checks): one 2-subject `train_one_epoch` step
  under `paper_intent`: model_loss −3.3279, view_loss −12.9785, all gradients of
  both networks finite, γ_orig=1.0 used explicitly.

## FINAL STAGE-5 BLOCK

```
EQ18_PRINTS_BETA_FUNCTION_NOT_DISTRIBUTION =
    YES

BETA_DISTRIBUTION_INTERPRETATION =
    SUPPORTED_BY_MIXUP_TEXT_AND_AUTHORS_NP_RANDOM_BETA

PAPER_PRINTED_BETA_ORDER =
    (gamma,1-gamma)

PAPER_INTENT_BETA_ORDER =
    (1-gamma,gamma)

PAPER_INTENT_SUPPORT =
    PAPER_PROSE_PLUS_AUTHORS_DROP_RATE_BETA

PAPER_GAMMA_ORIGINAL =
    1

PAPER_GAMMA_AUGMENTED =
    MEAN_GATE

PAPER_EQ19 =
    X_TOPO_PLUS_LAMBDA_X_ATTE

PAPER_PRINTED_PROFILE =
    PASS

PAPER_INTENT_PROFILE =
    PASS

AUTHORS_RELEASE_PROFILE =
    PASS  (bitwise vs authors' own released module, identical weights)

ABIDE_STABLE_LEGACY_REPRODUCED =
    PASS  (bitwise vs pristine 46940f5, all probes)

CURRENT_OLD_MIX_OFF_WAS_AUTHORS_LITERAL =
    NO

CURRENT_ATTENTION_OPERATOR_STATUS =
    PAPER_CONSISTENT_NOT_UNIQUELY_SPECIFIED

ALL_STAGE5_PROFILE_TESTS =
    PASS  (63/63 + authors cross-check)

SAFE_TO_FREEZE_STAGE5 =
    YES
```

---

# STAGE 5 — FINAL HARDENING (2026-08-17, post-implementation)

No architecture redesign. Profiles, Stages 1-4, and Θ/Φ ownership untouched. No epochs.

**1. Paper-profile layerwise bug (real, untested configuration).** With
`pooling_type='layerwise'`, xpool is built from `xs` — the PRE-FUSION per-layer GCN
tensors — so a paper profile's graph embedding would silently bypass the Eq. 19 fused
representation. The paper does not specify this project-specific pooling mode, so no
paper-layerwise semantics were invented: construction now FAILS LOUDLY for
`paper_printed`/`paper_intent` + non-standard pooling. Guard is scoped: legacy and
authors_release still accept layerwise (unchanged behavior).

**2. Gamma range validation.** Paper profiles previously rejected None/NaN/Inf but
silently clamped any finite out-of-range gamma. Now: finite gamma outside [0, 1]
RAISES (a retention ratio out of range is a caller bug, not a boundary case); the
epsilon clamp [1e-4, 1−1e-4] remains ONLY for the legitimate boundary values 0/1
required by Beta shape validity. Verified: γ=0 accepted (output bitwise == γ=eps),
γ=1 accepted (bitwise == γ=1−eps), and −0.1 / 1.1 / NaN / Inf all raise.

**3. Stale comment fixed (comment-only).** The legacy mix-off branch no longer claims
"matches the original released code exactly"; it now states: historical pre-Stage5
mix-off behavior — attention discarded, but unlike `authors_release` it does NOT
normalize X_topo before pooling.

**4. get_attentions guard.** `TransConv.get_attentions` has zero production callers
(verified repo-wide) and no batch argument — single-graph logic that would silently
mix subjects. It now fails loudly unless the input is exactly one 90-ROI subject
([90, F]); batched attention-map extraction deliberately NOT designed here.
Verified: [90,3] works ([layers, 90, 90]); [180,3], [45,3], and 1-D all raise.

**5. paper_intent boundary — documented, NOT changed.** Original/eval γ=1 →
γ_safe=0.9999 → E[λ]=1e-4: `paper_intent` is topology-dominant for the
unaugmented/evaluation view, EXPECTED under the prose reading (less retained
connectivity → more global attention). Augmented graphs with lower γ receive
materially larger attention weight — measured: ‖X_update − X_topo‖ = 0.0058 at γ=1
vs 40.54 at γ=0.3 (≈7000×).

**6-7.** All four profiles kept; `abide_stable_legacy` remains regression-
reproducibility only, never a scientific primary. CLI defaults unchanged
(`abide_stable_legacy`) for backward compatibility, with a TODO in BOTH parsers:
Stage 10/orchestration must define one explicit canonical corrected configuration
and never rely on argparse defaults for final experiments.

**8. Tests.** Hardening battery 28/28 (layerwise guards, gamma range, standard-pooling
BITWISE regression vs pre-hardening `63db75d` capture incl. seeded train forwards,
get_attentions guard, boundary-lambda asserts) + full previous profile battery
re-run 63/63 (legacy probes re-captured from post-hardening code, still bitwise ==
pristine 46940f5) + the 2-subject paper_intent `train_one_epoch` step reproducing
identical losses (−3.3279 / −12.9785). Total 91/91.

## FINAL HARDENING BLOCK

```
PAPER_PROFILE_STANDARD_POOLING =
    PASS  (bitwise unchanged vs pre-hardening, eval + seeded train)

PAPER_PROFILE_LAYERWISE =
    UNSUPPORTED_FAIL_LOUD

PAPER_GAMMA_RANGE_VALIDATION =
    PASS  (finite out-of-[0,1] raises; 0/1 accepted then eps-clamped)

PAPER_INTENT_ORIGINAL_EVAL_LAMBDA =
    ~1e-4

PAPER_INTENT_ORIGINAL_ATTENTION_STATUS =
    MINIMAL_BY_DESIGN  (prose reading; augmented low-gamma views get ~7000x more)

GET_ATTENTIONS_BATCH_STATUS =
    SINGLE_SUBJECT_GUARDED  (diagnostic only, zero production callers)

ALL_STAGE5_PROFILE_TESTS =
    PASS  (63/63 profile + 28/28 hardening = 91/91)

SAFE_TO_FREEZE_STAGE5 =
    YES
```

---

# STAGE 5 — FREEZE (2026-08-17, final defensive guard)

One guard added, nothing else: `paper_printed`/`paper_intent` gamma now enforces the
exact shape contract — scalar (expanded to [num_graphs]) or 1-D [num_graphs]; any
other shape ([B,1], [1,B], wrong length) RAISES before it could silently broadcast
into `lambda_[batch]` and corrupt per-subject fusion. All previous checks retained
(finite, [0,1] range, eps Beta-boundary clamp, layerwise guard, get_attentions guard,
all four profiles).

Tests: 6 shape tests per paper profile (scalar 1.0 PASS, [2] PASS, [2,1]/[1,2]/[1]/[3]
RAISE) = 12/12 · full profile battery 63/63 (2-subject paper_intent step losses
identical: −3.3279/−12.9785) · hardening battery 28/28 (incl. standard-pooling
BITWISE regression vs pre-hardening capture) — **103/103**.

```
PAPER_GAMMA_SHAPE_VALIDATION =
    PASS

STAGE5_FINAL_TESTS =
    PASS  (103/103)

SAFE_TO_FREEZE_STAGE5 =
    YES

STAGE5_STATUS =
    FROZEN
```
