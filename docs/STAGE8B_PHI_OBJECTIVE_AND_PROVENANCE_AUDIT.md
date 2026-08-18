# STAGE 8B — Φ IDENTITY, Φ OBJECTIVE, AND RUNTIME/CACHE PROVENANCE AUDIT

Date: 2026-08-18 · Stages 1–6 frozen · Stage 7 complete · Stage 8A at `e142c75` ·
**DIAGNOSTIC ONLY: zero production edits, no epochs, no accuracy, no nested CV, no tuning.**

Question answered: *which executable parameters constitute Φ in our adaptation, what objective
should Φ optimize, and is our Phase-1 update coherent with the strongest defensible reading of
the paper?*

Nothing in this stage reuses `stage7_harness.py` or `stage8a_harness.py`. Every measurement
instruments the REAL production modules, with a runtime assertion that each resolves under the
repository root.

---

## 0. RUNTIME / CACHE / REPOSITORY PROVENANCE GATE

```
pwd                      /users/3171356m/muhammad/GraSTIACL
git rev-parse --show-toplevel   /users/3171356m/muhammad/GraSTIACL
git branch --show-current       main
git rev-parse HEAD              e142c75f54939dd6b2be496ad8a0fdfa3868d71f
git rev-parse origin/main       e142c75f54939dd6b2be496ad8a0fdfa3868d71f   (identical)
git status --short (tracked)    clean — 0 modified tracked files
```

Stage commits verified as **ancestors of HEAD** via `git merge-base --is-ancestor`:
`3b1355a` + `d761219` (Stage 6E) · `8bed3f7` + `f13a69c` (Stage 7) · `e142c75` (Stage 8A). No
branch switch, no reset.

**Python runtime** — `cwd = <repo>`, `sys.executable =
/users/3171356m/miniconda3/envs/grastiacl/bin/python3`, `sys.path[0] = ''`.

| module | resolved `__file__` | verdict |
|---|---|---|
| `datasets` (pkg) | `<repo>/datasets/__init__.py` | OK — **ours, not HuggingFace `datasets`** |
| `datasets.Dataset` | `<repo>/datasets/Dataset.py` | OK |
| `unsupervised.training` | `<repo>/unsupervised/training.py` | OK |
| `unsupervised.learning.ginfominmax` | `<repo>/unsupervised/learning/ginfominmax.py` | OK |
| `unsupervised.encoder.TA_encoder` | `<repo>/unsupervised/encoder/TA_encoder.py` | OK |
| `unsupervised.view_learner` | `<repo>/unsupervised/view_learner.py` | OK |
| `unsupervised.learning.GraSTI` | `<repo>/unsupervised/learning/GraSTI.py` | OK |

No module resolved to another clone, to `/tmp/claude...`, to a Stage-7 harness, or to
site-packages.

**SHA256 of production sources**

```
e3bd78e048ffd71a730c18bf24032993b37b8258c8c8c86ed76350e45ae0232a  unsupervised/training.py
4328b51598fb2185c2d76f85a56b262161d0c647e5666638f1fe13cca1b71044  unsupervised/learning/ginfominmax.py
ef133124434e3ae4b60ae03d8392b230bd68a54365ff22f4d45a0a0e1b2d6c1a  datasets/Dataset.py
b529ab514970dc9171b82a884463660a5fcf46cf1779c76b67cb873c862fcbc9  unsupervised/encoder/TA_encoder.py
7bf4dae92e32aac4e26bbd1ed7d596831c96e9aeeb4f79b2617f9c1c52ca4d7f  unsupervised/view_learner.py
be6d7de59ba27288895608ce70f5caa426cbc254a55e66f0b25b7a0bec5e3181  unsupervised/learning/GraSTI.py
f6f3f64c7bfd2f026c2ddcd98973a76f791e89873bb3a0f42fcd558179e53f3b  unsupervised/convs/gcn_conv.py
9d90329f9802ba4d85148d7fd9f32cddd90c6353e07380c271d0ec6770f27422  GraSTIACL.py
```

**Data gate — both modes instantiated independently**

| | `alff_new_z` | `alff_m1_z` |
|---|---|---|
| source NPZ | `alff_new/non_combat/alff_new.npz` | `ALFF_func_proc/method1/alff_roi_first.npz` |
| NPZ sha256 | `69da61aa…c5a22b` (mtime 2026-08-15 19:46) | `647a1d87…cc3cf4` (mtime 2026-08-17 23:43) |
| processed cache | `…/processed/data_alff_new_z.pt` | `…/processed/data_alff_m1_z.pt` |
| cache sha256 | `392b3b18…9e414e` (mtime 2026-08-18 17:47) | `6895e9ea…713264` (mtime 2026-08-18 17:47) |
| loaded N | **956** (expected 956) ✓ | **954** (expected 954) ✓ |
| unique IDs / duplicates | 956 / **0** | 954 / **0** |
| first, last ID | `CMU_a_0050642`, `Yale_0050578` | `CMU_a_0050642`, `Yale_0050578` |
| `[N,90,3]` contract | ✓ | ✓ |
| all finite | ✓ | ✓ |
| sampled per-band mean≈0 / std≈1 | ✓ | ✓ |
| **content proof** | **x == z(NPZ[id]) BITWISE 956/956** | **BITWISE 954/954** |

`Dataset.py` is **not** newer than either cache, and — more importantly — staleness was decided
by **content**, not timestamps: each cached `x` was compared bitwise against the per-band
z-score recomputed from the source NPZ. `alff_m1_z`'s excluded set is **exactly**
`{CMU_b_0050669, Leuven_1_0050706}` (the documented zero-ROI cases) and both are absent from
the loaded cohort. Control `data_alff_paper.pt` recorded at sha256 `925c3488…1642c2`.

```
RUNTIME_REPO_CORRECT             = YES
RUNTIME_MODULE_PATHS_CORRECT     = YES
DATASET_MODE_ROUTING_CORRECT     = YES
PROCESSED_CACHE_STALE_OR_WRONG   = NO
SUBJECT_COHORT_CORRECT           = YES
SAFE_TO_INTERPRET_STAGE8B_RESULTS = YES
```

---

## 1. SIGN CONVENTION — settled algebraically and numerically BEFORE any directional language

**Numeric mapping.** Eq. 20 printed:
`Î_R = (1/m) Σ_i log[ exp(sim(z_i1,z_i2)) / Σ_{i'≠i} exp(sim(z_i1,z_i'2)) ]`.
Evaluating our `calc_loss(temperature=1, sym=False)` against a hand implementation of Eq. 20:

| case | I_R (Eq. 20) | calc_loss(τ=1,sym=F) | I_R + L_CL |
|---|---|---|---|
| randn B8 d32 | −2.00422096 | +2.00422096 | **0.00e+00** |
| randn B16 d64 | −2.71080875 | +2.71080875 | **0.00e+00** |
| corr B8 | −0.98902953 | +0.98902953 | **0.00e+00** |
| scaled B8 | −2.05915570 | +2.05915546 | 2.4e−07 |

**`L_CL = −I_R` exactly.** (Production adds τ=0.2 and `sym=True`; Stage 7 established both are
inherited from the authors' release and absent from printed Eq. 20. They are monotone /
symmetrized variants and do not affect the sign relation.)

**Executable Phase-1 algebra** (`training.py`):
```
view_loss = L_CL − λ_CE·L_CE − λ_REG·L_REG − λ_KLD·L_KLD
(-view_loss).backward() ; view_optimizer.step()
    ⇒ Adam MINIMIZES (−view_loss) ⇒ MAXIMIZES view_loss
    ⇒ Φ ASCENDS L_CL, DESCENDS L_CE, DESCENDS L_REG, DESCENDS L_KLD
Since L_CL = −I_R:  ASCEND L_CL  ≡  MINIMIZE I_R.
```

**Paper side.** Eq. 13 `Î_N = mean[L_CE − β·D_KL]`; Eq. 14 `I_N = mean[L_CE + β·D_KL]`
(explicitly rewritten "to find a more accurate information bottleneck", so Eq. 14 supersedes);
Eq. 15 `min_Φ max_Θ I_N`; Eq. 21 `min_Ψ max_Ω I_R`; Eq. 22 `min_{Φ,Ψ} max_{Θ,Ω} (I_R + I_N)`.

| quantity | paper expression | authors executable sign (for Φ) | our executable sign (for Φ) | interpretation |
|---|---|---|---|---|
| I_R | Eq. 20; Φ **minimizes** it (Eq. 22) | no gradient path to Φ at all | **minimized** (L_CL ascended) | **MATCH** |
| L_CL | `= −I_R` (proven above) | unreachable from Φ | **ascended** | **MATCH** |
| CE | Eq. 14 term; Φ **minimizes** (Eq. 15) | routed to Θ, not Φ | **descended** | **MATCH** |
| KLD | Eq. 14 `+β·D_KL`; Φ **minimizes** | descended by Φ | **descended** | **MATCH** |
| REG | **absent from the paper** | descended by Φ | **descended** | PROJECT_ADAPTATION |

Hence the paper-mapped Φ objective is `J_Φ = I_R + I_N`, **minimized**, i.e.
`J_Φ ≈ −L_CL + λ_CE·L_CE + λ_KLD·L_KLD`, and the executable
`(−view_loss) = J_Φ + λ_REG·L_REG`. **Phase 1 minimizes exactly the paper's Φ objective plus
the project-added REG term.**

```
PURE_ADVERSARIAL_CL_EXPECTATION : Φ should make L_CL INCREASE  (Eq.22 min_Φ I_R)
PAPER_IN_EXPECTATION            : Φ should make CE  DECREASE   (Eq.14/15)
                                  Φ should make KLD DECREASE   (Eq.14/15)
FULL_PAPER_PHI_DIRECTION_RECOVERABLE = YES  (all three terms recoverable and unambiguous for Φ)
```

⚠ Separately, and **unchanged from Stage 7/8A**: the Θ/Ψ assignment *is* inconsistent — Eq. 15
calls the node-vector encoder **Θ (max)** while Eq. 21 calls the TAE **Ψ (min)**, and the TAE
*is* the node-vector encoder. So `PAPER_PARAMETER_OWNERSHIP = AMBIGUOUS` **for Θ/Ψ**, while Φ
itself is fully determined.

---

## 2. WHAT IS Φ? — paper vs authors vs ours

⚠ **STAGE-8A CORRECTION.** Stage 8A asserted the paper defines Φ as "the augmenter GNN and
MLP" and concluded that both being dead was an ownership bug. Re-reading **p.3** shows that
phrasing belongs to the **AD-GCL background paragraph**, not to GraSTI's own Φ:

> "…Φ corresponds to the learnable parameters of the augmenter GNN and multilayer perceptron
> (MLP)… **However, this method makes it difficult to determine whether redundant information
> in the node vectors has been sufficiently discarded.**"

GraSTI then explicitly departs: *"In contrast, our strategy first perform graph node IB by
leveraging both the edge weight matrix W and the nodal feature inner product matrix M."*
GraSTI's **own** definition of Φ is at Eq. 15: **"Φ are the network parameters encoding the
adjacency matrix A."** That is the node-IB / VIB network mapping W → A — which in our code is
`view.net`.

| candidate | what it does | paper status | authors status | our status |
|---|---|---|---|---|
| `view.encoder` + `view.mlp_edge_model` | AD-GCL-style augmenter: GNN embeddings → concat → MLP → Bernoulli logits | **DEAD_SCAFFOLDING** — describes AD-GCL, the method GraSTI explicitly departs from | present, unused for the gate | present, **zero gradient from every term** (grad is `None`) |
| `model.net` | ToyNet/VIB on the Θ side | **INCOMPATIBLE** with Eq. 15 (Φ must be separate from Θ) | **AUTHORS_RELEASE** — this is their actual gate producer | never executed (`dyn_weight=None` at all four `model()` call sites) |
| `view.net` (ToyNet/VIB) | W (+ dynamic W) → VIB → adjacency logits → Eq. 12 gate → A | **PAPER_SUPPORTED** — "the network parameters encoding the adjacency matrix A" (Eq. 15), realized by the node-IB of Eq. 4/13/14 | present but the contrastive term never reaches it | **PROJECT_ADAPTATION that is also PAPER_SUPPORTED — our active Φ** |

So the dead `view.encoder`/`view.mlp_edge_model` are **inherited AD-GCL scaffolding**, not a
missing paper requirement. `PAPER_GNN_MLP_PATH_NEEDS_RESURRECTION = NO`.

---

## 3. EXECUTABLE GRADIENT ROUTING (autograd, `allow_unused=True`, 8 seeds — identical every seed)

Proven chain for OUR HEAD:
`view.net → edge logits → Binary-Concrete/logistic gate → W_aug = PCC·gate → augmented
representation → L_CL`.

| group | L_CL | L_CE | L_REG | L_KLD |
|---|---|---|---|---|
| `view.net/FC_mean` | EXISTS (1.269e−02) | EXISTS (3.084e−02) | EXISTS (8.588e−02) | EXISTS (1.984e+01) |
| `view.net/FC_var` | EXISTS (3.216e−05) | EXISTS (7.421e−05) | EXISTS (3.243e−05) | EXISTS (1.035e+02) |
| `view.net/encode` | EXISTS (1.067e−02) | EXISTS (3.812e−02) | EXISTS (6.768e−02) | EXISTS (1.123e+02) |
| `view.net/decode` | EXISTS (1.288e−02) | EXISTS (3.018e−02) | EXISTS (8.607e−02) | **NOT_EXECUTED/DETACHED** (KLD depends only on μ,σ) |
| `view.encoder` | NOT_EXECUTED/DETACHED | NOT_EXECUTED/DETACHED | NOT_EXECUTED/DETACHED | NOT_EXECUTED/DETACHED |
| `view.mlp_edge_model` | NOT_EXECUTED/DETACHED | NOT_EXECUTED/DETACHED | NOT_EXECUTED/DETACHED | NOT_EXECUTED/DETACHED |

```
OUR_ACTIVE_PHI_RECEIVES_CL_GATE_GRADIENT = YES  (‖∂L_CL/∂view.net‖ ≈ 2.1e−02, nonzero, 4/4 submodules)
AUTHORS_PHI_RECEIVES_CL_GATE_GRADIENT    = NO   (see §9 — measured 0.000000e+00, 0/32 tensors)
```

---

## 4. SAME-SURFACE DIAGNOSTIC — how equality is proven

All four terms are differentiated **from ONE shared computational graph** via
`torch.autograd.grad(term, params, retain_graph=True)`. They therefore *cannot* see different
tensors — CL, CE, REG and KLD read the identical `edge_logits`, the identical 8100-per-subject
Binary-Concrete ε draw, the identical `gate`, `W_aug`, `γ_aug`, `x` and `x_aug` **by
construction**, not by re-seeding. Additionally pinned: module modes reproduce production
Phase 1 exactly (`model.eval()`, `view_learner.train()`), the ToyNet VIB `randn_like` seed
(it is *not* training-gated), the ε seed, and the forward seed shared by the original and
augmented passes. Same weights, same batch, same subject ordering, same ALFF/PCC/dyn-PCC.

---

## 5. SCALED GRADIENT DECOMPOSITION (mean over 8 seeds × 8 distinct batches)

| group | #par | raw CL | raw CE | raw REG | raw KLD | **scl CL** | **scl CE** | **scl REG** | **scl KLD** | scl FULL | cos(g_CL, u_FULL) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `view.net/FC_mean` | 52,000 | 1.27e−02 | 3.08e−02 | 8.59e−02 | 1.98e+01 | 1.27e−02 | 6.17e−02 | 1.72e−02 | 5.95e−02 | 9.75e−02 | −0.0308 |
| `view.net/FC_var` | 52,000 | 3.22e−05 | 7.42e−05 | 3.24e−05 | 1.04e+02 | 3.22e−05 | 1.48e−04 | 6.49e−06 | **3.11e−01** | 3.11e−01 | +0.0155 |
| `view.net/encode` | 27,264 | 1.07e−02 | 3.81e−02 | 6.77e−02 | 1.12e+02 | 1.07e−02 | 7.62e−02 | 1.35e−02 | **3.37e−01** | 3.85e−01 | −0.0007 |
| `view.net/decode` | 72,090 | 1.29e−02 | 3.02e−02 | 8.61e−02 | 0 | 1.29e−02 | 6.04e−02 | 1.72e−02 | 0 | 6.17e−02 | −0.0661 |
| `view.encoder` | 4,672 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| `view.mlp_edge_model` | 4,225 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| `model.net` | 203,354 | — module never executed in our routing; zero gradient from every term (Stage 8A) | | | | | | | | | |

**Raw KLD norms are 3–4 orders of magnitude larger than CL, but after the production
coefficient λ_KLD = 0.003 the applied norms are comparable** (e.g. FC_mean: KLD 5.95e−02 vs
CL 1.27e−02). Reporting raw norms alone — as Stage 8A did — overstates KLD by ~10³.

---

## 6. UPDATE-DIRECTION COSINES (u = the direction parameters actually move), 8 seeds

| pair | mean ± sd | range |
|---|---|---|
| **cos(u_FULL, u_PURE_ADVERSARIAL_CL)** | **−0.0062 ± 0.0920** | [−0.1379, +0.1945] |
| **cos(u_FULL, u_PAPER_Φ_OBJECTIVE J=I_R+I_N)** | **+0.9985 ± 0.0004** | [+0.9976, +0.9989] |
| cos(u_PAPER_J, u_PURE_CL) | +0.0071 ± 0.0919 | [−0.1313, +0.2057] |
| cos(g_CL, signed-scaled g_CE) | −0.2020 ± 0.1249 | [−0.3864, −0.0256] |
| cos(g_CL, signed-scaled g_REG) | −0.2417 ± 0.0797 | [−0.3623, −0.1169] |
| **cos(g_CL, signed-scaled g_KLD)** | **+0.0067 ± 0.0574** | [−0.0961, +0.0851] |
| cos(sCE, sKLD) | +0.2641 ± 0.0861 | [+0.0980, +0.3865] |
| cos(sCE, sREG) | +0.0087 ± 0.0880 | [−0.1023, +0.1448] |
| cos(sREG, sKLD) | +0.0140 ± 0.0548 | [−0.0837, +0.1030] |

‖u_FULL‖ 5.10e−01 · ‖u_CL‖ 2.10e−02 · ‖u_J‖ 5.09e−01.

**The headline.** Our actual Φ update is essentially *collinear with the paper's own Φ
objective* (cos **+0.9985**) and essentially *orthogonal* — not opposed — to a pure-CL
adversarial direction (cos −0.006). Crucially, **the paper's own composite objective is
itself near-orthogonal to pure CL** (cos +0.007). The near-orthogonality is therefore a
property of the paper's multi-objective Φ, **not** an artifact of our implementation.

Per the spec's explicit warning: **KLD is magnitude-large but directionally ORTHOGONAL to CL**
(cos +0.0067), so it does *not* "dominate CL" directionally. **CE is the only term that both
carries larger applied magnitude (≈3–5× CL) and points against CL (cos −0.20).**

---

## 7–8. TINY-STEP CAUSAL ABLATION — exact directional derivatives

⚠ **Method correction, disclosed.** A first finite-difference pass at η ∈ {1e−6…5e−4} produced
ΔL_CL values of exactly ±2.98e−08 / ±5.96e−08 — the **float32 ULP of L_CL ≈ 2** — and
first-order ratios as absurd as +105.96 and −13.63. Those numbers were unresolvable noise and
were **discarded**. For an SGD step `Φ ← Φ − η·g_arm` the exact first-order effects are
`dL_CL/dη = −⟨g_CL, g_arm⟩` and `dJ/dη = −⟨g_J, g_arm⟩`, computed analytically below.

| arm | dL_CL/dη (mean ± sd) | sign>0 | dJ_paper/dη (mean ± sd) | sign<0 | cos(update, g_CL) |
|---|---|---|---|---|---|
| **A pure CL** | **+5.85e−04 ± 7.2e−04** | **8/8** | −2.67e−04 ± 1.4e−03 | 4/8 | +1.0000 ± 0.0000 |
| B CL+CE | +1.88e−04 ± 8.2e−04 | **3/8** | −3.11e−02 ± 2.0e−02 | 8/8 | +0.0015 ± 0.2415 |
| C CL+REG | +4.51e−04 ± 6.8e−04 | **8/8** | −5.03e−04 ± 1.3e−03 | 5/8 | +0.4094 ± 0.2460 |
| D CL+KLD | +6.64e−04 ± 1.3e−03 | 6/8 | −2.43e−01 ± 1.1e−01 | 8/8 | +0.0559 ± 0.0838 |
| E CL+CE+REG | +5.42e−05 ± 8.0e−04 | 2/8 | −3.14e−02 ± 2.0e−02 | 8/8 | −0.0561 ± 0.2470 |
| F CL+CE+KLD | +2.67e−04 ± 1.4e−03 | 4/8 | −2.74e−01 ± 1.3e−01 | 8/8 | +0.0071 ± 0.0919 |
| **G FULL (current)** | **+1.33e−04 ± 1.4e−03** | **3/8** | **−2.74e−01 ± 1.3e−01** | **8/8** | −0.0062 ± 0.0920 |

Readings:

- **`PURE_CL_STEP_MAKES_TASK_HARDER = YES`** — arm A raises L_CL in **8/8** seeds.
- **`FULL_STEP_MOVES_PURE_CL_IN_EXPECTED_ADVERSARIAL_DIRECTION = MIXED`** — arm G is positive
  in only 3/8, mean +1.3e−04 with sd 1.4e−03 (i.e. indistinguishable from zero).
- **`WHICH_TERM_CAUSES_DIRECTIONAL_REVERSAL = CE`** — adding REG alone preserves 8/8 (C); KLD
  alone is mild (6/8); adding **CE** collapses it to 3/8 (B) and 2/8 (E). This matches §6: CE
  is the only auxiliary that is both larger and directionally opposed.
- **§8, the decisive reframe: the FULL step improves the paper's Φ objective in 8/8 seeds
  (dJ/dη = −2.74e−01), while its effect on pure L_CL is statistically null.** Conversely arm A
  (pure CL only) improves the paper objective in only 4/8 — because it ignores I_N.

This is a **multi-objective trade-off within a coherent composite objective**, not a sign error.

---

## 9. AUTHORS-RELEASE REFERENCE ARM (proven executably, not inferred from return names)

Their Phase 1 passes `dyn_weight` to `model(...)`, so `GInfoMinMax` returns
`z, node_emb, logits` — **`logits` comes from `model.net`** — and the view learner's own first
return value is discarded (`_, mu, std, edge_prod = view_learner(...)`). Reconstructed with our
modules and differentiated:

```
AUTHORS_GATE_PRODUCER                = model.net (GInfoMinMax's ToyNet) — the THETA side
AUTHORS_VIEW_LEARNER_CL_GRAD_NORM    = 0.000000e+00   (0/32 tensors)   [grad→model 5.484397e+00, 32/32]
AUTHORS_VIEW_LEARNER_KLD_GRAD_NORM   = 2.124541e+02   (8/32 tensors)   [grad→model 0.000000e+00]
AUTHORS_VIEW_LEARNER_CE_GRAD_NORM    = 0.000000e+00   (0/32 tensors)   [grad→model 6.086551e-02, 10/32]
AUTHORS_RELEASE_PHI_EFFECTIVE_OBJECTIVE = minimize the KL divergence ONLY
```

Their `view_loss = L_CL − λ_KLD·KLD` is maximized, but **L_CL has no gradient path into their
view learner at all**, so their Φ is not an adversary on the contrastive objective in any
sense. Our routing through `view.net` is therefore both a deliberate deviation from their
release **and** the reading that actually realizes Eq. 15/22.

---

## 10. INIT-ONLY BIAS CHECK

Repository searched for `*.pth`/`*.pt`/`*.ckpt` outside the PyG `processed/` caches; the
`checkpoints/` directory is **empty**. No trained checkpoint of verifiable provenance exists.

```
TRAINED_STATE_STAGE8B_CHECK = NOT_AVAILABLE
```

Per the spec, **no training was launched to satisfy this section.** All Stage-8B conclusions
are therefore initialization-state conclusions; that limitation is carried explicitly.

---

## 11. NUMERICAL SAFETY (observation only — no epsilon, no clamping added)

| tensor | min norm | non-finite | <1e−8 | <1e−12 | <1e−20 |
|---|---|---|---|---|---|
| `x` (original view) | 2.503438 | 0 | 0 | 0 | 0 |
| `x_aug` (augmented view) | 69.410591 | 0 | 0 | 0 | 0 |

(Stage 7's unguarded-cosine CONCERN is unchanged and remains open; nothing here approaches it.)

---

## FINAL VERDICT BLOCK

```
RUNTIME_REPO_CORRECT = YES
RUNTIME_MODULE_PATHS_CORRECT = YES
DATASET_MODE_ROUTING_CORRECT = YES
PROCESSED_CACHE_STALE_OR_WRONG = NO
SUBJECT_COHORT_CORRECT = YES

LCL_RELATION_TO_IR = LCL_EQUALS_NEG_IR
    (exact; max |I_R + L_CL| = 2.4e-07 at tau=1/sym=False, four independent constructions)

PURE_PHI_CL_DIRECTION = INCREASE_LCL
    (Eq.22 min_Phi I_R with L_CL = -I_R; and arm A raises L_CL in 8/8 seeds)

PAPER_PHI_IDENTITY = "the network parameters encoding the adjacency matrix A" (Eq.15),
    realized by the node-IB/VIB of Eq.4/13/14 that maps the edge-weight matrix W to A.
    In our code that is view.net. The "augmenter GNN + MLP" wording on p.3 belongs to the
    AD-GCL BACKGROUND paragraph that GraSTI explicitly departs from -- it is NOT GraSTI's
    definition of Phi. (This CORRECTS Stage 8A.)

PAPER_PHI_OBJECTIVE = minimize J_Phi = I_R + I_N   (Eq.22)
    = minimize I_R (i.e. ASCEND L_CL) AND minimize I_N = mean[L_CE + beta*D_KL]
      (i.e. DESCEND CE and DESCEND KLD).   All three terms recoverable and unambiguous.

PAPER_PARAMETER_OWNERSHIP_CONSISTENT = NO
    (Phi is consistent and fully determined; the THETA/PSI assignment is not -- Eq.15 makes
     the node-vector encoder Theta/maximized while Eq.21 makes the TAE Psi/minimized, and the
     TAE IS the node-vector encoder. Unchanged from Stage 7/8A.)

PUBLIC_SPECIFICATION_UNDERDETERMINED = YES  (for Theta/Psi only; NOT for Phi)

AUTHORS_GATE_PRODUCER = model.net (GInfoMinMax ToyNet) -- the Theta side
AUTHORS_EFFECTIVE_PHI = view_learner, optimizing the KL divergence ONLY
AUTHORS_PHI_RECEIVES_CL_GATE_GRADIENT = NO   (measured 0.000000e+00 over 0/32 tensors)

OUR_GATE_PRODUCER = view.net (ToyNet/VIB), via edge logits -> Eq.12 Binary-Concrete gate
OUR_EFFECTIVE_PHI = view.net (4 submodules, 203,354 params)
    dead-but-registered: view.encoder (4,672) + view.mlp_edge_model (4,225) = AD-GCL scaffolding
OUR_ACTIVE_PHI_RECEIVES_CL_GATE_GRADIENT = YES  (||dL_CL/d view.net|| ~ 2.1e-02, 4/4 submodules)

COS_PURE_CL_VS_FULL_UPDATE = -0.0062 +- 0.0920   (range [-0.1379, +0.1945], 8 seeds)

FULL_UPDATE_PURE_CL_EFFECT = MIXED
    (dL_CL/deta = +1.33e-04 +- 1.4e-03, positive in 3/8 -- statistically indistinguishable
     from zero; the update is ORTHOGONAL to pure CL, not opposed to it)

FULL_UPDATE_PAPER_OBJECTIVE_EFFECT = COHERENT
    (cos(u_FULL, u_J) = +0.9985 +- 0.0004; dJ/deta = -2.74e-01, improving in 8/8 seeds)

WHICH_TERM_CAUSES_DIRECTIONAL_REVERSAL = CE
    (arm A 8/8 positive -> arm C (+REG) 8/8 -> arm D (+KLD) 6/8 -> arm B (+CE) 3/8,
     arm E (+CE+REG) 2/8. CE is the only auxiliary that is BOTH larger in applied magnitude
     (~3-5x CL) AND directionally opposed (cos -0.20).)

AUXILIARY_DOMINANCE = MAGNITUDE_ONLY for KLD ; BOTH for CE ; neither for REG
    (KLD applied norm is comparable-to-larger than CL but cos(g_CL, sKLD) = +0.0067 -- an
     ORTHOGONAL subspace, so it does NOT dominate CL directionally. Stage 8A's 48:1 claim was
     computed on norms only and is corrected here.)

STAGE8A_SIGN_BUG_REASSESSMENT = DOWNGRADED_TO_MULTI_OBJECTIVE_TRADEOFF
    The literal signs were always correct; Stage 8A additionally measured only dL_CL and
    called the composite non-adversarial. Measured against the paper's own Phi objective the
    same update is coherent in 8/8 seeds at cosine 0.9985. There is no sign bug.

CURRENT_VIEW_NET_AS_PHI = PAPER_SUPPORTED
    (and simultaneously a reasoned project adaptation away from the authors' release, which
     routes the gate through model.net and leaves its Phi with no contrastive gradient)

PAPER_GNN_MLP_PATH_NEEDS_RESURRECTION = NO
    (view.encoder + view.mlp_edge_model implement AD-GCL's augmenter, the approach GraSTI
     explicitly criticizes and replaces. They are DEAD_SCAFFOLDING, not a missing requirement.
     Whether to delete them is a hygiene question, not a correctness one.)

TRAINED_STATE_STAGE8B_CHECK = NOT_AVAILABLE
    (no checkpoint of verifiable provenance exists; none was trained, per the spec)

STAGE8B_CONFIRMED_OBJECTIVE_ROUTING_BUG = NO
    Phi is correctly identified, correctly connected to L_CL through the gate, and updated in
    a direction collinear (0.9985) with the paper's own Phi objective.

BEST_SUPPORTED_PHI_FOR_STAGE8C = view.net (ToyNet/VIB) -- UNCHANGED from current production
BEST_SUPPORTED_PHI_OBJECTIVE_FOR_STAGE8C =
    minimize J_Phi = I_R + I_N  ==  ascend L_CL, descend CE, descend KLD
    -- which is what the code already does, plus the paper-absent lambda_REG*L_REG term.
    The ONLY paper-unsupported component of Phi's objective is REG (PROJECT_ADAPTATION,
    justified in-code as preventing graph destruction). No change proposed here.

SAFE_TO_DESIGN_STAGE8C = YES
```

## Confirmed issues only

1. **No objective-routing bug and no sign bug.** Both Stage-8A headline verdicts are corrected
   by this stage's algebra and directional measurements.
2. **`REG` is the sole paper-absent component of Φ's objective** (PROJECT_ADAPTATION).
3. **Dead scaffolding remains dead** (`view.encoder`, `view.mlp_edge_model`, and `model.net`
   under our routing) — a hygiene matter, not a correctness one.
4. **Θ/Ψ ownership remains publicly underdetermined** (Eq. 15 vs Eq. 21) — untouched here.
5. **All conclusions are initialization-state only** (`TRAINED_STATE_STAGE8B_CHECK =
   NOT_AVAILABLE`).

## Proposed Stage-8C direction (proposal only — NOT started)

With Φ settled, the outstanding blocker is the **Stage-7 Phase-1/Phase-2 module-mode
mismatch** (`model.eval()` vs `model.train()`), which is what actually makes the two players
optimize different surfaces. Stage 8C should be a *diagnostic* on that mismatch under the now-
settled Φ definition, chosen on mathematical coherence and minimal deviation — **not** on
accuracy. The Θ/Ψ reading should be fixed by explicit project decision and documented as such,
since it is not publicly recoverable.

**No production code was modified. No epochs. No accuracy. Stage 8C not started.**
