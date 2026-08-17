# STAGE 6B — REPRESENTATION RESCUE / ROOT-CAUSE ISOLATION

Date: 2026-08-17 · Stages 1-5 frozen, Stage 6 evidence base · DIAGNOSTIC HARNESS ONLY:
zero production edits, no epochs, no accuracy, nothing frozen touched · Seed 42, identical
weights/subjects wherever parameters permit · Signed PCC everywhere.

## 1. ALFF diagnostic (identical subjects/PCC/weights; full 956 for subject rank)

| variant | R0 ROI cos | R0 eR | R2 eR | R3 eR | R7 subj eR | PCC shuffle | PCC delete | ALFF replace |
|---|---|---|---|---|---|---|---|---|
| A alff_paper [0,1] | 0.972 | 1.48 | 1.21 | 1.08 | 2.20 | 0.007 | 0.023 | 0.138 |
| B raw (no norm) | 0.990 | 1.41 | 1.17 | 1.06 | 1.81 | 0.006 | 0.007 | 0.377 |
| C mean-centered | 0.024 | 1.76 | 4.58 | 2.33 | 5.15 | 0.586 | 15.53 | 1.955 |
| D per-band z-score | 0.023 | 1.76 | 4.67 | 2.35 | 5.43 | 0.526 | 17.91 | 1.141 |

Centering (C/D) repairs ALL metrics simultaneously — node rank ×4 at GCN1, subject rank
×2.5, connectivity sensitivity ×80 (shuffle) / ×700 (delete). Raw (B) is as bad as [0,1]:
**positivity, not scale, is the poison** (all 90 ROI vectors in the positive octant).
Stage-1 data untouched — feature substitution at diagnostic time only.

## 2. Aggregator diagnostic — WGINConv algebra note first

Repo `WGINConv` message = `ReLU(x_j) · w_ij`, then `out += (1+ε)x_i`, then ONE linear.
This is NOT A-GCL's `h_i' = MLP(h_i + Σ_j w_ji h_j)`: (a) neighbors are RECTIFIED before
edge weighting (negative feature components destroyed; under signed w a rectified message
is negated whole); (b) transform is a single linear, not the GIN MLP. Therefore the
harness implements the A-GCL equation faithfully from scratch (Linear-ReLU-Linear MLP,
seed 42; Σ_j over all j including the explicit self-loop w_ii=1, matching the dataset's
complete graph). The comparison arms: current signed-safe GCN
(D_|W|^-1/2 W D_|W|^-1/2 X), diagnostic unnormalized `lin(x) + W·lin(x)` reusing the SAME
learned lin/BN, and the exact A-GCL GIN.

## 3-6. Minimum experiment set (E0-E8; feat A = [0,1] alff_paper, B = raw, C = centered)

| cfg | feat | encoder | readout | R0eR | L1eR | nodeR | ROIcos | subjR | PCCsh | PCCdel | ALFFrep | ROIperm | finite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | A | GCN2 | sum | 1.48 | 1.21 | 1.08 | 0.971 | 2.20 | 0.007 | 0.023 | 0.138 | 0.011 | ✓ |
| E1 | B | GCN2 | sum | 1.41 | 1.17 | 1.06 | 0.973 | 1.81 | 0.006 | 0.007 | 0.377 | 0.008 | ✓ |
| E2 | C | GCN2 | sum | 1.76 | 4.58 | 2.33 | 0.815 | 5.15 | **0.586** | **15.53** | 1.955 | 0.454 | ✓ |
| E3 | A | GCN1 | sum | 1.48 | 1.21 | 1.21 | 0.987 | 2.41 | 0.007 | 0.025 | 0.146 | 0.010 | ✓ |
| E4 | A | UNNORM2 | sum | 1.48 | 1.19 | 1.07 | 0.960 | 1.79 | 0.072 | **0.998** | 0.137 | 0.021 | ✓ |
| E5 | A | AGIN2 | sum | 1.48 | 1.17 | 1.12 | 0.978 | 2.06 | 0.070 | **0.998** | 0.087 | 0.019 | ✓ |
| E6 | A | GCN2 | flat | 1.48 | 1.21 | 1.08 | 0.971 | 84.3 | 0.111 | 0.601 | 0.139 | 0.012 | ✓ |
| E7 | A | AGIN2 | flat | 1.48 | 1.17 | 1.12 | 0.978 | 67.8 | 0.214 | 0.998 | 0.088 | 0.020 | ✓ |
| E8 | A | AGIN1 | sum | 1.48 | 1.17 | 1.17 | 0.986 | 2.13 | 0.033 | 0.955 | 0.107 | 0.017 | ✓ |

Depth (section 3): GCN 1→2 layers: node eR 1.21→1.08, ROI cos 0.987→0.971, delete
sensitivity 0.025→0.023 — marginal; the collapse already exists at LAYER 1. AGIN 1→2:
1.17→1.12, delete 0.955→0.998 — same pattern. Depth is not the driver.

Readout test (section 4; SAME frozen node representation, [0,1]+GCN2, rel L2):

| readout | PCC-shuffle | PCC-delete | ALFF-permROI | ALFF-replace |
|---|---|---|---|---|
| global sum | 0.0072 | 0.0234 | 0.0108 | 0.1380 |
| global mean | 0.0072 | 0.0234 | 0.0108 | 0.1380 |
| concat(sum,mean,std) | 0.0073 | 0.0241 | 0.0108 | 0.1380 |
| flatten [90·32] | **0.1109** | **0.6014** | 0.0119 | 0.1386 |
| (node-level reference) | 0.1109 | 0.6014 | — | — |

Connectivity signal EXISTS at node level (0.60 on deletion) and is erased by EVERY
permutation-invariant statistic — sum, mean, and moments are equally blind, because the
90 node vectors are near-identical, so any symmetric aggregate of them barely moves.
Decisively: E2 keeps plain sum pooling and reaches delete-sensitivity 15.5 — pooling is
only an eraser CONDITIONAL on collinear nodes, not an independent cause.

Mechanism of E4/E5 (unnormalized / GIN with [0,1] features): without symmetric degree
normalization the aggregate scales with each node's total connectivity mass, so edge
DELETION becomes visible at the sum readout (0.998) — but edge REARRANGEMENT stays
nearly invisible (0.07; shuffling preserves total mass), node diversity stays collapsed
(eR ~1.1), and subject rank stays ~2. Normalization was cancelling total-mass
information; it is not the collinearity's cause.

## 7. ROOT-CAUSE DECISION

```
ALFF_NORMALIZATION_PRIMARY_CAUSE          = YES
    (positive-octant geometry; [0,1] and raw equally bad, centering repairs node rank,
     subject rank, and BOTH connectivity sensitivities while keeping the frozen
     architecture bit-for-bit)

DEGREE_NORMALIZATION_PRIMARY_CAUSE        = CONTRIBUTOR
    (cancels total-connectivity-mass information at the pooled level: removing it
     restores deletion sensitivity 0.023 -> 0.998, but not rearrangement sensitivity,
     node diversity, or subject rank)

GCN_DEPTH_PRIMARY_CAUSE                   = NO
    (layer 1 already produces the collapse; adding layer 2 changes metrics marginally)

SUM_POOLING_PRIMARY_CAUSE                 = CONTRIBUTOR
    (blind ONLY because nodes are collinear -- sum == mean == moments in effect; with
     centered features the same sum readout sees connectivity at 15.5)

EXACT_AGCL_GIN_RESCUES_CONNECTIVITY       = PARTIAL
    (mass/deletion sensitivity YES (0.998); rearrangement NO (0.07); node diversity and
     subject rank NOT rescued)

ROI_PRESERVING_READOUT_RESCUES_CONNECTIVITY = YES
    (exposes the full node-level signal -- 0.60 delete / 0.11 shuffle -- trivially, at
     the cost of a 2880-dim non-permutation-invariant readout; DIAGNOSTIC ONLY)

BEST_MINIMAL_RESCUE = per-subject per-band CENTERED (or z-scored) ALFF node features
    -- one feature-level change, zero architecture changes, keeps sum pooling, repairs
    every measured pathology at once (E2/variant D). Optional secondary lever if
    total-mass sensitivity is also wanted: GIN-style unnormalized aggregation (E5),
    which composes with centering.
```

## 8. Scientific note (recorded verbatim constraints)

Do not equate this study with A-GCL. A-GCL: ABIDE-I **987** subjects after QC,
**fMRIPrep**, **AAL1 116** ROIs, joint [0,1] ALFF normalization, weighted **GIN**
primary encoder, reports **80.65%** mean 5-fold accuracy. This study: **956** subjects,
current ABIDE preprocessing profile, **AAL90**. A-GCL additionally reports similar
performance replacing GIN with GCN or GraphSAGE — so GIN is a CANDIDATE, not a
pre-decided fix. Consistency observation (not a claim about A-GCL): the E5 row shows
that GIN-style aggregation retains total-connectivity-mass sensitivity even under
[0,1] features — i.e., A-GCL's own joint-[0,1] + GIN pairing is not self-contradictory
with our diagnosis, which specifically concerns [0,1] features paired with
degree-NORMALIZED GCN + sum pooling in THIS architecture at initialization.

## 9. Frozen-state confirmation

Untouched: Dataset.py, Stage-1 caches, Stage-2 signed graph semantics, Stage-3 ToyNet,
Stage-4 M_ij, Stage-5 profiles, training loss, optimizers. All experiments ran in the
scratchpad harness (`stage6b_rescue.py`, results `stage6b_results.pkl`).

STOP: root-cause table delivered. No Stage 7.
