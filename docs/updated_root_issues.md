# fMRI-DP — Root Cause Report

Two defects. Both break the adversarial training itself, which is why fixing the
encoder details did not help.

---

# THE EVIDENCE: your loss numbers decode

`calc_loss` with batch 32, T=0.2, 31 negatives:

| State | loss |
|---|---|
| perfect (pos=1, neg=0) | −1.57 |
| good (pos=0.5, neg=0) | 0.93 |
| **random (pos = neg)** | **3.43** |
| pos worse than neg by 0.5 | 5.93 |
| pos worse than neg by 1.0 | 8.43 |

Your `model_loss` = `calc_loss + 2·ce_loss`, and BCE ∈ [0, ~1], so `calc_loss` runs
roughly **2.9 → 6.5** over ten epochs.

It starts near random (3.43) and ends **3 nats worse than random**. A subject's own
augmented view becomes *less* similar to itself than to other random subjects —
cosine gap ≈ −0.61.

That is not weak learning. That is the representation being actively destroyed.

---

# ISSUE A — `GraSTIACL.py:272` and `:356`
## The same tensors get gradient ascent and gradient descent on the same loss

```python
# Phase A, line 271-273
(-view_loss).backward()      # ascent
opt_shared.step()            # <- shared_encoder + shared_net

# Phase B, line 355-357
model_loss.backward()        # descent
opt_shared.step()            # <- the SAME shared_encoder + shared_net
```

Both `view_loss` and `model_loss` contain `model.calc_loss(x, x_aug)`.

So `opt_shared` maximises the contrastive loss in Phase A and minimises it in Phase B,
on **identical parameters**. That is `min_θ max_θ L(θ)` — mathematically degenerate.
The updates oscillate and the representation drifts wherever the imbalance pushes it.

### Why this is wrong per the paper
Eq. 15: `min_Φ max_Θ I_N` · Eq. 21: `min_Ψ max_Ω I_R` · Eq. 22: `min_{Φ,Ψ} max_{Θ,Ω}(I_R + I_N)`

**Φ ≠ Θ and Ψ ≠ Ω.** A min–max is only meaningful over *distinct* parameter sets. Eq. 3
(which the paper inherits from AD-GCL) is explicit: Φ = "the augmenter GNN and MLP",
Θ = "the GNN f". Two networks. AD-GCL gives its view learner its own encoder, and the
released GraSTI-ACL code creates two independent `TAEncoder`+`ToyNet` pairs.

### This is my fault
I recommended sharing the backbone, based on §4.2's *"approximately 415k parameters
shared between them."* I computed earlier in this project that **ToyNet is exactly
415,292 parameters**. So 422k ≈ 415k + TAE + proj_head, and 420k ≈ 415k + TAE + edge MLP.

"415k shared" describes **both modules containing a ToyNet of the same 415k
architecture** — not weight tying. I read it as tensor sharing and it broke the min–max.

### Fix
Revert to two independent instances:

```python
model = GInfoMinMax(
    TA_encoder.TAEncoder(num_dataset_features=3, emb_dim=args.emb_dim,
                         num_gc_layers=args.num_gc_layers,
                         drop_ratio=args.drop_ratio,
                         pooling_type=args.pooling_type),
    GraSTI.ToyNet(input_dim=args.template * (1 + args.num_dyn_windows),
                  hidden_dim=args.vib_hidden_dim),
    args.emb_dim).to(device)

view_learner = ViewLearner(
    TA_encoder.TAEncoder(...),        # fresh instance
    GraSTI.ToyNet(...)).to(device)    # fresh instance

model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
view_optimizer  = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)
```

### Downstream check
- The three-optimizer split (`opt_shared` / `opt_model_head` / `opt_view_head`) becomes
  unnecessary and should go — with no shared tensors, two optimizers cannot conflict.
  **Issue 8 from my previous report dissolves entirely.**
- The `model.encoder.train()` / `model.net.train()` re-assertions after `model.eval()`
  and `view_learner.eval()` also become unnecessary. Harmless if left, but delete them
  or they will confuse you later.
- `init_module_weights` must be called on **both** encoders and **both** nets.
- Parameter count goes ~427k → ~842k, matching the paper's 422k + 420k.
- `edge_logits_vl` still comes from `view_learner`, `edge_logits` from `model` — with
  separate nets these are now genuinely different quantities, which is correct
  (see Issue C).

---

# ISSUE B — `GraSTIACL.py:265`
## The edge-retention regularizer is computed and then thrown away

```python
reg = torch.stack(reg).mean()                                    # line 260
...
view_loss = model.calc_loss(x, x_aug) - args.kld_lambda * kld_loss   # line 265  <- no reg
...
reg_all += reg.item() * num_graph_with_edges                     # line 267, logging only
```

`reg` = mean edge-**drop** probability. It is computed at lines 250–260, used for
logging and for the (now-dead) `fin_reg_beta`, and **never enters any loss**.

### Why this matters
The view learner maximises `view_loss`. With no penalty on how much of the graph it
destroys, its optimal strategy is to drive the gate toward 0 and make `x_aug`
uninformative. `λ_KL = 0.003` constrains the VIB's μ/σ — it does **not** constrain edge
retention.

This is precisely the trajectory your loss shows: unconstrained augmenter wins the arms
race, contrastive loss climbs past random, representation collapses.

AD-GCL — which this codebase is built on and which the paper cites as Eq. 3 — includes
exactly this regularizer for exactly this reason. §3.2 of the paper describes the same
intent: *"To prevent excessive perturbations that could compromise useful information,
we regulate the temperature coefficient… instead of discarding certain edges, we weaken
their connections."*

### Fix
```python
parser.add_argument('--reg_lambda', type=float, default=1.0)
...
view_loss = model.calc_loss(x, x_aug) \
          - args.reg_lambda * reg \
          - args.kld_lambda * kld_loss
```

Sign: `view_loss` is **maximised**, so `−reg_lambda·reg` means *minimise the drop
probability* — keep edges. Correct direction.

Start at 1.0. If `fin_reg` still climbs toward 1.0 across epochs, raise it; if the gate
saturates near 1.0 (no augmentation at all), lower it. **Log `fin_reg` every epoch and
watch it** — a healthy run holds it somewhere around 0.2–0.5.

### Downstream check
- `reg` and `num_graph_with_edges` are already computed — no new state.
- `reg_all` / `fin_reg` logging is unaffected.
- `model_loss` is untouched, so Phase B is unchanged.
- `fin_reg_beta` feeds only the dead `beta` variable — unaffected either way.

---

# ISSUE C — consequence of fixing A
## `ce_loss` must use the view learner's logits

```python
edge_logits_sig = torch.sigmoid((edge_logits + gate_inputs_) / temperature)   # ~line 292
```

`edge_logits` comes from `model(...)`. While the backbone was shared these were two
samples from one ToyNet — near-equivalent. **After Issue A they come from genuinely
different networks**, so this compares the wrong adjacency against `M_ij`.

Eq. 13's `L_CE(A_ij, M_ij)` needs `A` = the adjacency that actually built the augmented
graph.

### Fix
```python
edge_logits_sig = torch.sigmoid((edge_logits_vl + gate_inputs_) / temperature)
```

### Downstream check
`edge_logits` from the model's forward is used **only** here — grep confirms.
`edge_logits_vl` is already in scope at that line. `gate_inputs_` unchanged, so the noise
term stays paired with the logits it was drawn for.

---

# ORDER

```
Run 0   Issue A + B + C together
        (A and B are the two halves of the same broken adversarial loop;
         C is required for A to be correct)
Run 1   if calc_loss now sits BELOW 3.43 and falls: continue to capacity
        (--num_gc_layers 1 --emb_dim 128 --pooling_type layerwise --batch_size 128)
```

## The diagnostic that tells you it worked

Log `calc_loss` separately from `model_loss` every epoch, and log `fin_reg`.

| Observation | Meaning |
|---|---|
| `calc_loss` falls below **3.43** and keeps falling | adversarial loop is healthy — proceed |
| `calc_loss` sits at ~3.43 | views carry no shared information — check the gate |
| `calc_loss` climbs above 3.43 | augmenter still winning — raise `reg_lambda` |
| `fin_reg` → 1.0 | augmenter destroying the graph — raise `reg_lambda` |
| `fin_reg` → 0.0 | no augmentation happening — lower `reg_lambda` |

**3.43 is the number to watch.** Anything above it means the model is worse than random
at matching a subject to its own augmented view, and no downstream classifier can
recover from that.

---

# STILL OPEN FROM THE PREVIOUS REPORT

Verified as already fixed in this commit: deterministic λ at inference (`:325-328`),
`fin_reg` clamped before `np.random.beta`, three-way optimizer split (superseded by
Issue A).

Still applicable, but **only after Run 0**: oversmoothing on the complete graph
(`--num_gc_layers 1`), 32-dim embeddings (`--emb_dim 128 --pooling_type layerwise`),
`F.normalize` before `global_add_pool`, 31 InfoNCE negatives (`--batch_size 128`),
`|W|·A'` double-counting FC in the augmented view, hardcoded 90 in ToyNet.

None of those matter while the adversarial loop is destroying the representation.
