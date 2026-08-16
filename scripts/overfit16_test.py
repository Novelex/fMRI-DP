"""Overfit-16 learnability microscope (test harness, not a final run).

A working contrastive model must be able to memorize 16 subjects. Arms are
selected by CLI flags; alignment (each subject's original embedding finding
its own augmented version as nearest neighbour) is now tracked every 10
epochs -- the real pipeline harvests the best-validation checkpoint, so the
PEAK is the paper-protocol-relevant number, not the endpoint (verified: old
nested-CV runs peaked at eval idx 0-4, i.e. within the first ~25 epochs).

Flags:
  --replicate        authors' config: multiplicative gate, attention mix OFF,
                     ce_loss in Phase 2.
  --gc1 / --layerwise  encoder-geometry variants.
  --freeze_adversary view_learner optimizer lr=0.0 (adversary pinned at init).
  --rebalance        live adversary, view_lr=5e-5, reg_lambda=1.0.
  --reg5             live adversary, reg_lambda=5.0 -- AD-GCL's own shipped
                     default (verified: susheels/adgcl test_minmax_tu.py
                     default=5.0), 25x stronger than this project's 0.2.

Pass criteria (written before running):
  1. model_loss falls >= 30% from its early value.
  2. PEAK alignment >= 15/16 across all measured epochs.
  3. NaN-guard log stays empty; all losses finite throughout.
"""
import sys
import os
import logging

sys.path.insert(0, '/users/3171356m/muhammad/GraSTIACL')
os.chdir('/users/3171356m/muhammad/GraSTIACL')

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from datasets import ADNIDataset
from unsupervised.training import build_model_and_view_learner, train_one_epoch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EPOCHS = 100
SEED = 123

REPLICATE = '--replicate' in sys.argv
NUM_LAYERS = 1 if '--gc1' in sys.argv else 2
POOLING = 'layerwise' if '--layerwise' in sys.argv else 'standard'
FREEZE_ADVERSARY = '--freeze_adversary' in sys.argv
REBALANCE = '--rebalance' in sys.argv
REG5 = '--reg5' in sys.argv

VIEW_LR = 0.00005 if REBALANCE else 0.0005
REG_LAMBDA = 5.0 if REG5 else (1.0 if REBALANCE else 0.2)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}, REPLICATE={REPLICATE}, layers={NUM_LAYERS}, pooling={POOLING}, "
      f"FROZEN_ADVERSARY={FREEZE_ADVERSARY}, view_lr={VIEW_LR}, reg_lambda={REG_LAMBDA}")

dataset = ADNIDataset('data/GraSTIACL_ABIDE_979', 'GraSTIACL_ABIDE_979', node_feature_mode='alff')

# Balanced 16-subject subset: first 8 ASD (y=1) + first 8 NC (y=0) in dataset order.
asd_idx = [i for i in range(len(dataset)) if int(dataset[i].y.item()) == 1][:8]
nc_idx = [i for i in range(len(dataset)) if int(dataset[i].y.item()) == 0][:8]
subset = torch.utils.data.Subset(dataset, asd_idx + nc_idx)
loader = DataLoader(subset, batch_size=16, shuffle=True, drop_last=True)

# enable_attention_mix=not REPLICATE matches GraSTIACL.py's own wiring: the
# authors' released code runs GCN-branch-only (mix line commented out).
model, view_learner, model_opt, view_opt, gamma_orig, beta = build_model_and_view_learner(
    num_dataset_features=3, emb_dim=32, num_gc_layers=NUM_LAYERS, drop_ratio=0.0,
    pooling_type=POOLING, gamma_mode='paper_literal', mij_source='alff', num_dyn_windows=3,
    vib_hidden_dim=400, model_lr=0.0005, view_lr=VIEW_LR, device=device,
    enable_attention_mix=not REPLICATE)
if FREEZE_ADVERSARY:
    view_opt = torch.optim.Adam(view_learner.parameters(), lr=0.0)

eval_batch = next(iter(DataLoader(subset, batch_size=16, shuffle=False))).to(device)
gcn_w_eval = eval_batch.edge_weight.abs().clamp(1e-6, 1.0)


def eval_alignment(current_beta):
    """Alignment + pure contrastive loss on the eval batch. Saves/restores the
    RNG state so mid-training evals don't perturb the training noise stream."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training_m, was_training_v = model.training, view_learner.training
    model.eval()
    view_learner.eval()
    with torch.no_grad():
        z, _ = model(eval_batch.batch, eval_batch.x, eval_batch.edge_index, current_beta, None,
                     gcn_w_eval, eval_batch.edge_weight, None, gamma=gamma_orig)
        edge_logits = view_learner(eval_batch.batch, eval_batch.x, eval_batch.edge_index, current_beta,
                                   None, gcn_w_eval, eval_batch.edge_weight, None, gamma=gamma_orig)
        torch.manual_seed(SEED)
        bias = 1e-4
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size(), device=device) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps) + edge_logits
        if REPLICATE:
            aug_w = gcn_w_eval * torch.sigmoid(gate_inputs).squeeze()
        else:
            aug_w = torch.sigmoid(gate_inputs).squeeze()
        gamma_aug = aug_w.view(eval_batch.num_graphs, -1).mean(dim=1)
        z_aug, _ = model(eval_batch.batch, eval_batch.x, eval_batch.edge_index, current_beta, None,
                         aug_w, eval_batch.edge_weight, None, gamma=gamma_aug)
        zn = F.normalize(z, dim=1)
        zan = F.normalize(z_aug, dim=1)
        sim = zn @ zan.T
        n_correct = int((sim.argmax(dim=1) == torch.arange(16, device=device)).sum().item())
        ec = model.calc_loss(z, z_aug).item()
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    if was_training_m:
        model.train()
    if was_training_v:
        view_learner.train()
    return n_correct, ec


losses = []
history = []  # (epoch, alignment, eval_contrastive)
n0, ec0 = eval_alignment(beta)
history.append((0, n0, ec0))
print(f"epoch   0 (untrained): alignment {n0}/16, eval_contrastive {ec0:.4f}")

for epoch in range(1, EPOCHS + 1):
    fin_model_loss, fin_view_loss, fin_reg, _, _, beta = train_one_epoch(
        model, view_learner, model_opt, view_opt, loader, device,
        beta, gamma_orig, ce_lambda=2.0, reg_lambda=REG_LAMBDA, kld_lambda=0.003, template=90,
        replicate_original_code=REPLICATE)
    losses.append(fin_model_loss)
    if not np.isfinite(fin_model_loss) or not np.isfinite(fin_view_loss):
        print(f"FAIL: non-finite loss at epoch {epoch} (model={fin_model_loss}, view={fin_view_loss})")
        sys.exit(1)
    if epoch % 10 == 0:
        n, ec = eval_alignment(beta)
        history.append((epoch, n, ec))
        print(f"epoch {epoch:3d}: model_loss={fin_model_loss:.4f} view_loss={fin_view_loss:.4f} "
              f"reg={fin_reg:.4f} | alignment {n}/16, eval_contrastive {ec:.4f}")

# ---- Criterion 1: loss fall ----
early = float(np.mean(losses[:5]))
late = float(np.mean(losses[-5:]))
fall_pct = 100.0 * (early - late) / abs(early) if early != 0 else 0.0
print(f"\nloss early(mean of first 5)={early:.4f}  late(mean of last 5)={late:.4f}  fall={fall_pct:.1f}%")
c1 = fall_pct >= 30.0

# ---- Criterion 2: PEAK alignment across all measured epochs ----
peak_epoch, peak_align, peak_ec = max(history, key=lambda t: (t[1], -t[2]))
final_epoch, final_align, final_ec = history[-1]
print(f"PEAK alignment: {peak_align}/16 at epoch {peak_epoch} (eval_contrastive {peak_ec:.4f})")
print(f"FINAL alignment: {final_align}/16 at epoch {final_epoch} (eval_contrastive {final_ec:.4f})")
print("full history (epoch, align, eval_contrastive): " +
      ", ".join(f"({e},{n},{c:.3f})" for e, n, c in history))
c2 = peak_align >= 15

print(f"\n{'='*60}")
print(f"Criterion 1 (loss fall >= 30%):        {'PASS' if c1 else 'FAIL'} ({fall_pct:.1f}%)")
print(f"Criterion 2 (PEAK alignment >= 15/16): {'PASS' if c2 else 'FAIL'} (peak {peak_align}/16 @ epoch {peak_epoch})")
print(f"Criterion 3 (no NaN-guard fires): check log above -- any 'NaN detected' line = FAIL")
print("OVERFIT-16: " + ("PASS" if (c1 and c2) else "FAIL"))
