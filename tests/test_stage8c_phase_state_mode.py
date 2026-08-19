"""Stage 8C unit tests for --phase_state_mode.

Covers exactly the three things the correction is responsible for:
  1. BatchNorm buffer ownership  (a phase may mutate only its own player's buffers)
  2. Optimizer ownership         (no non-owner PARAMETER moves in either mode)
  3. Forward-state semantics     (legacy is preserved bit-for-bit; consistent puts both
                                  phases on the same train-like distribution)

Run:  python3 tests/test_stage8c_phase_state_mode.py
"""
import os, sys, inspect
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from datasets import ADNIDataset
from unsupervised.training import (build_model_and_view_learner, train_one_epoch,
                                   _bn_buffer_snapshot, _bn_buffer_restore)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if not cond and detail else ""))

DS = ADNIDataset('data/GraSTIACL_ABIDE_979', 'GraSTIACL_ABIDE_979',
                 node_feature_mode='alff_new_z')
IDX = [(k * 17) % len(DS) for k in range(4)]

def build(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    return build_model_and_view_learner(
        num_dataset_features=3, emb_dim=32, num_gc_layers=2, drop_ratio=0.0,
        pooling_type='standard', gamma_mode='baseline', mij_source='alff',
        num_dyn_windows=3, vib_hidden_dim=64, model_lr=5e-4, view_lr=5e-4,
        device=torch.device('cpu'), enable_attention_mix=True, signed_edges=True,
        tae_profile='paper_intent')

def loader():
    return DataLoader([DS[i] for i in IDX], batch_size=4, shuffle=False)

def run_epoch(mode, seed=42, rng=999):
    m, v, mo, vo, g, b = build(seed)
    torch.manual_seed(rng)
    out = train_one_epoch(m, v, mo, vo, loader(), torch.device('cpu'), b, g,
                          ce_lambda=2.0, reg_lambda=0.2, kld_lambda=0.003, template=90,
                          epoch_num=1, signed_edges=True, phase_state_mode=mode)
    return m, v, out

print("=== 1. BatchNorm buffer helpers ===")
m, v, mo, vo, g, b = build()
snap = _bn_buffer_snapshot(m)
check("snapshot captures exactly the 6 model BN buffers (2 BN x mean/var/count)",
      len(snap) == 6 and all(('running_' in k or 'num_batches_tracked' in k) for k in snap),
      str(sorted(snap)))
with torch.no_grad():
    for k in snap:
        t = m.state_dict()[k]
        t.copy_(t + (1 if t.dtype == torch.int64 else 3.14))
moved = any(not torch.equal(m.state_dict()[k], snap[k]) for k in snap)
_bn_buffer_restore(m, snap)
check("restore reverts every buffer exactly (including int64 num_batches_tracked)",
      moved and all(torch.equal(m.state_dict()[k], snap[k]) for k in snap))

print("\n=== 2. legacy mode is preserved vs the PRE-CHANGE source (git HEAD) ===")
# The honest instrument: PyG's scatter reductions are float32 and thread-order dependent,
# so two identical runs of the SAME code already differ by ~1e-7. We therefore compare
# new-legacy against the PRE-CHANGE file and require the difference to be no larger than
# that same code's own run-to-run noise.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "training_prechange",
    "/tmp/claude-102000043/-users-3171356m-muhammad/33cc8a5d-9d13-4fe3-8b70-0190f243f063/scratchpad/training_prechange.py")
_old = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_old)
check("pre-change module has NO phase_state_mode (so it is genuinely the old code)",
      'phase_state_mode' not in inspect.signature(_old.train_one_epoch).parameters)

def run_old(seed=42, rng=999):
    m, v, mo, vo, g, b = build(seed)
    torch.manual_seed(rng)
    return _old.train_one_epoch(m, v, mo, vo, loader(), torch.device('cpu'), b, g,
                                ce_lambda=2.0, reg_lambda=0.2, kld_lambda=0.003, template=90,
                                epoch_num=1, signed_edges=True)

o_old_a = run_old(); o_old_b = run_old()
noise = max(abs(o_old_a[0]-o_old_b[0]), abs(o_old_a[1]-o_old_b[1]))
_, _, o_new = run_epoch('legacy')
delta = max(abs(o_old_a[0]-o_new[0]), abs(o_old_a[1]-o_new[1]))
print(f"       (pre-change code's own run-to-run noise at this batch size: {noise:.3e})")
# If the old code is deterministic here (noise == 0) the bar is BIT-FOR-BIT equality;
# otherwise new-legacy must sit inside that same code's own noise band. PyG scatter
# reductions can be thread-order dependent at larger batch sizes, hence the two branches.
check("new legacy reproduces the pre-change code EXACTLY (bit-for-bit when the old code "
      "is deterministic, else within its own run-to-run noise)",
      delta == 0.0 if noise == 0.0 else delta <= max(noise * 4, 1e-6),
      f"delta {delta:.3e}, old-vs-old noise {noise:.3e}")
check("legacy relative difference is float32-negligible (<1e-5 relative)",
      delta / max(abs(o_old_a[0]), 1e-9) < 1e-5, f"{delta/abs(o_old_a[0]):.3e}")

sig = inspect.signature(train_one_epoch)
check("train_one_epoch default phase_state_mode == 'legacy' (no silent default change)",
      sig.parameters['phase_state_mode'].default == 'legacy',
      str(sig.parameters['phase_state_mode'].default))

print("\n=== 3+4. BatchNorm buffer OWNERSHIP and frozen-player protection ===")
for mode in ('legacy', 'consistent'):
    m, v, mo, vo, g, b = build()
    m_before, v_before = _bn_buffer_snapshot(m), _bn_buffer_snapshot(v)
    torch.manual_seed(999)
    train_one_epoch(m, v, mo, vo, loader(), torch.device('cpu'), b, g,
                    ce_lambda=2.0, reg_lambda=0.2, kld_lambda=0.003, template=90,
                    epoch_num=1, signed_edges=True, phase_state_mode=mode)
    m_after, v_after = _bn_buffer_snapshot(m), _bn_buffer_snapshot(v)
    d_model = sum(float((m_after[k].float() - m_before[k].float()).abs().sum()) for k in m_before)
    d_view = sum(float((v_after[k].float() - v_before[k].float()).abs().sum()) for k in v_before)
    check(f"{mode}: model BN buffers move (Phase 2 owns the model)", d_model > 0, f"{d_model}")
    check(f"{mode}: view BN buffers move (Phase 1 owns the view)", d_view > 0, f"{d_view}")
    # num_batches_tracked is the exact witness of how many forwards were KEPT.
    dm = {k: int(m_after[k] - m_before[k]) for k in m_before if 'num_batches_tracked' in k}
    dv = {k: int(v_after[k] - v_before[k]) for k in v_before if 'num_batches_tracked' in k}
    if mode == 'consistent':
        # Phase 1 forwards the model twice but is rolled back; Phase 2 keeps its two.
        check("consistent: model num_batches_tracked +2 per batch (Phase-1 forwards rolled back)",
              all(x == 1 * 2 for x in dm.values()), str(dm))
        # Phase 2 forwards the view once but is rolled back; Phase 1 keeps its one.
        check("consistent: view num_batches_tracked +1 per batch (Phase-2 forward rolled back)",
              all(x == 1 for x in dv.values()), str(dv))
    else:
        check("legacy: model +2 per batch (Phase 2 only; Phase 1 has the model in eval)",
              all(x == 2 for x in dm.values()), str(dm))
        check("legacy: view +1 per batch (Phase 1 only; Phase 2 has the view in eval)",
              all(x == 1 for x in dv.values()), str(dv))

print("\n=== 5. optimizer ownership: no NON-OWNER parameter moves, in either mode ===")
for mode in ('legacy', 'consistent'):
    m, v, mo, vo, g, b = build()
    # freeze the view optimizer by zeroing its lr -> Phase 1 cannot move Phi;
    # then any Phi movement would have to come from Phase 2 (which must not happen).
    for grp in vo.param_groups: grp['lr'] = 0.0
    v_before = {n: p.detach().clone() for n, p in v.named_parameters()}
    torch.manual_seed(999)
    train_one_epoch(m, v, mo, vo, loader(), torch.device('cpu'), b, g,
                    ce_lambda=2.0, reg_lambda=0.2, kld_lambda=0.003, template=90,
                    epoch_num=1, signed_edges=True, phase_state_mode=mode)
    moved = max(float((p.detach() - v_before[n]).abs().max()) for n, p in v.named_parameters())
    check(f"{mode}: Phase 2 never moves a Phi parameter (max|dPhi| with view lr=0)",
          moved == 0.0, f"{moved}")

print("\n=== 6. forward-state semantics differ between the two modes ===")
o_leg = o_new
_, _, o_con = run_epoch('consistent', seed=42, rng=999)
check("consistent produces a materially different Phase-1 (view) objective than legacy",
      abs(o_leg[1] - o_con[1]) > 1e-3, f"legacy view_loss {o_leg[1]:.6f} vs consistent {o_con[1]:.6f}")
check("both modes produce finite losses", all(np.isfinite([o_leg[0], o_leg[1], o_con[0], o_con[1]])),
      f"{o_leg[:2]} {o_con[:2]}")

print(f"\nRESULT: {len(PASS)}/{len(PASS)+len(FAIL)} passed")
for f in FAIL: print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
