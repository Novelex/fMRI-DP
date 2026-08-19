"""Stage 8E unit tests for the experimental --lambda_pairing_mode control.

Guarantees exactly three things:
  1. 'production' is BIT-FOR-BIT the pre-Stage-8E code (no frozen result moves).
  2. the helper returns the intended (gamma_orig, gamma_aug) pairing for each mode.
  3. every non-production mode actually equalises the two views' lambda and is finite.

Run:  python3 tests/test_stage8e_lambda_pairing.py
"""
import os, sys, inspect, importlib.util
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, torch
from torch_geometric.loader import DataLoader
from datasets import ADNIDataset
from unsupervised.training import build_model_and_view_learner, train_one_epoch, _pair_gammas

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if not cond and detail else ""))

DS = ADNIDataset('data/GraSTIACL_ABIDE_979', 'GraSTIACL_ABIDE_979', node_feature_mode='alff_new_z')
IDX = [(k * 17) % len(DS) for k in range(4)]

def build(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    return build_model_and_view_learner(
        num_dataset_features=3, emb_dim=32, num_gc_layers=2, drop_ratio=0.0,
        pooling_type='standard', gamma_mode='baseline', mij_source='alff',
        num_dyn_windows=3, vib_hidden_dim=64, model_lr=5e-4, view_lr=5e-4,
        device=torch.device('cpu'), enable_attention_mix=True, signed_edges=True,
        tae_profile='paper_intent')
def loader(): return DataLoader([DS[i] for i in IDX], batch_size=4, shuffle=False)

def run(mode, state='legacy', seed=42, rng=999):
    m, v, mo, vo, g, b = build(seed); torch.manual_seed(rng)
    out = train_one_epoch(m, v, mo, vo, loader(), torch.device('cpu'), b, g,
                          ce_lambda=2.0, reg_lambda=0.2, kld_lambda=0.003, template=90,
                          epoch_num=1, signed_edges=True, phase_state_mode=state,
                          lambda_pairing_mode=mode)
    return m, v, out

print("=== 1. default is 'production' and 'production' is bit-for-bit the pre-8E code ===")
sig = inspect.signature(train_one_epoch)
check("train_one_epoch default lambda_pairing_mode == 'production'",
      sig.parameters['lambda_pairing_mode'].default == 'production',
      str(sig.parameters['lambda_pairing_mode'].default))

_spec = importlib.util.spec_from_file_location("training_pre8e", "/tmp/training_pre8e.py")
_old = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_old)
check("reference module has NO lambda_pairing_mode (genuinely the pre-8E code)",
      'lambda_pairing_mode' not in inspect.signature(_old.train_one_epoch).parameters)

for state in ('legacy', 'consistent'):
    m0, v0, mo, vo, g, b = build(); torch.manual_seed(999)
    o_old = _old.train_one_epoch(m0, v0, mo, vo, loader(), torch.device('cpu'), b, g,
                                 ce_lambda=2.0, reg_lambda=0.2, kld_lambda=0.003, template=90,
                                 epoch_num=1, signed_edges=True, phase_state_mode=state)
    m1, v1, o_new = run('production', state)
    d_loss = max(abs(a - c) for a, c in zip(o_old[:3], o_new[:3]))
    d_w = max(float((p.detach() - dict(m0.named_parameters())[n].detach()).abs().max())
              for n, p in m1.named_parameters())
    d_v = max(float((p.detach() - dict(v0.named_parameters())[n].detach()).abs().max())
              for n, p in v1.named_parameters())
    check(f"{state}: production reproduces pre-8E losses bit-for-bit", d_loss == 0.0, f"{d_loss:.3e}")
    check(f"{state}: production reproduces pre-8E Theta weights bit-for-bit", d_w == 0.0, f"{d_w:.3e}")
    check(f"{state}: production reproduces pre-8E Phi weights bit-for-bit", d_v == 0.0, f"{d_v:.3e}")

print("\n=== 2. pairing helper semantics ===")
ga = torch.tensor([0.2, 0.4, 0.6, 0.8])
check("production returns (gamma_orig, gamma_aug) unchanged",
      _pair_gammas('production', 1.0, ga) == (1.0, ga) or
      (_pair_gammas('production', 1.0, ga)[0] == 1.0 and
       torch.equal(_pair_gammas('production', 1.0, ga)[1], ga)))
o, a = _pair_gammas('matched', 1.0, ga)
check("matched gives BOTH views the augmented gamma", torch.equal(o, ga) and torch.equal(a, ga))
check("attention_off gives both views gamma=1.0", _pair_gammas('attention_off', 1.0, ga) == (1.0, 1.0))
check("balanced gives both views gamma=0.5", _pair_gammas('balanced', 1.0, ga) == (0.5, 0.5))
try:
    _pair_gammas('nonsense', 1.0, ga); ok = False
except ValueError: ok = True
check("unknown pairing mode raises", ok)
try:
    run('nonsense'); ok = False
except ValueError: ok = True
check("train_one_epoch rejects an unknown lambda_pairing_mode", ok)

print("\n=== 3. every non-production mode runs, is finite, and equalises lambda ===")
LAM = lambda g: 1 - float(torch.as_tensor(g, dtype=torch.float32).clamp(1e-4, 1 - 1e-4).mean())
for mode, want in (('matched', None), ('attention_off', 1e-4), ('balanced', 0.5)):
    _, _, out = run(mode)
    check(f"{mode}: losses finite", all(np.isfinite(out[:3])), str(out[:3]))
    o, a = _pair_gammas(mode, 1.0, ga)
    check(f"{mode}: lambda_orig == lambda_aug (the asymmetry is removed)",
          abs(LAM(o) - LAM(a)) < 1e-9, f"{LAM(o)} vs {LAM(a)}")
    if want is not None:
        # float32: 1 - clamp(1.0) is 1.0001659e-4, not exactly 1e-4. attention_off is
        # therefore the CLAMP FLOOR lambda=1e-4, not literally lambda=0 -- documented,
        # not asserted away.
        check(f"{mode}: shared lambda == {want} to float32 precision",
              abs(LAM(o) - want) <= max(1e-9, 1e-3 * want), f"{LAM(o)}")

# the defect Stage 8E is testing: production really is asymmetric
o, a = _pair_gammas('production', 1.0, ga)
check("production really IS lambda-asymmetric (this is the defect under test)",
      abs(LAM(o) - LAM(a)) > 0.1, f"{LAM(o)} vs {LAM(a)}")

print(f"\nRESULT: {len(PASS)}/{len(PASS)+len(FAIL)} passed")
for f in FAIL: print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
