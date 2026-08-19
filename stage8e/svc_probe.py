"""Stage 8E SECONDARY evidence — locked LinearSVC embedding probe.

Classification is SECONDARY evidence in Stage 8E and is never used to select an
epoch or a configuration. The probe is locked before use:

  * estimator : LinearSVC(dual=False, fit_intercept=True, max_iter=10000) -- the exact
                estimator GraSTIACL.py already uses for embedding evaluation.
  * folds     : StratifiedKFold(n_splits=5, shuffle=True, random_state=42) -- the SAME
                fold objects are reused for every arm and every epoch, so the arms are
                compared on identical splits.
  * embedding : eval-mode graph embedding of the ORIGINAL view (the downstream path),
                z-scored per feature on the training fold only.
  * epochs    : 0, 10, 30 only.
  * metric    : balanced accuracy (the cohort is 455 ASD / 501 NC).

Raw reference baselines (from the frozen ML baseline stage, not recomputed here):
  FC ~ 0.663   ALFF ~ 0.591   FC+ALFF ~ 0.657
"""
import os, sys, json, glob
REPO = '/users/3171356m/muhammad/GraSTIACL'
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, torch
torch.set_num_threads(8)
from torch_geometric.loader import DataLoader
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from datasets import ADNIDataset
from unsupervised.training import build_model_and_view_learner

SEED, EPOCHS = 42, (0, 10, 30)
ds = ADNIDataset('data/GraSTIACL_ABIDE_979', 'GraSTIACL_ABIDE_979',
                 node_feature_mode='alff_new_z')
loader = DataLoader(ds, batch_size=64, shuffle=False)
Y = np.array([int(ds[i].y.view(-1)[0]) for i in range(len(ds))])
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(Y)), Y))


def embed(ckpt_path, cfg):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    torch.manual_seed(cfg['seed'])
    m, v, _, _, g0, beta = build_model_and_view_learner(
        num_dataset_features=3, emb_dim=cfg.get('emb_dim', 32),
        num_gc_layers=cfg.get('num_gc_layers', 2), drop_ratio=cfg.get('drop_ratio', 0.0),
        pooling_type='standard', gamma_mode='baseline', mij_source=cfg.get('mij_source', 'alff'),
        num_dyn_windows=cfg.get('num_dyn_windows', 3), vib_hidden_dim=cfg.get('vib_hidden_dim', 64),
        model_lr=5e-4, view_lr=5e-4, device=torch.device('cpu'),
        enable_attention_mix=True, signed_edges=True, tae_profile=cfg.get('tae_profile', 'paper_intent'))
    m.load_state_dict(ck['model']); m.eval()
    Z = []
    with torch.no_grad():
        for b in loader:
            z, _ = m(b.batch, b.x, b.edge_index, beta, None, b.edge_weight,
                     b.edge_weight, None, gamma=g0)
            Z.append(z)
    return torch.cat(Z).numpy()


def score(Z):
    accs = []
    for tr, te in FOLDS:
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        clf = LinearSVC(dual=False, fit_intercept=True, max_iter=10000)
        clf.fit((Z[tr] - mu) / sd, Y[tr])
        accs.append(balanced_accuracy_score(Y[te], clf.predict((Z[te] - mu) / sd)))
    return float(np.mean(accs)), float(np.std(accs))


targets = sys.argv[1:] or sorted(glob.glob('stage8e/E*') + glob.glob('stage8e/R*'))
print('LOCKED LinearSVC probe -- SECONDARY evidence only. Baselines: FC 0.663, '
      'ALFF 0.591, FC+ALFF 0.657')
print(f"{'arm':24s} {'epoch':>6s} {'bal.acc':>9s} {'sd':>7s}")
out = {}
for d in targets:
    cfgp = f'{d}/config.json'
    if not os.path.exists(cfgp):
        continue
    cfg = json.load(open(cfgp))
    for ep in EPOCHS:
        p = f'{d}/checkpoints/epoch{ep}.pt'
        if not os.path.exists(p):
            print(f"{os.path.basename(d):24s} {ep:6d} {'MISSING':>9s}")
            continue
        a, s = score(embed(p, cfg))
        out[f'{os.path.basename(d)}@{ep}'] = a
        print(f"{os.path.basename(d):24s} {ep:6d} {a:9.4f} {s:7.4f}")
json.dump(out, open('stage8e/svc_probe.json', 'w'), indent=1)
