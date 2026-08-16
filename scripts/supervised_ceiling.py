"""T1: supervised encoder ceiling test.

Removes the entire self-supervised game (no ViewLearner, no ToyNet, no
augmentation, no contrastive loss): TAEncoder (mix OFF) + one linear head,
trained end-to-end with cross-entropy on the diagnosis label. Answers the one
question no prior number answers: with labels handed to it directly, how much
can this encoder extract?

Protocol (honest by construction):
  - Stratified 5-fold outer CV; per fold, the training portion splits again
    (stratified 80/20) into train/val.
  - Encoder+head train on train only; best epoch chosen on val; the test fold
    scored at every eval but READ only at the best-val epoch.
  - Per fold, train AND test are reported (the train-test gap is half the
    diagnosis: high/low = regularization problem; low/low = capability problem).
  - --shuffle_labels permutes labels ONCE (fixed rng) BEFORE folding, so folds
    stratify on the shuffled labels -- the leak detector must land ~52%.

Reading rules (pre-registered): check T1c FIRST (if >~52%, all arms void).
T1d (raw ALFF, d=3) is read against T1a (same d, different normalization),
not against T1b. Benchmarks: chance 52.4 | classical ALFF 57-59 | classical
PCC 64-68.
"""
import argparse
import logging
import sys

sys.path.insert(0, '/users/3171356m/muhammad/GraSTIACL')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets import ADNIDataset
from unsupervised.encoder import TA_encoder
from unsupervised.training import init_module_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class SupervisedGraphClassifier(nn.Module):
    def __init__(self, emb_dim, num_gc_layers, in_dim=3):
        super().__init__()
        self.encoder = TA_encoder.TAEncoder(
            num_dataset_features=in_dim, beta=0.5, emb_dim=emb_dim, num_gc_layers=num_gc_layers,
            drop_ratio=0.3, pooling_type='standard',
            beta_convention='literal', gamma_orig_mode='signal_strength',
            enable_attention_mix=False)
        self.head = nn.Linear(emb_dim, 2)

    def forward(self, batch, x, edge_index, edge_weight):
        # mix OFF -> gamma machinery skipped inside forward; gamma value unused.
        z, _ = self.encoder(batch, x, edge_index, 0.5, edge_weight, gamma=1.0)
        return self.head(z)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            gcn_w = b.edge_weight.abs().clamp(1e-6, 1.0)
            logits = model(b.batch, b.x, b.edge_index, gcn_w)
            pred = logits.argmax(dim=1)
            correct += int((pred == b.y.view(-1)).sum())
            total += b.num_graphs
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--node_feature_mode', choices=['alff', 'alff_pcc', 'alff_raw', 'alff_paper'], default='alff')
    ap.add_argument('--shuffle_labels', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--eval_interval', type=int, default=5)
    ap.add_argument('--emb_dim', type=int, default=32)
    ap.add_argument('--num_gc_layers', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("T1 supervised ceiling: mode=%s shuffle=%s device=%s args=%s",
                 args.node_feature_mode, args.shuffle_labels, device, vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = ADNIDataset('data/GraSTIACL_ABIDE_979', 'GraSTIACL_ABIDE_979',
                          node_feature_mode=args.node_feature_mode)
    if args.node_feature_mode == 'alff_pcc':
        in_dim = 93
    else:
        in_dim = 3

    y = np.array([int(dataset[i].y.item()) for i in range(len(dataset))])
    if args.shuffle_labels:
        # Leak detector: permute ONCE before folding; folds stratify on the
        # shuffled labels. Any score meaningfully above chance = harness leak.
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(y))
        y_used = y[perm]
        # write shuffled labels onto the dataset copies used for training
        label_lookup = {i: int(y_used[i]) for i in range(len(y_used))}
        logging.info("shuffle_labels ON: label agreement with truth = %.4f "
                     "(expect ~0.5)", float((y_used == y).mean()))
    else:
        y_used = y
        label_lookup = None

    class Relabel:
        """Wraps a subject index list into a dataset view with (possibly
        shuffled) labels applied at access time -- the cached dataset on disk
        is never modified."""
        def __init__(self, indices):
            self.indices = list(indices)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, k):
            d = dataset[self.indices[k]].clone()
            if label_lookup is not None:
                d.y = torch.tensor([label_lookup[self.indices[k]]])
            return d

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_train, fold_val, fold_test, fold_best_ep = [], [], [], []

    for fold, (train_val_idx, test_idx) in enumerate(outer.split(np.zeros(len(y_used)), y_used)):
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        tr_idx, va_idx = train_test_split(
            train_val_idx, test_size=0.2, stratify=y_used[train_val_idx],
            random_state=args.seed + fold)

        train_loader = DataLoader(Relabel(tr_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Relabel(va_idx), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(Relabel(test_idx), batch_size=args.batch_size, shuffle=False)
        train_eval_loader = DataLoader(Relabel(tr_idx), batch_size=args.batch_size, shuffle=False)

        model = SupervisedGraphClassifier(args.emb_dim, args.num_gc_layers, in_dim=in_dim)
        init_module_weights(model)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val = -1.0
        best_ep = 0
        test_at_best = train_at_best = 0.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for b in train_loader:
                b = b.to(device)
                gcn_w = b.edge_weight.abs().clamp(1e-6, 1.0)
                logits = model(b.batch, b.x, b.edge_index, gcn_w)
                loss = F.cross_entropy(logits, b.y.view(-1))
                assert torch.isfinite(loss), f"non-finite loss at fold {fold} epoch {epoch}"
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % args.eval_interval == 0:
                va = evaluate(model, val_loader, device)
                te = evaluate(model, test_loader, device)
                tr = evaluate(model, train_eval_loader, device)
                logging.info("fold %d epoch %d: train=%.4f val=%.4f test=%.4f",
                             fold, epoch, tr, va, te)
                if va > best_val:
                    best_val, best_ep = va, epoch
                    test_at_best, train_at_best = te, tr

        fold_train.append(train_at_best)
        fold_val.append(best_val)
        fold_test.append(test_at_best)
        fold_best_ep.append(best_ep)
        logging.info("FOLD %d RESULT: best_ep=%d train=%.4f val=%.4f test=%.4f",
                     fold, best_ep, train_at_best, best_val, test_at_best)

    logging.info("T1SUMMARY mode=%s shuffle=%s train=%.4f+-%.4f val=%.4f+-%.4f "
                 "test=%.4f+-%.4f best_eps=%s",
                 args.node_feature_mode, args.shuffle_labels,
                 float(np.mean(fold_train)), float(np.std(fold_train)),
                 float(np.mean(fold_val)), float(np.std(fold_val)),
                 float(np.mean(fold_test)), float(np.std(fold_test)),
                 fold_best_ep)


if __name__ == '__main__':
    main()
