"""Nested-CV deep-pipeline training, with an optional per-outer-fold ComBat
harmonization toggle (--combat).

Adapts GraSTIACL.py's run() (same two-phase adversarial training loop, same
model construction) to one outer fold at a time: the shared encoder/net are
freshly initialized and trained using ONLY this fold's train subjects (never
the held-out test subjects, harmonized or not -- this is what makes it
"nested", and also what fixes the transductive-encoder concern that the plain
single-run jobs in scripts/ still carry). Evaluation reads a single train/val/
test split (train/val from an 80/20 split of the fold's train subjects, test =
the fold's true held-out subjects) via EmbeddingEvaluation.embedding_evaluation,
not kf_embedding_evaluation's internal K-fold (which would defeat the point of
having a fixed outer fold at all).

One process = one (emb_dim, batch_size, fold, combat) run. Run all 5 folds for
a given config (as separate SLURM array tasks -- see run_*_array.sh), then use
aggregate_results.py to combine them into the mean/std per config.
"""
import argparse
import json
import logging
import os
import os.path as osp
import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC, SVC
from torch_geometric.loader import DataLoader

from datasets import TUEvaluator
from unsupervised.embedding_evaluation import EmbeddingEvaluation
from unsupervised.training import build_model_and_view_learner, train_one_epoch
from supervised.sampler import ClassBalancedBatchSampler

from nested_cv.data import load_all_subjects, harmonize_fold, build_windowed_data_list, compute_alff_pcc_scale_stats

RESULTS_DIR = osp.join(osp.dirname(__file__), "results")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d, Fold: %d, ComBat: %s" % (args.seed, args.fold, args.combat))
    logging.info(args)
    setup_seed(args.seed)

    evaluator = TUEvaluator()

    logging.info("Loading raw subject data...")
    subject_ids, x_all, ew_all, dw_all, y_all, covars = load_all_subjects()
    logging.info("Loaded %d subjects (%d ASD, %d NC)" % (len(subject_ids), int((y_all == 1).sum()), int((y_all == 0).sum())))

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(outer_cv.split(np.arange(len(y_all)), y_all))
    train_val_idx, test_idx = splits[args.fold]

    x_h, ew_h, dw_h = harmonize_fold(x_all, ew_all, dw_all, covars, train_val_idx, test_idx, combat_on=args.combat)

    # Fixed 80/20 train/val split of this fold's train pool, held constant for
    # the whole run (not re-drawn per eval_interval) -- val subjects still
    # never appear in test_idx, and the encoder never trains on test_idx.
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=args.seed, stratify=y_all[train_val_idx],
    )

    num_dataset_features = 93 if args.node_feature_mode == 'alff_pcc' else 3
    scale_stats = {}
    if args.node_feature_mode == 'alff_pcc':
        # Fit on train_idx only (this fold's train pool, pre-80/20-split --
        # matches ComBat's own train_val_idx/test_idx boundary above), then
        # apply unchanged to train/val/test alike. Unlike the single-run
        # ADNIDataset pipeline (population-wide across every subject), this
        # keeps the fold's true held-out test subjects from influencing the
        # scaling stats at all.
        alff_min, alff_max, pcc_min, pcc_max = compute_alff_pcc_scale_stats(x_h, ew_h, train_val_idx)
        scale_stats = dict(node_feature_mode='alff_pcc', alff_min=alff_min, alff_max=alff_max,
                            pcc_min=pcc_min, pcc_max=pcc_max)
        logging.info("alff_pcc scale stats (train_val_idx only): alff=[%.4f,%.4f] pcc=[%.4f,%.4f]" %
                     (alff_min, alff_max, pcc_min, pcc_max))

    train_dataset = build_windowed_data_list(train_idx, x_h, ew_h, dw_h, y_all, **scale_stats)
    val_dataset = build_windowed_data_list(val_idx, x_h, ew_h, dw_h, y_all, **scale_stats)
    test_dataset = build_windowed_data_list(test_idx, x_h, ew_h, dw_h, y_all, **scale_stats)
    logging.info("Train/Val/Test sizes: %d/%d/%d" % (len(train_dataset), len(val_dataset), len(test_dataset)))

    if args.contrastive_mode == "supervised":
        # SupCon's positive sets (same-class peers within a batch) need
        # every batch to reliably contain both classes -- plain shuffle=True
        # gives no such guarantee. train_idx's own labels (not the full
        # y_all) since batch_sampler indexes into train_dataset, which is
        # itself already ordered by train_idx.
        train_labels = y_all[train_idx]
        batch_sampler = ClassBalancedBatchSampler(train_labels, args.batch_size, seed=args.seed)
        dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Construction (and the per-epoch training step below) live in
    # unsupervised/training.py -- shared with GraSTIACL.py so every
    # architectural fix (separate encoders, gamma_mode, reg_lambda,
    # mij_source, Item 4a/4b's L_CE placement, dyn_weight=None speedup,
    # gradient clipping) only ever needs to happen in one place.
    model, view_learner, model_optimizer, view_optimizer, gamma_orig, beta = build_model_and_view_learner(
        num_dataset_features=num_dataset_features, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
        drop_ratio=args.drop_ratio, pooling_type=args.pooling_type, gamma_mode=args.gamma_mode,
        mij_source=args.mij_source, num_dyn_windows=args.num_dyn_windows, vib_hidden_dim=args.vib_hidden_dim,
        model_lr=args.model_lr, view_lr=args.view_lr, device=device, weight_decay=args.weight_decay)

    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True, max_iter=10000), evaluator,
                                 "classification", 1, device, param_search=True, seed=args.seed)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, "classification", 1, device, param_search=True, seed=args.seed)

    valid_curve, test_curve, train_curve = [], [], []
    valid_f1_curve, test_f1_curve, train_f1_curve = [], [], []
    valid_sen_curve, test_sen_curve, train_sen_curve = [], [], []
    valid_spe_curve, test_spe_curve, train_spe_curve = [], [], []
    valid_pre_curve, test_pre_curve, train_pre_curve = [], [], []
    test_auc_curve = []
    reg_curve = []

    for epoch in range(1, args.epochs + 1):
        # beta returned here is the real, adaptive fin_reg-derived value
        # (fixes the previously-disclosed dead-parameter Issue #9 for free --
        # this file used to draw a meaningless fixed Beta(0.5,0.5) every
        # epoch instead).
        fin_model_loss, fin_view_loss, fin_reg, _, _, beta = train_one_epoch(
            model, view_learner, model_optimizer, view_optimizer, dataloader, device,
            beta, gamma_orig, args.ce_lambda, args.reg_lambda, args.kld_lambda, args.template,
            contrastive_mode=args.contrastive_mode, supervised_temperature=args.supervised_temperature)
        logging.info('Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(
            epoch, fin_model_loss, fin_view_loss, fin_reg))

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            model.eval()
            (train_score, val_score, test_score, train_f1, val_f1, test_f1,
             train_sen, val_sen, test_sen, train_spe, val_spe, test_spe,
             train_pre, val_pre, test_pre, test_auc, running_time) = ee.embedding_evaluation(
                model.encoder, beta, dataloader, val_loader, test_loader, flag=True)

            logging.info("Epoch {}: Train_acc {} Val_acc {} Test_acc {} Test_auc {}".format(
                epoch, train_score, val_score, test_score, test_auc))

            train_curve.append(train_score); valid_curve.append(val_score); test_curve.append(test_score)
            train_f1_curve.append(train_f1); valid_f1_curve.append(val_f1); test_f1_curve.append(test_f1)
            train_sen_curve.append(train_sen); valid_sen_curve.append(val_sen); test_sen_curve.append(test_sen)
            train_spe_curve.append(train_spe); valid_spe_curve.append(val_spe); test_spe_curve.append(test_spe)
            train_pre_curve.append(train_pre); valid_pre_curve.append(val_pre); test_pre_curve.append(test_pre)
            test_auc_curve.append(test_auc)
            reg_curve.append(fin_reg)

    best_val_epoch = int(np.argmax(np.array(valid_curve)))
    logging.info('FinishedTraining! BestEpoch (eval-interval index): {}'.format(best_val_epoch))

    result = {
        "emb_dim": args.emb_dim,
        "batch_size": args.batch_size,
        "fold": args.fold,
        "contrastive_mode": args.contrastive_mode,
        "weight_decay": args.weight_decay,
        "combat": args.combat,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "best_val_epoch_idx": best_val_epoch,
        "train_accuracy": train_curve[best_val_epoch],
        "val_accuracy": valid_curve[best_val_epoch],
        "test_accuracy": test_curve[best_val_epoch],
        "train_f1": train_f1_curve[best_val_epoch],
        "val_f1": valid_f1_curve[best_val_epoch],
        "test_f1": test_f1_curve[best_val_epoch],
        "train_sensitivity": train_sen_curve[best_val_epoch],
        "val_sensitivity": valid_sen_curve[best_val_epoch],
        "test_sensitivity": test_sen_curve[best_val_epoch],
        "train_specificity": train_spe_curve[best_val_epoch],
        "val_specificity": valid_spe_curve[best_val_epoch],
        "test_specificity": test_spe_curve[best_val_epoch],
        "train_precision": train_pre_curve[best_val_epoch],
        "val_precision": valid_pre_curve[best_val_epoch],
        "test_precision": test_pre_curve[best_val_epoch],
        "test_auc": test_auc_curve[best_val_epoch],
        "fin_reg": reg_curve[best_val_epoch],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    combat_tag = "combat" if args.combat else "nocombat"
    out_path = osp.join(
        RESULTS_DIR, f"{combat_tag}__emb{args.emb_dim}_bs{args.batch_size}__fold{args.fold}.json",
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info("Wrote %s" % out_path)
    logging.info(json.dumps(result, indent=2))
    return result


def arg_parse():
    parser = argparse.ArgumentParser(description='GraSTI-ACL nested-CV (+ optional ComBat)')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                        help='outer fold index (5-fold StratifiedKFold)')
    parser.add_argument('--combat', action='store_true',
                        help='apply per-fold ComBat harmonization (fit on train, apply to test)')
    parser.add_argument('--num_dyn_windows', type=int, default=3)
    parser.add_argument('--model_lr', type=float, default=0.0005)
    parser.add_argument('--view_lr', type=float, default=0.0005)
    parser.add_argument('--num_gc_layers', type=int, default=2)
    parser.add_argument('--pooling_type', type=str, default='standard')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--vib_hidden_dim', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--drop_ratio', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--kld_lambda', default=0.003, type=float)
    parser.add_argument('--reg_lambda', default=0.2, type=float,
                        help='Penalty on mean edge-drop rate in view_loss -- same role as in GraSTIACL.py.')
    parser.add_argument('--gamma_mode', type=str, default='baseline',
                        choices=['baseline', 'literal_beta', 'signal_strength', 'paper_literal'],
                        help='Same four arms as GraSTIACL.py --gamma_mode (Issue D).')
    parser.add_argument('--node_feature_mode', type=str, default='alff',
                        choices=['alff', 'alff_pcc'],
                        help='Same as GraSTIACL.py --node_feature_mode. alff_pcc scale stats are fit on this '
                             'fold\'s train_val_idx only, never test_idx (see compute_alff_pcc_scale_stats).')
    parser.add_argument('--mij_source', type=str, default='alff',
                        choices=['alff', 'alff_pcc'],
                        help='Same as GraSTIACL.py --mij_source.')
    parser.add_argument('--template', type=int, default=90,
                        help='Dataset template (ROI count) -- same as GraSTIACL.py --template.')
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--downstream_classifier', type=str, default="linear")
    parser.add_argument('--ce_lambda', type=float, default=2.0)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 penalty on both optimizers (model_optimizer and view_optimizer). Default 0.0 '
                             '(PyTorch Adam default, no change to existing behavior). Not tested/specified by '
                             'the paper -- our own addition, targeting the overfitting pattern diagnosed this '
                             'session (train accuracy hitting 100%% while val/test stay flat).')
    parser.add_argument('--contrastive_mode', type=str, default='self_supervised',
                        choices=['self_supervised', 'supervised'],
                        help='self_supervised=current behavior (GInfoMinMax.calc_loss, both phases). '
                             'supervised=Khosla et al. 2020 SupCon (L^sup_out, supervised/loss.py), applied '
                             'ONLY in Phase 2 (model_loss) -- Phase 1 (view_loss/view_learner) always stays on '
                             'the self-supervised loss regardless of this flag (see supervised/loss.py docstring '
                             'for why). Also switches the train DataLoader to ClassBalancedBatchSampler.')
    parser.add_argument('--supervised_temperature', type=float, default=0.1,
                        help='Temperature for the supervised contrastive loss when --contrastive_mode=supervised. '
                             'Khosla et al.\'s own validated value (Sec 4.5: "All our results used tau=0.1"), '
                             'distinct from the self-supervised calc_loss default of 0.2. No effect when '
                             'contrastive_mode=self_supervised.')
    parser.add_argument('--seed', type=int, default=123)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    run(args)
