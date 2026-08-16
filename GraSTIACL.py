import argparse
import logging
import random
import os

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC, SVC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from datasets import ADNIDataset
from datasets import TUEvaluator
from unsupervised.embedding_evaluation import EmbeddingEvaluation, get_emb_y
from unsupervised.utils import initialize_node_features, set_tu_dataset_y_shape
from unsupervised.training import build_model_and_view_learner, train_one_epoch
from numpy import interp



def calc_regloss(z, aug, memory, temperature: float = 0.1, pos_only: bool = False):
    device = z.device
    b = z.size(0)
    z = F.normalize(z, dim=-1)
    aug = F.normalize(aug, dim=-1)
    memory = F.normalize(memory, dim=-1)

    logits = torch.einsum("if, jf -> ij", z, aug) / temperature
    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((b, b), dtype=torch.bool, device=device)
    pos_mask.fill_diagonal_(True)

    m_logits = torch.einsum("if, jf -> ij", z, memory) / temperature
    exp_logits = torch.exp(m_logits)
    log_prob = logits if pos_only else logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)

    loss = -mean_log_prob_pos.mean()

    return loss


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
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ADNIDataset(args.path, args.name, transform=my_transforms, node_feature_mode=args.node_feature_mode)
    num_dataset_features = 93 if args.node_feature_mode == 'alff_pcc' else 3

    dataset.data.y = dataset.data.y.squeeze()


    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False )

    evaluator = TUEvaluator()

    # model (Theta) and view_learner (Phi) are two independent networks, each
    # with its own TAEncoder + ToyNet -- not a shared backbone. The paper
    # defines Phi as "the learnable parameters of the augmenter GNN and MLP"
    # and Theta as "the learnable parameters of the GNN f" (page 6, following
    # Eq. 3, inherited from AD-GCL): two separately-named networks. Eq. 22's
    # min_{Phi,Psi} max_{Theta,Omega} also only makes sense over distinct
    # parameter sets -- a min-max game needs two different players, not one
    # set of weights doing both. gamma_mode picks one of four arms for how
    # lambda_ behaves on the original/eval view (Issue D) -- see
    # unsupervised/training.py's build_model_and_view_learner and
    # TAEncoder's beta_convention docstring for the full derivation.
    # Construction (and the per-epoch training step below) live in
    # unsupervised/training.py -- shared with nested_cv/run_nested_cv.py so
    # every architectural fix only ever needs to happen in one place.
    model, view_learner, model_optimizer, view_optimizer, gamma_orig, beta = build_model_and_view_learner(
        num_dataset_features=num_dataset_features, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
        drop_ratio=args.drop_ratio, pooling_type=args.pooling_type, gamma_mode=args.gamma_mode,
        mij_source=args.mij_source, num_dyn_windows=args.num_dyn_windows, vib_hidden_dim=args.vib_hidden_dim,
        model_lr=args.model_lr, view_lr=args.view_lr, device=device, weight_decay=args.weight_decay,
        enable_attention_mix=not args.replicate_original_code, signed_edges=args.signed_edges)
    if args.freeze_adversary:
        view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=0.0)

    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True, max_iter=10000), evaluator,
                                 dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=True, seed=args.seed)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=True, seed=args.seed)

    model.eval()
    train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, beta, dataset)
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score, test_score))

    if args.supervised_holdout:
        # Arm E: honest 80/20 supervised-contrastive protocol. The encoder
        # trains ONLY on the 80% (labels used by calc_loss_supervised); epoch
        # selection uses 5-fold CV on the 80% side's embeddings only; the 20%
        # is scored once per eval and never influences any decision.
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        y_all = dataset.data.y.cpu().numpy()
        idx_all = np.arange(len(dataset))
        train_idx, test_idx = train_test_split(idx_all, test_size=0.2, stratify=y_all,
                                               random_state=args.seed)
        logging.info("SupervisedHoldout: encoder trains on %d subjects (labels used); "
                     "%d held out untouched", len(train_idx), len(test_idx))
        holdout_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx.tolist()),
                                    batch_size=args.batch_size, shuffle=True, drop_last=True)
        emb_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        val_curve_h, test_curve_h, epochs_h = [], [], []
        for epoch in range(1, args.epochs + 1):
            fin_model_loss, fin_view_loss, fin_reg, _, _, beta = train_one_epoch(
                model, view_learner, model_optimizer, view_optimizer, holdout_loader, device,
                beta, gamma_orig, args.ce_lambda, args.reg_lambda, args.kld_lambda, args.template,
                contrastive_mode="supervised",
                replicate_original_code=args.replicate_original_code,
                epoch_num=epoch, signed_edges=args.signed_edges)
            logging.info('Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(
                epoch, fin_model_loss, fin_view_loss, fin_reg))
            if epoch % args.eval_interval == 0:
                model.eval()
                emb, emb_y = model.encoder.get_embeddings(emb_loader, beta, device)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
                clf = make_pipeline(StandardScaler(),
                                    LinearSVC(dual=False, fit_intercept=True, max_iter=10000))
                val_acc = cross_val_score(clf, emb[train_idx], emb_y[train_idx],
                                          cv=cv, scoring='accuracy').mean()
                clf2 = make_pipeline(StandardScaler(),
                                     LinearSVC(dual=False, fit_intercept=True, max_iter=10000))
                clf2.fit(emb[train_idx], emb_y[train_idx])
                test_acc = clf2.score(emb[test_idx], emb_y[test_idx])
                val_curve_h.append(val_acc)
                test_curve_h.append(test_acc)
                epochs_h.append(epoch)
                logging.info("HOLDOUT epoch %d: val(5-fold on train80)=%.4f "
                             "test(held-out 20)=%.4f", epoch, val_acc, test_acc)
        best_i = int(np.argmax(np.array(val_curve_h)))
        logging.info('FinishedTraining!')
        logging.info('HOLDOUT BestEpoch: {}'.format(epochs_h[best_i]))
        logging.info('BestValidationScore: acc_mean: {} (holdout protocol, train80 5-fold CV)'.format(
            val_curve_h[best_i]))
        logging.info('BestTestScore: acc_mean: {} (held-out 20 percent, selected by train-side val only)'.format(
            test_curve_h[best_i]))
        return val_curve_h[best_i]

    model_losses = []
    view_losses = []
    view_regs = []
    valid_curve = []
    test_curve = []
    train_curve = []
    valid_std_curve = []
    test_std_curve = []
    train_std_curve = []
    valid_f1_curve = []
    test_f1_curve = []
    train_f1_curve = []
    valid_f1_std_curve = []
    test_f1_std_curve = []
    train_f1_std_curve = []
    valid_sen_curve = []
    test_sen_curve = []
    train_sen_curve = []
    valid_sen_std_curve = []
    test_sen_std_curve = []
    train_sen_std_curve = []
    valid_spe_curve = []
    test_spe_curve = []
    train_spe_curve = []
    valid_spe_std_curve = []
    test_spe_std_curve = []
    train_spe_std_curve = []
    train_pre_curve = []
    valid_pre_curve = []
    test_pre_curve = []
    train_pre_std_curve = []
    valid_pre_std_curve = []
    test_pre_std_curve = []

    test_auc_curve = []
    test_auc_std_curve = []
    aug_edge_weights_asd = []
    aug_edge_weights_nc = []
    test_emb_pic = []

    # --early_stop_patience (default 0 = disabled): stop once val accuracy hasn't
    # improved for this many evals. 0 reproduces today's exact behavior. Checked
    # only inside the "if epoch % args.eval_interval == 0" block below, where
    # val_score is freshly computed -- this file's curve-append calls (unlike
    # nested_cv/run_nested_cv.py's) run every epoch off a stale val_score in
    # between evals, so checking improvement every epoch would compare an
    # unchanged value against itself and could stop prematurely.
    best_val_so_far = -1.0
    evals_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        should_stop = False
        fin_model_loss, fin_view_loss, fin_reg, fin_aug_edge_weight_asd, fin_aug_edge_weight_nc, beta = \
            train_one_epoch(model, view_learner, model_optimizer, view_optimizer, dataloader, device,
                             beta, gamma_orig, args.ce_lambda, args.reg_lambda, args.kld_lambda, args.template,
                             replicate_original_code=args.replicate_original_code,
                             epoch_num=epoch, signed_edges=args.signed_edges)
        logging.info(
            'Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(epoch, fin_model_loss,
                                                                   fin_view_loss,
                                                                   fin_reg))
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)
        aug_edge_weights_asd.append(fin_aug_edge_weight_asd)
        aug_edge_weights_nc.append(fin_aug_edge_weight_nc)
        if epoch % args.eval_interval == 0:
            model.eval()
            train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, beta, dataset, flag=True)

            if args.early_stop_patience > 0:
                if val_score[0] > best_val_so_far:
                    best_val_so_far = val_score[0]
                    evals_since_improvement = 0
                else:
                    evals_since_improvement += 1
                if evals_since_improvement >= args.early_stop_patience:
                    logging.info(
                        "Early stop at epoch %d: val accuracy hasn't improved for %d evals "
                        "(best_val=%.4f)" % (epoch, args.early_stop_patience, best_val_so_far))
                    should_stop = True

            logging.info(
                "Metric: {} Train_mean: {} Val_mean: {} Test_mean: {}".format(evaluator.eval_metric, train_score[0],
                                                                              val_score[0],
                                                                              test_score[0]))
            logging.info(
                "Metric: {} Train_std: {} Val_std: {} Test_std: {}".format(evaluator.eval_metric, train_score[1],
                                                                           val_score[1],
                                                                           test_score[1]))
            logging.info(
                "Metric: f1 Train_mean: {} Val_mean: {} Test_mean: {}".format(train_score[2], val_score[2],
                                                                              test_score[2]))

            logging.info(
                "Metric: f1 Train_std: {} Val_std: {} Test_std: {}".format(train_score[3], val_score[3], test_score[3]))

            logging.info(
                "Metric: sen Train_mean: {} Val_mean: {} Test_mean: {}".format(train_score[4], val_score[4],
                                                                               test_score[4]))

            logging.info(
                "Metric: sen Train_std: {} Val_std: {} Test_std: {}".format(train_score[5], val_score[5],
                                                                            test_score[5]))

            logging.info(
                "Metric: spe Train_mean: {} Val_mean: {} Test_mean: {}".format(train_score[6], val_score[6],
                                                                               test_score[6]))

            logging.info(
                "Metric: spe Train_std: {} Val_std: {} Test_std: {}".format(train_score[7], val_score[7],
                                                                            test_score[7]))
            logging.info(
                "Metric: precision Train_mean: {} Val_mean: {} Test_mean: {}".format(train_score[8], val_score[8],
                                                                                     test_score[8]))
            logging.info(
                "Metric: precision Train_std: {} Val_std: {} Test_std: {}".format(train_score[9], val_score[9],
                                                                                  test_score[9]))

        train_f1_curve.append(train_score[2])
        valid_f1_curve.append(val_score[2])
        test_f1_curve.append(test_score[2])
        train_f1_std_curve.append(train_score[3])
        valid_f1_std_curve.append(val_score[3])
        test_f1_std_curve.append(test_score[3])

        train_sen_curve.append(train_score[4])
        valid_sen_curve.append(val_score[4])
        test_sen_curve.append(test_score[4])
        train_sen_std_curve.append(train_score[5])
        valid_sen_std_curve.append(val_score[5])
        test_sen_std_curve.append(test_score[5])

        train_spe_curve.append(train_score[6])
        valid_spe_curve.append(val_score[6])
        test_spe_curve.append(test_score[6])
        train_spe_std_curve.append(train_score[7])
        valid_spe_std_curve.append(val_score[7])
        test_spe_std_curve.append(test_score[7])

        train_curve.append(train_score[0])
        valid_curve.append(val_score[0])
        test_curve.append(test_score[0])
        train_std_curve.append(train_score[1])
        valid_std_curve.append(val_score[1])
        test_std_curve.append(test_score[1])

        train_pre_curve.append(train_score[8])
        valid_pre_curve.append(val_score[8])
        test_pre_curve.append(test_score[8])
        train_pre_std_curve.append(train_score[9])
        valid_pre_std_curve.append(val_score[9])
        test_pre_std_curve.append(test_score[9])

        test_auc_curve.append(test_score[10])
        test_auc_std_curve.append(test_score[11])

        if should_stop:
            break

    # Model/epoch selection uses the validation-accuracy curve ONLY -- never the
    # test curve. Every reported metric, for train/val/test alike, is then read
    # off at this single epoch, so "BestTestScore" describes one real trained
    # checkpoint rather than combining each metric's own independently-chosen
    # (and test-set-derived) peak epoch into a composite that no single model
    # ever actually achieved simultaneously.
    best_val_epoch = np.argmax(np.array(valid_curve))

    logging.info('FinishedTraining!')
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info(
        'BestTrainScore: acc_mean: {} acc_std: {} f1_mean: {} f1_std: {} sen_mean: {} sen_std: {} spe_mean: {} spe_std: {} pre_mean: {} pre_std: {}'.format(
            train_curve[best_val_epoch], train_std_curve[best_val_epoch],
            train_f1_curve[best_val_epoch], train_f1_std_curve[best_val_epoch],
            train_sen_curve[best_val_epoch], train_sen_std_curve[best_val_epoch],
            train_spe_curve[best_val_epoch], train_spe_std_curve[best_val_epoch],
            train_pre_curve[best_val_epoch], train_pre_std_curve[best_val_epoch]))
    logging.info(
        'BestValidationScore: acc_mean: {} acc_std: {} f1_mean: {} f1_std: {} sen_mean: {} sen_std: {} spe_mean: {} spe_std: {} pre_mean: {} pre_std: {}'.format(
            valid_curve[best_val_epoch], valid_std_curve[best_val_epoch],
            valid_f1_curve[best_val_epoch], valid_f1_std_curve[best_val_epoch],
            valid_sen_curve[best_val_epoch], valid_sen_std_curve[best_val_epoch],
            valid_spe_curve[best_val_epoch], valid_spe_std_curve[best_val_epoch],
            valid_pre_curve[best_val_epoch], valid_pre_std_curve[best_val_epoch]))
    logging.info(
        'BestTestScore: acc_mean: {} acc_std: {} f1_mean: {} f1_std: {} sen_mean: {} sen_std: {} spe_mean: {} spe_std: {} pre_mean: {} pre_std: {} auc_mean:{} auc_std:{}'.format(
            test_curve[best_val_epoch], test_std_curve[best_val_epoch],
            test_f1_curve[best_val_epoch], test_f1_std_curve[best_val_epoch],
            test_sen_curve[best_val_epoch], test_sen_std_curve[best_val_epoch],
            test_spe_curve[best_val_epoch], test_spe_std_curve[best_val_epoch],
            test_pre_curve[best_val_epoch], test_pre_std_curve[best_val_epoch],
            test_auc_curve[best_val_epoch], test_auc_std_curve[best_val_epoch]))

    return valid_curve[best_val_epoch]


def arg_parse():
    parser = argparse.ArgumentParser(description='GraSTI-ACL ADNI')

    parser.add_argument('--name', type=str, default='GraSTI-ACL',
                        help='dataset.')
    parser.add_argument('--path', type=str, default='',
                        help='path of dataset.')
    parser.add_argument('--template', type=int, default=90,
                        help='dataset template.')
    parser.add_argument('--num_dyn_windows', type=int, default=3,
                        help='number of local/dynamic PCC windows (T in Eq. 13).')
    parser.add_argument('--model_lr', type=float, default=0.0005,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.0005,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=2,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--vib_hidden_dim', type=int, default=400,
                        help='max length of memory bank')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.3,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Train Epochs')
    parser.add_argument('--kld_lambda', default=0.003, type=float,
                        help='Regularization coefficients for loss of KL diverse')
    parser.add_argument('--reg_lambda', default=1.0, type=float,
                        help='Penalty on mean edge-drop rate in view_loss (prevents the view_learner from maximizing loss by destroying the whole graph)')
    parser.add_argument('--gamma_mode', type=str, default='baseline',
                        choices=['baseline', 'literal_beta', 'signal_strength', 'paper_literal'],
                        help='Issue D: how lambda_ behaves for the original/eval view. baseline=gamma_orig=1.0 '
                             'with reversed Beta (attention~dead for this view). literal_beta=Option A, Eq.18 '
                             'literal symbol order (attention alive, contradicts Sec 3.3 prose). '
                             'signal_strength=Option B, keep reversed Beta but gamma_orig=mean(|W|). '
                             'paper_literal=both together, literal Beta AND gamma_orig=mean(|W|) -- the '
                             'combination actually specified verbatim by the paper\'s own text.')
    parser.add_argument('--node_feature_mode', type=str, default='alff',
                        choices=['alff', 'alff_pcc', 'alff_raw', 'alff_paper'],
                        help='alff=LEGACY old node-feature cache: DPARSF voxelwise mALFF -> ROI averaging -> '
                             'per-subject per-band z-scoring, [90,3] (default, kept for backward '
                             'compatibility with recorded results -- NOT raw ALFF despite the name). '
                             'alff_raw=raw ROI-first 3-band ALFF from alff_new.npz, un-normalised, [90,3]. '
                             'alff_paper=Stage-1 validated ROI-first raw 3-band ALFF followed by ONE shared '
                             'per-subject [0,1] min-max over the whole [90,3] (A-GCL Sec 2.1 exact). '
                             'alff_pcc=x is [min-max ALFF ->[0,1] ; min-max own PCC row ->[-1,1]], [90,93] -- '
                             'each node also gets its own connectivity profile as input content, not just as '
                             'an edge weight (BrainGNN\'s established practice). Population-wide min/max, '
                             'fit once across all subjects, not per-subject.')
    parser.add_argument('--mij_source', type=str, default='alff',
                        choices=['alff', 'alff_pcc'],
                        help='Which slice of x feeds M_ij (Eq. 4)\'s dot product. alff=x[:,:3], raw ALFF only -- '
                             'literal Eq. 4 (d=3), default. alff_pcc=full x (all 93 columns when '
                             'node_feature_mode=alff_pcc) -- deliberate deviation from Eq. 4, mirroring the same '
                             'alff->alff_pcc enrichment already applied to the main node features. No effect when '
                             'node_feature_mode=alff (x is already [N,3] either way).')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear",
                        help="Downstream classifier is linear or non-linear")
    parser.add_argument('--ce_lambda', type=float, default=2.0,
                        help='Regularization coefficients for loss of cross entrpy')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 penalty on both optimizers (model_optimizer and view_optimizer). Default 0.0 '
                             '(PyTorch Adam default, no change to existing behavior). Not tested/specified by '
                             'the paper -- same addition already applied and verified in '
                             'nested_cv/run_nested_cv.py, targeting the overfitting pattern diagnosed this '
                             'session (train accuracy climbing while val/test stay flat).')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop once val accuracy hasn\'t improved for this many evals (eval_interval apart). '
                             'Default 0 = disabled, reproduces existing behavior exactly (always trains the '
                             'full --epochs). Same addition already applied and verified in '
                             'nested_cv/run_nested_cv.py.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--freeze_adversary', action='store_true',
                        help='Pin the view_learner at its initialization (optimizer lr=0.0; every other '
                             'computation identical). Named deviation from the paper -- equivalent to '
                             'random-augmentation contrastive learning. Motivated by the overfit-16 campaign: '
                             'live-adversary arms peak at epoch 0 and decay to uniformity (eval contrastive '
                             '= ln(batch-1) exactly); the frozen arm is the only configuration that learns '
                             '(alignment 11-12/16, still improving at epoch 100).')
    parser.add_argument('--signed_edges', action='store_true',
                        help='Feed the RAW SIGNED PCC to the GCN (anticorrelation preserved) using the '
                             'signed_safe normalization (degree from |w|, signed weight kept in message '
                             'passing). Without signed_safe this would NaN on negative degrees.')
    parser.add_argument('--supervised_holdout', action='store_true',
                        help='Arm E protocol: stratified 80/20 split; encoder trains with the supervised '
                             'contrastive loss on the 80 percent (labels used); epoch selection by 5-fold CV '
                             'on the 80 percent side only; the 20 percent is scored once per eval and never '
                             'influences any choice. Replaces the transductive kf evaluation entirely.')
    parser.add_argument('--replicate_original_code', action='store_true',
                        help='Match the authors\' released code on the remaining verified differences: (1) the '
                             'attention-branch mixup (Eq. 19) is computed but never applied there -- pooling runs '
                             'on the GCN branch alone; (2) ce_loss is added to model_loss in Phase 2 (the model '
                             'reduces it), not subtracted from view_loss in Phase 1. NOTE: the multiplicative '
                             'augmented edge weight (weight * gate) is now UNIVERSAL semantics (Stage 2 Fix A), '
                             'no longer controlled by this flag. Default False.')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    run(args)

