"""Shared model construction + one-epoch training step for the two-phase
adversarial contrastive loop, used by both GraSTIACL.py (single transductive
run) and nested_cv/run_nested_cv.py (per-fold retraining). Extracted so every
architectural fix (separate encoders, gamma_mode, reg_lambda, mij_source,
Item 4a/4b's L_CE placement, the dyn_weight=None speedup, gradient clipping)
lives in exactly one place instead of being duplicated -- and drifting -- across
two copies of the same loop.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter


def _guard_nan(tensor, name, epoch_num=None):
    # ToyNet's edge_logits going NaN after ~70-95 epochs of adversarial ascent
    # is a real, observed failure mode (crashed jobs 1867590, 1867597 via a
    # CUDA assertion inside binary_cross_entropy -- sigmoid(NaN) is NaN, and
    # NaN fails BCE's [0,1] range check). Guards Inf as well, not just NaN --
    # weight drift produces inf BEFORE nan (inf - inf = nan), so an isnan-only
    # guard misses the first stage of the failure. Detect + log loudly (with
    # the epoch, so "faithful up to epoch N" is writable in any report), then
    # sanitize to a neutral value rather than silently continuing or crashing.
    # 0.0 is a neutral logit (sigmoid(0)=0.5, "50/50 keep or drop").
    n_bad = (~torch.isfinite(tensor)).sum().item()
    if n_bad > 0:
        epoch_str = f"epoch {epoch_num}: " if epoch_num is not None else ""
        logging.warning(
            "%sNon-finite (NaN/Inf) detected in %s: %d of %d values -- sanitizing to 0.0 "
            "(neutral logit). Results derived from this run are faithful only up to this event.",
            epoch_str, name, n_bad, tensor.numel())
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor

from unsupervised.encoder import TA_encoder
from unsupervised.learning import GInfoMinMax
from unsupervised.learning import GraSTI
from unsupervised.view_learner import ViewLearner
from supervised.loss import calc_loss_supervised


def init_module_weights(module):
    # Xavier-initializes a module's own Linear layers exactly once. Used for
    # each of model_encoder/model_net/view_encoder/view_net -- each wrapper's
    # own init_emb() only touches its own private head (proj_head /
    # mlp_edge_model), so the encoder/net backbones need this explicit call.
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


def build_model_and_view_learner(num_dataset_features, emb_dim, num_gc_layers, drop_ratio,
                                  pooling_type, gamma_mode, mij_source, num_dyn_windows,
                                  vib_hidden_dim, model_lr, view_lr, device, weight_decay=0.0,
                                  enable_attention_mix=True, signed_edges=False):
    # model (Theta) and view_learner (Phi) are two independent networks, each
    # with its own TAEncoder + ToyNet -- not a shared backbone (Issue A).
    beta_convention = "literal" if gamma_mode in ("literal_beta", "paper_literal") else "reversed"
    gamma_orig_mode = "signal_strength" if gamma_mode in ("signal_strength", "paper_literal") else "one"
    gamma_orig = None if gamma_orig_mode == "signal_strength" else 1.0

    beta = 0.5  # TransConv's own weighting scalar -- unrelated to beta_convention's Beta distribution.

    model_encoder = TA_encoder.TAEncoder(num_dataset_features=num_dataset_features, beta=beta, emb_dim=emb_dim,
                                         num_gc_layers=num_gc_layers, drop_ratio=drop_ratio,
                                         pooling_type=pooling_type,
                                         beta_convention=beta_convention, gamma_orig_mode=gamma_orig_mode,
                                         enable_attention_mix=enable_attention_mix, signed_safe=signed_edges)
    model_net = GraSTI.ToyNet(input_dim=90 * (1 + num_dyn_windows), hidden_dim=vib_hidden_dim)
    init_module_weights(model_encoder)
    init_module_weights(model_net)

    view_encoder = TA_encoder.TAEncoder(num_dataset_features=num_dataset_features, beta=beta, emb_dim=emb_dim,
                                        num_gc_layers=num_gc_layers, drop_ratio=drop_ratio,
                                        pooling_type=pooling_type,
                                        beta_convention=beta_convention, gamma_orig_mode=gamma_orig_mode,
                                        enable_attention_mix=enable_attention_mix, signed_safe=signed_edges)
    view_net = GraSTI.ToyNet(input_dim=90 * (1 + num_dyn_windows), hidden_dim=vib_hidden_dim)
    init_module_weights(view_encoder)
    init_module_weights(view_net)

    model = GInfoMinMax(model_encoder, model_net, emb_dim).to(device)
    view_learner = ViewLearner(view_encoder, view_net, mij_source=mij_source).to(device)

    # Two independent optimizers -- each network only ever moves in its own
    # one direction (view_learner always ascends, model always descends), so
    # there is no shared tensor for two optimizers to disagree about.
    # weight_decay applied to BOTH (unlike the SupCon loss change, which is
    # deliberately asymmetric between phases): weight_decay doesn't change
    # *what* either network optimizes for, just discourages weights from
    # growing arbitrarily large regardless of objective -- so it isn't handing
    # the adversary a new attack vector the way a shared supervised loss
    # would. It also directly complements the gradient-clipping fix already
    # in Phase 1 below, both addressing the same unconstrained-ascent
    # instability from a different angle.
    model_optimizer = torch.optim.Adam(model.parameters(), lr=model_lr, weight_decay=weight_decay)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=view_lr, weight_decay=weight_decay)

    return model, view_learner, model_optimizer, view_optimizer, gamma_orig, beta


def train_one_epoch(model, view_learner, model_optimizer, view_optimizer, dataloader, device,
                     beta, gamma_orig, ce_lambda, reg_lambda, kld_lambda, template,
                     contrastive_mode="self_supervised", supervised_temperature=0.1,
                     replicate_original_code=False, epoch_num=None, signed_edges=False):
    # signed_edges: feed the RAW SIGNED PCC to the GCN. Only valid when the
    # encoders were built with signed_safe=True: DEGREE normalization uses
    # |w| (D_i = sum_j |W_ij|) while message passing retains the SIGNED W_ij
    # (+0.8 -> positive message, -0.8 -> negative message). Without
    # signed_safe, literal signed normalization produces NaN degrees on 246
    # of 956 subjects (measured). Note the distinction: abs() here is for
    # DEGREE normalization only -- it is NOT abs(PCC)-as-edge-weight, which
    # is the default profile's separate, documented choice.
    # replicate_original_code now controls ONE remaining author-code
    # difference: ce_loss is added to model_loss in Phase 2 there (the model
    # is pushed to reduce it), not subtracted from view_loss in Phase 1.
    # The multiplicative augmented edge weight (weight * gate) that this flag
    # previously also controlled is now UNIVERSAL graph semantics (Stage 2
    # Fix A) -- matching both Sec. 3.2's "weaken their connections" prose and
    # the authors' released code.
    model_loss_all = 0
    view_loss_all = 0
    reg_all = 0
    num_graphs_seen = 0
    num_graph_with_edges_seen = 0
    # Accumulated separately per diagnostic group (not pooled) so the two
    # group-level averages can later be contrasted (Fig. 3/4 interpretability).
    aug_edge_weight_all_asd = torch.zeros(template, template)
    aug_edge_weight_all_nc = torch.zeros(template, template)
    n_asd_seen = 0
    n_nc_seen = 0

    for batch in dataloader:
        batch = batch.to(device)
        # The contrastive loss (calc_loss) normalizes each positive pair
        # against the OTHER batch members (sum - pos_sim); with a single-graph
        # batch that denominator is exactly 0 -> division by zero -> inf/NaN
        # loss with no error raised. DataLoader(drop_last=True) usually
        # prevents this, but nothing enforced it structurally until now.
        assert batch.num_graphs >= 2, (
            f"train_one_epoch got a batch with {batch.num_graphs} graph(s); "
            "the in-batch contrastive loss needs >= 2 (its denominator is the "
            "other batch members). Use drop_last=True or a batch size that "
            "divides the dataset accordingly.")
        if signed_edges:
            # RAW signed PCC, unclamped: range certified within [-1,1] at the
            # dataset layer (956-subject diagnostic: min -0.760, max 1.000);
            # silent clamping would hide a corrupted source instead of failing.
            gcn_w = batch.edge_weight
        else:
            gcn_w = batch.edge_weight.abs().clamp(1e-6, 1.0)

        # ---- Phase 1: train view_learner to maximize contrastive loss ----
        view_learner.train()
        view_optimizer.zero_grad()
        model.eval()

        # dyn_weight=None: model's own edge_logits/mu/std from get_mu_std_logits
        # are never used downstream -- passing None skips ToyNet's expensive
        # batch_size x 90 row-by-row loop entirely for this call.
        x, _ = model(batch.batch, batch.x, batch.edge_index, beta, None, gcn_w,
                    batch.edge_weight, None, gamma=gamma_orig)
        # edge_logits_vl (not model's edge_logits) drives the gate below --
        # view_learner IS the augmenter (Eq. 3's Phi), so the augmentation it
        # produces must come from its own forward pass.
        edge_logits_vl, mu, std, edge_prod = view_learner(batch.batch, batch.x, batch.edge_index, beta, None,
                                             gcn_w, batch.edge_weight, batch.dyn_weight, gamma=gamma_orig)
        edge_logits_vl = _guard_nan(edge_logits_vl, "edge_logits_vl (Phase 1)", epoch_num)
        edge_prod = _guard_nan(edge_prod, "edge_prod (Phase 1)", epoch_num)
        temperature = 1
        bias = 0.0 + 0.0001
        eps = (bias - (1 - bias)) * torch.rand(edge_logits_vl.size()) + (1 - bias)
        gate_inputs_ = torch.log(eps) - torch.log(1 - eps)
        # Reassigned in place -- ce_loss's use of gate_inputs_ below must also
        # be on the right device (this is what crashed on GPU before the fix).
        gate_inputs_ = gate_inputs_.to(device)
        gate_inputs = (gate_inputs_ + edge_logits_vl) / temperature
        # Eq. (12) produces the RETENTION GATE A in [0,1]. The gate is a mask,
        # not an edge weight: the augmented GCN edge weight is the original
        # connectivity WEAKENED by the gate (Sec. 3.2: "instead of discarding
        # certain edges, we weaken their connections"), i.e. gcn_w * gate --
        # universal semantics for every profile, matching the authors' released
        # code (no longer controlled by replicate_original_code).
        gate = torch.sigmoid(gate_inputs).squeeze()
        batch_aug_edge_weight = gcn_w * gate
        # gamma = PER-SUBJECT mean of the GATE (retention information), NOT
        # mean(gcn_w * gate) -- after multiplication that product is
        # connectivity-strength x retention, a different quantity from Eq. 18's
        # retained-edge ratio. Per-subject, never batch-wide (measured effect
        # of the batch-wide version: 85.5% of a typical between-subject
        # distance, confirmed gamma-mediated).
        gamma_aug = gate.view(batch.num_graphs, -1).mean(dim=1)
        x_aug, _ = model(batch.batch, batch.x, batch.edge_index, beta, None, batch_aug_edge_weight,
                        batch.edge_weight, None, gamma=gamma_aug)
        row, col = batch.edge_index
        edge_batch = batch.batch[row]
        # gate = retention probability; 1-gate = drop probability. fin_reg
        # (its mean) keeps its historical meaning exactly -- this is a reuse of
        # the already-computed gate, not a semantic change.
        edge_drop_out_prob = 1 - gate

        uni, edge_batch_num = edge_batch.unique(return_counts=True)
        sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")
        reg = []
        for b_id in range(batch.num_graphs):
            if b_id in uni:
                num_edges = edge_batch_num[uni.tolist().index(b_id)]
                reg.append(sum_pe[b_id] / num_edges)
            # else: no edges in that graph -- don't include.
        num_graph_with_edges = len(reg)
        reg = torch.stack(reg)
        reg = reg.mean()

        mu = torch.reshape(mu, [batch.num_graphs, -1])
        std = torch.reshape(std, [batch.num_graphs, -1])
        kld_loss = - 0.5 * torch.mean((1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1))
        # L_CE (Eq. 13/14): M_ij (edge_prod) is a fixed, non-learnable target
        # (.detach()ed below) -- its only real gradient path is into
        # edge_logits_vl, i.e. view_net (Phi). Kept entirely within Phase 1's
        # own ascent step (Item 4b) -- a separate view_optimizer.step() after
        # model_loss.backward() in Phase 2 would also pick up calc_loss's own
        # gradient into view_net, inverting the adversarial objective.
        edge_prod_sig = torch.sigmoid(edge_prod.squeeze()).detach()
        edge_logits_sig = torch.sigmoid((edge_logits_vl + gate_inputs_) / temperature)
        ce_loss = F.binary_cross_entropy(edge_logits_sig, edge_prod_sig)
        # reg = mean edge-drop probability -- penalizes high drop rates so
        # the view_learner can't maximize view_loss just by destroying the
        # whole graph (Issue B).
        if replicate_original_code:
            # ce_loss moves to Phase 2/model_loss (see below) -- not part of
            # view_loss in this mode.
            view_loss = model.calc_loss(x, x_aug) - reg_lambda * reg - kld_lambda * kld_loss
        else:
            view_loss = model.calc_loss(x, x_aug) - ce_lambda * ce_loss \
                        - reg_lambda * reg - kld_lambda * kld_loss
        view_loss_all += view_loss.item() * batch.num_graphs
        reg_all += reg.item() * num_graph_with_edges
        num_graphs_seen += batch.num_graphs
        num_graph_with_edges_seen += num_graph_with_edges

        (-view_loss).backward()
        # Unconstrained ascent has no natural upper bound -- observed
        # empirically to diverge (NaN) on some configs. Clipping
        # view_learner's gradient norm only is the standard mitigation.
        torch.nn.utils.clip_grad_norm_(view_learner.parameters(), max_norm=5.0)
        view_optimizer.step()

        # ---- Phase 2: train model to minimize contrastive loss ----
        model.train()
        view_learner.eval()
        model_optimizer.zero_grad()

        x, _ = model(batch.batch, batch.x, batch.edge_index, beta, None, gcn_w,
                    batch.edge_weight, None, gamma=gamma_orig)
        # mu/std/edge_prod discarded here -- ce_loss (Item 4b) moved entirely
        # to Phase 1; only edge_logits_vl (for the gate) is needed here.
        # (replicate_original_code=True needs edge_prod here too, to compute
        # ce_loss on the model side, matching the original's Phase 2.)
        edge_logits_vl, _, _, edge_prod2 = view_learner(batch.batch, batch.x, batch.edge_index, beta, None,
                                             gcn_w, batch.edge_weight, batch.dyn_weight, gamma=gamma_orig)
        edge_logits_vl = _guard_nan(edge_logits_vl, "edge_logits_vl (Phase 2)", epoch_num)
        edge_prod2 = _guard_nan(edge_prod2, "edge_prod2 (Phase 2)", epoch_num)
        eps = (bias - (1 - bias)) * torch.rand(edge_logits_vl.size()) + (1 - bias)
        gate_inputs_ = torch.log(eps) - torch.log(1 - eps)
        gate_inputs_ = gate_inputs_.to(device)
        gate_inputs = (gate_inputs_ + edge_logits_vl) / temperature
        gate = torch.sigmoid(gate_inputs).squeeze()
        batch_aug_edge_weight = gcn_w * gate      # universal: weight x retention (see Phase 1)
        gamma_aug = gate.view(batch.num_graphs, -1).mean(dim=1)  # per-subject mean of the GATE
        x_aug, _ = model(batch.batch, batch.x, batch.edge_index, beta, None, batch_aug_edge_weight,
                        batch.edge_weight, None, gamma=gamma_aug)

        # Fig. 3/4 interpretability: batch_aug_edge_weight is the EFFECTIVE
        # AUGMENTED FC (original weight * gate), NOT the learned retention
        # gate A itself. This matches the authors' released code, which
        # accumulates exactly the same product for its figures, and the
        # paper's Fig. 3/4 ("strengthened and weakened connectivities" -- an
        # FC-like quantity). Summed separately per diagnostic group,
        # normalized by subject count at epoch end.
        per_subject_aug = batch_aug_edge_weight.detach().reshape(
            batch.num_graphs, template, template).cpu()
        y_flat = batch.y.view(-1)
        asd_mask = (y_flat == 1).cpu()
        nc_mask = (y_flat == 0).cpu()
        if asd_mask.any():
            aug_edge_weight_all_asd += per_subject_aug[asd_mask].sum(dim=0)
            n_asd_seen += int(asd_mask.sum())
        if nc_mask.any():
            aug_edge_weight_all_nc += per_subject_aug[nc_mask].sum(dim=0)
            n_nc_seen += int(nc_mask.sum())

        # ce_loss/L_CE moved to Phase 1's view_loss (Item 4b) -- model_loss no
        # longer includes it.
        # contrastive_mode="supervised": only Phase 2 (Theta, the network
        # actually evaluated downstream) uses label information -- Phase 1
        # (view_learner/Phi, the adversary) deliberately stays on the
        # self-supervised calc_loss above regardless of this flag. See
        # supervised/loss.py's module docstring for why: an adversary
        # maximizing a label-aware loss would be directly incentivized to
        # attack class-discriminative structure specifically, not just
        # generic redundant information.
        if contrastive_mode == "supervised":
            labels = batch.y.view(-1)
            model_loss = calc_loss_supervised(x, x_aug, labels, temperature=supervised_temperature)
        else:
            model_loss = model.calc_loss(x, x_aug)
        if replicate_original_code:
            # Original: model_loss = calc_loss(x, x_aug) + ce_lambda * ce_loss
            # -- the model (not the augmenter) is pushed to reduce ce_loss.
            edge_prod2_sig = torch.sigmoid(edge_prod2.squeeze()).detach()
            edge_logits2_sig = torch.sigmoid((edge_logits_vl + gate_inputs_) / temperature)
            ce_loss2 = F.binary_cross_entropy(edge_logits2_sig, edge_prod2_sig)
            model_loss = model_loss + ce_lambda * ce_loss2
        model_loss_all += model_loss.item() * batch.num_graphs

        model_loss.backward()
        model_optimizer.step()

    fin_model_loss = model_loss_all / num_graphs_seen
    fin_view_loss = view_loss_all / num_graphs_seen
    fin_reg = reg_all / num_graph_with_edges_seen
    fin_aug_edge_weight_asd = aug_edge_weight_all_asd / max(n_asd_seen, 1)
    fin_aug_edge_weight_nc = aug_edge_weight_all_nc / max(n_nc_seen, 1)
    # np.random.beta requires both shape parameters strictly > 0; fin_reg can
    # reach exactly 0 or 1 if the gate saturates, which would otherwise crash
    # the run. Clamp only for this call -- fin_reg itself (returned) stays
    # the true, unclamped value.
    fin_reg_beta = float(np.clip(fin_reg, 1e-4, 1 - 1e-4))
    next_beta = np.random.beta(fin_reg_beta, 1 - fin_reg_beta)

    return fin_model_loss, fin_view_loss, fin_reg, fin_aug_edge_weight_asd, fin_aug_edge_weight_nc, next_beta
