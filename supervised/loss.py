"""Supervised Contrastive Learning (Khosla et al., NeurIPS 2020,
arXiv:2004.11362), L^sup_out variant (their Eq. 2) -- confirmed in the paper's
own Table 1 to outperform the L^sup_in alternative (78.7% vs 67.4% top-1 on
ImageNet/ResNet-50), and provably so via Jensen's inequality (log is concave,
so L^sup_in <= L^sup_out always).

Kept in a separate module from unsupervised/learning/ginfominmax.py -- that
file's calc_loss (the self-supervised version) is untouched. This is used
ONLY in Phase 2 (the main model's own training step); Phase 1 (view_learner,
the adversary) stays on the original self-supervised loss deliberately: under
the adversarial min-max game, an adversary maximizing THIS loss would be
directly incentivized to attack class-discriminative structure specifically
(push an ASD subject's embedding toward the TD cluster), inverting the
information bottleneck's purpose of stripping redundant-not-relevant
information. See conversation record for the full reasoning.

Adapted to this codebase's bipartite x-vs-x_aug comparison (not Khosla's full
2N "multiviewed batch" of two independently-augmented copies per sample):
x[i] and x_aug[i] are already two distinct tensors (original vs adversarially
-augmented view), so no self-exclusion is needed the way Khosla's A(i) = I \\ {i}
excludes the literal same index -- here, x[i] vs x_aug[i] is precisely the
original self-supervised positive pair, so it's automatically included in the
new positive set P(i) = {j : label[j] == label[i]} (trivially true for j=i).
"""
import torch


def calc_loss_supervised(x, x_aug, labels, temperature=0.1, sym=True):
    """x, x_aug: [batch_size, dim] graph-level embeddings (same shapes/roles
    as the existing GInfoMinMax.calc_loss). labels: [batch_size] class labels
    (already flattened, e.g. batch.y.view(-1)). temperature: Khosla et al.'s
    own validated value (Sec 4.5: "All our results used a temperature of
    tau=0.1"), distinct from this codebase's self-supervised calc_loss
    default of 0.2.
    """
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()  # [batch_size, batch_size]; pos_mask[i,j]=1 iff label_i == label_j
    # |P(i)| is always >= 1 (the i==j entry is always a positive, since a
    # label always equals itself), so no clamp against divide-by-zero needed.
    pos_count = pos_mask.sum(dim=1)

    if sym:
        # direction 1: x[i] as anchor, compared against every x_aug[j] (row i of sim_matrix)
        denom_1 = sim_matrix.sum(dim=1)
        log_prob_1 = torch.log(sim_matrix) - torch.log(denom_1).unsqueeze(1)
        loss_1 = -(pos_mask * log_prob_1).sum(dim=1) / pos_count

        # direction 2: x_aug[j] as anchor, compared against every x[i] (column j of sim_matrix)
        denom_2 = sim_matrix.sum(dim=0)
        log_prob_2 = torch.log(sim_matrix) - torch.log(denom_2).unsqueeze(0)
        loss_2 = -(pos_mask * log_prob_2).sum(dim=0) / pos_count

        loss = (loss_1.mean() + loss_2.mean()) / 2.0
    else:
        denom_1 = sim_matrix.sum(dim=1)
        log_prob_1 = torch.log(sim_matrix) - torch.log(denom_1).unsqueeze(1)
        loss_1 = -(pos_mask * log_prob_1).sum(dim=1) / pos_count
        loss = loss_1.mean()

    return loss
