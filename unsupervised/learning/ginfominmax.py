import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F


class GInfoMinMax(torch.nn.Module):
    def __init__(self, encoder, net, proj_hidden_dim=300):
        super(GInfoMinMax, self).__init__()

        self.encoder = encoder
        self.input_proj_dim = self.encoder.out_graph_dim
        self.net = net
        self.proj_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
                                    Linear(proj_hidden_dim, proj_hidden_dim))

        self.init_emb()

    def init_emb(self):
        # Only proj_head -- self.encoder/self.net are the shared backbone
        # (Issue #16), also owned by ViewLearner. Walking self.modules() here
        # would re-initialize them a second time, discarding whichever call
        # (GInfoMinMax's or ViewLearner's) happened to run first.
        for m in self.proj_head.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, beta, edge_attr, gcn_edge_weight, pcc_weight, dyn_weight, gamma=None):
        # gcn_edge_weight: None (unweighted A, Eq. 17) for the original view, or the
        # sigmoid-gated augmentation weight for the augmented view -- never the raw
        # signed PCC, which would break GCNConv's weighted-degree normalization.
        # pcc_weight: always the raw signed PCC, used only as ToyNet's VIB input.
        # gamma: the true retained-edge ratio for this specific call (1.0 for the
        # unaugmented view, the real gate's own mean for the augmented view) --
        # see TAEncoder.forward's gamma docstring for why this can't be inferred
        # from gcn_edge_weight anymore.
        z, node_emb = self.encoder(batch, x, edge_index, beta, gcn_edge_weight, gamma)

        z = self.proj_head(z)

        if dyn_weight is None:
            return z,node_emb
        else:
            _, _, logits = self.net.get_mu_std_logits(pcc_weight, dyn_weight)
            # z shape -> Batch x proj_hidden_dim = 32*32
            return z, node_emb, logits


    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        # x and x_aug shape -> Batch x proj_hidden_dim
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        m1 = torch.einsum('ik,jk->ij', x, x_aug)
        m2 = torch.einsum('i,j->ij', x_abs, x_aug_abs)
        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1

        return loss
