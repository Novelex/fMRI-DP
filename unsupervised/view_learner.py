import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from unsupervised.learning import GraSTI


class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, net, mij_source="alff"):
        super(ViewLearner, self).__init__()
# data member
        self.encoder = encoder
        self.input_dim = self.encoder.out_node_dim
        self.net = net
        # "alff": M_ij (Eq. 4) uses only the raw ALFF portion of x (first 3
        # columns) -- v_i, v_j in R^3, matching Eq. 4's literal d=3.
        # "alff_pcc": M_ij uses the FULL x (all 93 columns: ALFF + each
        # node's own PCC row) as the similarity input -- a deliberate
        # deviation from Eq. 4, mirroring the same alff->alff_pcc enrichment
        # already applied to the main node features. Only meaningful when
        # node_feature_mode="alff_pcc" (x is [N,93]); if x is [N,3]
        # (node_feature_mode="alff"), both settings are identical.
        assert mij_source in ("alff", "alff_pcc")
        self.mij_source = mij_source
        self.mlp_edge_model = Sequential(
            Linear(self.input_dim*2, 64),
            ReLU(),
            Linear(64, 1))

            # Member function
        self.init_emb()

    def init_emb(self):
        # Only mlp_edge_model -- self.encoder/self.net are the shared backbone
        # (Issue #16), also owned by GInfoMinMax. Walking self.modules() here
        # would re-initialize them a second time (see identical note in
        # GInfoMinMax.init_emb).
        for m in self.mlp_edge_model.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, beta, edge_attr, gcn_edge_weight, pcc_weight, dyn_weight, gamma=None):
        # See GInfoMinMax.forward for why gcn_edge_weight and pcc_weight are
        # separate: the GCN must never receive the raw signed PCC as its weight.
        _, node_emb = self.encoder(batch, x, edge_index, beta, gcn_edge_weight, gamma)
        src, dst = edge_index[0], edge_index[1]

        if dyn_weight is None:
            emb_src = node_emb[src]
            emb_dst = node_emb[dst]
            edge_emb = torch.cat([emb_src, emb_dst], 1)
            edge_logits = self.mlp_edge_model(edge_emb)
            return edge_logits
        else:
            # M_ij (Eq. 4): sigma(v_i . v_j) -- not the encoder's learned
            # node_emb, but a fixed similarity computed directly from raw
            # per-node data. self.mij_source picks which slice of x:
            # "alff": x[:, :3] -- v_i, v_j in R^3, literal Eq. 4 (Sec 4.1
            # fixes d=3). With node_feature_mode="alff" x is already [N,3];
            # with "alff_pcc" x is [N,93] (ALFF first, PCC row after), so
            # this slice recovers just the ALFF portion either way.
            # "alff_pcc": the full x (all 93 columns) -- deliberately not
            # Eq. 4-literal; see ViewLearner.__init__ docstring.
            x_mij = x[:, :3] if self.mij_source == "alff" else x
            x_src = x_mij[src]
            x_dst = x_mij[dst]
            edge_prod = torch.sum(x_src * x_dst, dim=1)
            mu, std, edge_logits = self.net.get_mu_std_logits(pcc_weight, dyn_weight)
            return edge_logits, mu, std, edge_prod
