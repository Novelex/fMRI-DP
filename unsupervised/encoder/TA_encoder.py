import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from unsupervised.convs.wgin_conv import WGINConv

from unsupervised.convs import GATConv
from unsupervised.convs import GCNConv


def full_attention_conv(qs, ks, vs, batch=None, output_attn=False):
    """
    qs: query tensor [N, H, D]
    ks: key tensor [L, H, D]
    vs: value tensor [L, H, D]
    batch: [N] PyG graph-id per node. When multiple subjects' nodes are
        concatenated along dim 0 (PyG's batching for batch_size > 1), the
        softmax below must be restricted to within each subject's own
        nodes -- otherwise one subject's output depends on which other
        subjects happen to share its training batch.

    Eq. (16) of GraSTI-ACL ("we draw inspiration from Graphormer (Ying et al.,
    2021) and apply the Transformer principle to compute attention scores for
    all nodes in the graph") cites Graphormer's own Eq. (4):
        A = QK^T / sqrt(d),  Attn(H) = softmax(A) V.
    Graphormer's other structural additions (centrality/spatial/edge encoding
    bias terms) are deliberately not ported here: this graph is already fully
    connected (Appendix A.1), so shortest-path-distance-based spatial encoding
    would be degenerate (every distance = 1 hop), and the paper states this
    branch is meant to move away from topology toward "global information" --
    topology is the parallel GCN branch's (X_Topo) job, not this one's.

    return output [N, H, D]
    """
    D = qs.shape[-1]
    scale = D ** 0.5

    if batch is None:
        # single graph
        scores = torch.einsum("nhd,lhd->nlh", qs, ks) / scale  # [N, L, H]
        attn = torch.softmax(scores, dim=1)  # softmax over keys (L), per query node, per head
        attn_output = torch.einsum("nlh,lhd->nhd", attn, vs)  # [N, H, D]

        if output_attn:
            return attn_output, attn.mean(dim=-1)  # [N, N], averaged over heads
        return attn_output

    # Batched (multiple subjects): reshape [num_graphs*n, ...] -> [num_graphs, n, ...]
    # so softmax pools only over one subject's own n nodes (block-diagonal
    # attention), not the whole training batch. Assumes every graph in the batch has
    # the same node count n (true here: every subject has 90 ROIs from the fixed
    # AAL atlas), and that PyG concatenates nodes contiguously graph-by-graph, which
    # is exactly how Batch.from_data_list builds `batch`.
    if output_attn:
        raise NotImplementedError("output_attn is not supported for batched multi-subject input")

    num_graphs = int(batch.max().item()) + 1
    n = qs.shape[0] // num_graphs
    H = qs.shape[1]

    qs = qs.view(num_graphs, n, H, D)
    ks = ks.view(num_graphs, n, H, D)
    vs = vs.view(num_graphs, n, H, D)

    scores = torch.einsum("bnhd,blhd->bnlh", qs, ks) / scale  # [B, n, n, H]
    attn = torch.softmax(scores, dim=2)  # softmax over keys (l), per query node, per head, per subject
    attn_output = torch.einsum("bnlh,blhd->bnhd", attn, vs)  # [B, n, H, D]

    return attn_output.reshape(num_graphs * n, H, D)


class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, batch=None, edge_index=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, batch, output_attn
            )  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value, batch)  # [N, H, D]

        final_output = attention_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers=2,
            num_heads=1,
            alpha=0.5,
            dropout=0.5,
            use_bn=True,
            use_residual=True,
            use_weight=True,
            use_act=True,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(
                    hidden_channels,
                    hidden_channels,
                    num_heads=num_heads,
                    use_weight=use_weight,
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        # self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.alpha = alpha
        self.use_act = use_act
        self.numlayers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, batch=None, edge_index=None):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, batch, edge_index)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class TAEncoder(torch.nn.Module):
    def __init__(self, num_dataset_features, beta, emb_dim=300, num_gc_layers=5, num_trans_layers=1, num_trans_heads=1,
                 drop_ratio=0.0, pooling_type="standard",
                 alpha=0.5,
                 trans_use_bn=True,
                 trans_use_residual=True,
                 trans_use_weight=True,
                 trans_use_act=True,
                 use_graph=True,
                 trans_dropout=0.5,
                 is_infograph=False,
                 beta_convention="reversed",
                 gamma_orig_mode="one",
                 enable_attention_mix=True):
        super(TAEncoder, self).__init__()
        # The original authors' released code computes x_trans (attention branch)
        # and beta (Eq. 18's Beta-Mixup weight), but the line that would actually
        # mix them into x is commented out (unsupervised/encoder/TA_encoder.py:272-274
        # in github.com/BiaoHe2025/GraSTIACL) -- x_trans is discarded and only the
        # GCN branch (x) reaches pooling. enable_attention_mix=False replicates
        # that literal behavior for a direct comparison; default True keeps this
        # project's paper-prose-faithful implementation (Eq. 19) unchanged.
        self.enable_attention_mix = enable_attention_mix
        self.pooling_type = pooling_type
        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.is_infograph = is_infograph
        # "reversed": lambda_ ~ Beta(1-gamma, gamma), E[lambda_]=1-gamma -- matches
        # Sec 3.3's prose ("less connectivity -> more attention"), but forces
        # lambda_->0 when gamma=1 (original view, nothing dropped).
        # "literal": lambda_ ~ Beta(gamma, 1-gamma), matching Eq. 18's symbol order
        # literally -- gives lambda_->1 (full attention) when gamma=1 instead,
        # contradicting the prose but keeping attention alive for the original view.
        assert beta_convention in ("reversed", "literal")
        self.beta_convention = beta_convention
        # "one": gamma_orig=1.0 (current default -- retention-ratio reading).
        # "signal_strength": gamma_orig falls back to mean(|W|), the real
        # weighted-adjacency mean, instead of the hardcoded 1.0 -- Option B's
        # fix, keeping lambda_ non-trivial for the original/eval view even
        # under the "reversed" Beta convention.
        assert gamma_orig_mode in ("one", "signal_strength")
        self.gamma_orig_mode = gamma_orig_mode
        self.trans_conv = TransConv(num_dataset_features, emb_dim, num_trans_layers, num_trans_heads, alpha,
                                    trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)
        self.out_node_dim = self.emb_dim
        self.beta = beta
        if self.pooling_type == "standard":
            self.out_graph_dim = self.emb_dim
        elif self.pooling_type == "layerwise":
            self.out_graph_dim = self.emb_dim * self.num_gc_layers
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        emb_dim_list = [[32, 64, 128, 64, 32], [32, 64, 32], [32, 32]]
        emb_dims = emb_dim_list[1]
        #
        for i in range(num_gc_layers):

            if i:
                # nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))

                # nn = Linear(emb_dims[i-1], emb_dims[i])
                # add_self_loops=False: edge_index is already fully connected with
                # explicit self-loops (Dataset.py), so GCNConv shouldn't add more.
                conv = GCNConv(emb_dim, emb_dim, add_self_loops=False)
            else:
                # nn = Sequential(Linear(num_dataset_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))

                # nn = Linear(num_dataset_features, emb_dims[i])
                conv = GCNConv(num_dataset_features, emb_dim, add_self_loops=False)
            # conv = WGINConv(nn)
            # 通过全连接层构造卷积层
            # bn = torch.nn.BatchNorm1d(emb_dims[i])
            bn = torch.nn.BatchNorm1d(emb_dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, batch, x, edge_index, beta, edge_weight=None, gamma=None):
        xs = []
        # batch threaded through so attention (full_attention_conv) never
        # mixes different subjects' nodes together (Issue #15).
        x_trans = self.trans_conv(x, batch)
        for i in range(self.num_gc_layers):
            ##
            # if torch.isnan(edge_weight).any():
            #     raise ValueError("edge_weight contains NaN values")
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)

            if i == self.num_gc_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)
        # x_trans gets reduced to pure direction (its own raw magnitude is an
        # arbitrary artifact of the Q/K/V weights, not a meaningful quantity).
        # x is deliberately NOT normalized -- unlike x_trans, its magnitude IS
        # meaningful (it traces back to ALFF, a literal amplitude measure), and
        # forcing every node to unit length was erasing real, verified variation
        # (up to 37x between real ROIs) before it ever reached the classifier.
        if not self.enable_attention_mix:
            # Matches the original released code exactly: x_trans is computed
            # above (so its parameters still get gradients from wherever else
            # they're used) but never reaches x -- pooling runs on the GCN
            # branch alone. Skips the whole gamma/Beta machinery below, so
            # this path cannot hit the NaN/Beta crash at all.
            if self.pooling_type == "standard":
                xpool = global_add_pool(x, batch)
                return xpool, x
            elif self.pooling_type == "layerwise":
                xpool = [global_add_pool(xi, batch) for xi in xs]
                xpool = torch.cat(xpool, 1)
                return xpool, (torch.cat(xs, 1) if self.is_infograph else x)
            else:
                raise NotImplementedError

        x_trans = F.normalize(x_trans, dim=1)
        # Eq. (18): gamma = the true retained-edge ratio, not a proxy inferred from
        # edge_weight's magnitude. edge_weight now carries real connectivity strength
        # (gcn_w, or gcn_w * gate for the augmented view), so its mean is a
        # correlation magnitude, not a retention ratio -- the two are different
        # numbers. Callers must pass the real ratio explicitly: 1.0 for any
        # unaugmented pass (nothing was dropped, so nothing to compensate for),
        # or the real gate's own mean for the augmented view specifically.
        eps = 1e-4
        num_graphs = int(batch.max().item()) + 1
        if gamma is None:
            # Fallback only, for any call site not yet passing gamma explicitly.
            # PER-SUBJECT, not batch-wide: measured directly (batch-gamma test,
            # this session) that a batch-wide mean makes one subject's embedding
            # depend on whichever strangers share its batch -- a shift as large
            # as 85.5% of a typical between-subject distance, confirmed
            # gamma-mediated via a pinned-gamma control arm (0.0 difference).
            # Every subject owns exactly 8100 contiguous edges (the fixed 90x90
            # fully-connected structure, certified in Stage 1), so a plain
            # reshape recovers each subject's own edges safely.
            if edge_weight is not None:
                edges_per_subject = edge_weight.shape[0] // num_graphs
                gamma = edge_weight.view(num_graphs, edges_per_subject).mean(dim=1).clamp(eps, 1 - eps)
            else:
                gamma = torch.full((num_graphs,), 1 - eps, device=x.device)
        gamma = torch.as_tensor(gamma, device=x.device)
        if gamma.dim() == 0:
            # A caller passed a single scalar explicitly (e.g. a fixed override
            # for evaluation or a control test) -- broadcast it identically to
            # every subject. This is deliberately NOT batch-composition-dependent:
            # a true constant applied the same way to everyone is not the bug.
            gamma = gamma.expand(num_graphs).clone()
        if torch.isnan(gamma).any():
            # clamp() does not sanitize NaN (NaN comparisons are always False,
            # so it passes straight through to Beta() and crashes). gamma going
            # NaN mid-training is a real, observed adversarial-dynamics failure
            # mode (Stage B, 4/12 jobs, epoch 83-171); 0.5 is a neutral fallback
            # (evenly split retained/dropped) rather than a guess at the true value.
            gamma = torch.nan_to_num(gamma, nan=0.5)
        gamma = gamma.clamp(eps, 1 - eps)
        # Beta(1-gamma, gamma), giving E[lambda] = 1-gamma, not Beta(gamma, 1-gamma):
        # Sec. 3.3's prose is explicit that LESS connectivity (smaller gamma) should
        # mean MORE attention weight ("as the connectivity of the graph diminishes
        # during training, this encoder can appropriately reduce the influence of
        # topological information and focus more on global information") -- the
        # opposite of what Beta(gamma, 1-gamma) produces. The original released
        # code's own fin_reg/beta computation used the drop probability (1-gamma)
        # as the first Beta argument, matching this prose reading, not Eq. 18's
        # literal symbol order taken in isolation.
        #
        # Sampled only during training -- the same stochastic-regularizer-vs-eval
        # split every other source of randomness in this file already respects
        # (F.dropout above, gated on self.training). Without this, lambda_ was
        # being redrawn fresh on every call including at embedding-extraction
        # time, so the exact same trained model produced different embeddings
        # for the exact same subject from one call to the next.
        if self.beta_convention == "literal":
            # Eq. 18 literal symbol order: lambda ~ Beta(gamma, 1-gamma).
            if self.training:
                lambda_ = torch.distributions.Beta(gamma, 1 - gamma).sample()
            else:
                lambda_ = gamma
        else:
            if self.training:
                lambda_ = torch.distributions.Beta(1 - gamma, gamma).sample()
            else:
                lambda_ = 1 - gamma
        # Eq. (19): X_Update = X_Topo + lambda * X_Atte (additive, not convex blend).
        # x_trans is scaled by lambda_ times x's OWN magnitude (not a fixed 1.0),
        # so lambda_ alone controls attention's relative contribution regardless
        # of x_trans's or x's absolute scale -- otherwise the two branches'
        # unrelated natural scales could dominate the mixture instead of lambda_.
        # eps inside the sqrt (not clamped afterward) keeps the gradient of this
        # norm well-behaved even if x approaches the zero vector for some node --
        # clamping the output alone wouldn't fix the backward pass through sqrt.
        x_norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + 1e-12)
        # lambda_ is now per-subject [num_graphs] (one Beta sample per subject,
        # not one shared sample for the whole batch -- see the gamma fix above).
        # Expand to per-node [B*90, 1] via the batch-assignment vector so each
        # subject's own lambda_ only ever multiplies that subject's own rows --
        # NOT averaging lambda_ back to a scalar here, which would silently
        # recreate the same batch-dependence bug under a different name.
        lambda_per_node = lambda_[batch].unsqueeze(1)
        x = x + lambda_per_node * x_norm * x_trans
        # compute graph embedding using pooling
        if self.pooling_type == "standard":
            xpool = global_add_pool(x, batch)
            # print("xpool.shape:", xpool.shape)
            return xpool, x

        elif self.pooling_type == "layerwise":
            xpool = [global_add_pool(x, batch) for x in xs]
            xpool = torch.cat(xpool, 1)
            if self.is_infograph:
                return xpool, torch.cat(xs, 1)
            else:
                return xpool, x
        else:
            raise NotImplementedError

    def get_embeddings(self, loader, beta, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index
                # Eq. (17): X_Topo = GCN(X, A). Fig. 2 builds the original graph G
                # directly from real FC, and Sec. 3.2's "instead of discarding
                # certain edges, we weaken their connections" only makes sense if
                # there's a real connection to weaken -- A is the (non-negative)
                # connectivity itself, not a permanently unweighted placeholder.
                edge_weight = data.edge_weight.abs().clamp(1e-6, 1.0)

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                # No augmentation happens at evaluation time. gamma_orig_mode
                # decides what "no augmentation" means for lambda_: "one" ->
                # gamma=1.0 exactly (retention-ratio reading); "signal_strength"
                # -> gamma=None, which forward()'s own fallback resolves to
                # mean(|W|) (Option B) -- must match whatever mode training used.
                gamma_eval = None if self.gamma_orig_mode == "signal_strength" else 1.0
                x, _ = self.forward(batch, x, edge_index, beta, edge_weight, gamma=gamma_eval)

                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
