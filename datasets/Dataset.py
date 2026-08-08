import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from os import listdir
import numpy as np
import os.path as osp
import scipy.io as sio
import numpy as np


class WindowedData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        # dyn_weight is [T, 90, 90] per subject. PyG's default __cat_dim__ (0)
        # concatenates it across a batch into [B*T, 90, 90] -- indistinguishable
        # in shape from an actually-stacked [B, T, 90, 90] only by coincidence
        # (every subject having the same T, and PyG concatenating graphs in
        # order). Returning None here makes PyG stack along a new leading
        # dimension instead, giving a genuine [B, T, 90, 90] directly, not an
        # accidentally-recoverable-by-reshape one.
        if key == 'dyn_weight':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class ADNIDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, node_feature_mode='alff'):
        self.root = root
        self.name = name
        self.num_tasks = 1
        self.task_type = 'classification'
        self.eval_metric = 'accuracy'
        # "alff": x = raw ALFF, [90,3] (current/default behavior, unchanged).
        # "alff_pcc": x = [min-max ALFF -> [0,1] ; min-max own PCC row -> [-1,1]],
        # [90,93] -- each node's connectivity profile becomes part of its own
        # input feature (BrainGNN's own established practice, one of this
        # paper's cited baselines), on top of its ALFF. Min-max stats are
        # fit once, population-wide across all subjects, not per-subject --
        # per-subject min-max would erase genuine between-subject amplitude
        # differences that Issue #7 established are real, meaningful signal.
        assert node_feature_mode in ('alff', 'alff_pcc')
        self.node_feature_mode = node_feature_mode

        super(ADNIDataset, self).__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, self.processed_file_names)
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        # Different filename per mode -- both can be cached side by side in
        # the same processed/ directory without colliding or forcing a
        # rebuild when switching modes.
        return 'data.pt' if self.node_feature_mode == 'alff' else 'data_alff_pcc93.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def _load_raw_group(self, adj_dir, nf_dir, dw_dir, y):
        """Loads raw arrays only -- no feature construction yet. Kept separate
        from feature construction so "alff_pcc" mode can compute population-
        wide min/max across ALL subjects (both groups) before building any
        subject's actual feature vector."""
        # match subjects across ADJ/NF/DW by filename (subject ID), not os.listdir()
        # position -- os.listdir() order is filesystem-arbitrary and not guaranteed
        # consistent across separate directories, so position-based pairing can
        # silently combine one subject's adjacency with a different subject's
        # node features and dynamic weights.
        adj_files = {f.rsplit('_adj.mat', 1)[0]: f for f in os.listdir(adj_dir)}
        nf_files = {f.rsplit('_nf.mat', 1)[0]: f for f in os.listdir(nf_dir)}
        dw_files = {f.rsplit('_dw.mat', 1)[0]: f for f in os.listdir(dw_dir)}

        subject_ids = set(adj_files) & set(nf_files) & set(dw_files)
        assert subject_ids == set(adj_files) == set(nf_files) == set(dw_files), (
            f"Subject sets differ across {adj_dir}, {nf_dir}, {dw_dir}"
        )

        raw = []
        for subject_id in sorted(subject_ids):
            nf = sio.loadmat(osp.join(nf_dir, nf_files[subject_id]))
            x_alff = np.nan_to_num(nf['norm_matrix'])

            adj = sio.loadmat(osp.join(adj_dir, adj_files[subject_id]))
            pcc_matrix = np.nan_to_num(adj['cropped_matrix'])

            dw = sio.loadmat(osp.join(dw_dir, dw_files[subject_id]))
            dw_array = dw['correlation_matrices']
            dyn_weight_np = [dw_array[j, 0] for j in range(dw_array.shape[0])]
            dyn_weight = torch.stack([torch.tensor(m) for m in dyn_weight_np]).float()

            raw.append((x_alff, pcc_matrix, dyn_weight, y))

        return raw

    def _build_data_list(self, raw, alff_min=None, alff_max=None, pcc_min=None, pcc_max=None):
        data_list = []
        for x_alff, pcc_matrix, dyn_weight, y in raw:
            num_nodes = pcc_matrix.shape[0]

            if self.node_feature_mode == 'alff_pcc':
                alff_scaled = (x_alff - alff_min) / (alff_max - alff_min + 1e-12)
                pcc_scaled = 2 * (pcc_matrix - pcc_min) / (pcc_max - pcc_min + 1e-12) - 1
                x = np.concatenate([alff_scaled, pcc_scaled], axis=1)  # [90, 3+90=93]
            else:
                x = x_alff  # [90, 3]
            x = torch.Tensor(x)

            # Appendix A.1: "A is the adjacency matrix to be learned, initialized
            # as a fully connected graph" -- build all N x N pairs (self-loops
            # included) rather than deriving edges from nonzero PCC entries,
            # which would silently drop any exact-zero correlation as "no edge".
            # edge_weight is always the RAW pcc_matrix regardless of
            # node_feature_mode -- unaffected by whether x also carries a
            # (separately scaled) copy of connectivity as node content.
            row, col = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes), indexing='ij')
            edge_index = torch.tensor(np.stack([row.flatten(), col.flatten()]), dtype=torch.long)
            edge_weight = torch.tensor(pcc_matrix.flatten(), dtype=torch.float)

            data = WindowedData(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
            data.dyn_weight = dyn_weight
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        raw_asd = self._load_raw_group(
            osp.join(self.raw_dir, 'ASD_ADJ'),
            osp.join(self.raw_dir, 'ASD_NF'),
            osp.join(self.raw_dir, 'ASD_DW'),
            y=1,
        )
        raw_nc = self._load_raw_group(
            osp.join(self.raw_dir, 'NC_ADJ'),
            osp.join(self.raw_dir, 'NC_NF'),
            osp.join(self.raw_dir, 'NC_DW'),
            y=0,
        )
        raw_all = raw_asd + raw_nc

        if self.node_feature_mode == 'alff_pcc':
            all_alff = np.stack([r[0] for r in raw_all])   # [N, 90, 3]
            all_pcc = np.stack([r[1] for r in raw_all])     # [N, 90, 90]
            alff_min, alff_max = float(all_alff.min()), float(all_alff.max())
            pcc_min, pcc_max = float(all_pcc.min()), float(all_pcc.max())
            data_list = self._build_data_list(raw_all, alff_min, alff_max, pcc_min, pcc_max)
        else:
            data_list = self._build_data_list(raw_all)

        torch.save(self.collate(data_list),
                   osp.join(self.processed_dir, self.processed_file_names))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
