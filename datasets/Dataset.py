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
        # "alff": x = the old norm_matrix, [90,3] -- NOT raw ALFF: it is
        # DPARSF voxelwise mALFF, averaged into AAL ROIs, then z-scored per
        # subject per band (Notebook 3). Name kept for backward compatibility
        # with every recorded result; the comment previously calling it "raw"
        # was wrong and is corrected here.
        # "alff_pcc": x = [min-max ALFF -> [0,1] ; min-max own PCC row -> [-1,1]],
        # [90,93] -- each node's connectivity profile becomes part of its own
        # input feature (BrainGNN's own established practice, one of this
        # paper's cited baselines), on top of its ALFF. Min-max stats are
        # fit once, population-wide across all subjects, not per-subject --
        # per-subject min-max would erase genuine between-subject amplitude
        # differences that Issue #7 established are real, meaningful signal.
        # "alff_raw": x = raw (non-z-scored, non-mALFF-normalized) ALFF from
        # alff_new/non_combat/alff_new.npz, matched by subject ID. Preserves
        # between-subject global amplitude differences that the z-scored
        # norm_matrix erases by construction (measured: sum-pooled probe 54.9%
        # vs 50.8% for z-scored). Everything else (PCC, dyn windows, labels)
        # identical to 'alff' mode.
        # "alff_paper": A-GCL Sec 2.1's exact node features -- raw ROI-first
        # ALFF (alff_new.npz 'alff' key, same source alff_raw loads) mapped to
        # [0,1] per subject with ONE shared min/max over the whole (90,3):
        # x = (x - x.min()) / (x.max() - x.min()). The shared scalar (never
        # per-band axis=0) preserves the amplitude relationship BETWEEN bands,
        # which per-band normalisation would destroy. Per subject only; no
        # cohort statistic is ever computed.
        # "alff_new_z" (Stage 6E, from the accepted Stage-6D decision audit):
        # raw ROI-first ALFF from alff_new.npz (same source as alff_raw/
        # alff_paper), PER-SUBJECT PER-BAND z-scored across the 90 ROIs:
        # x[:,b] = (x[:,b] - mean) / std. 956 subjects. Fail-loud: std <= eps
        # raises; no nan_to_num anywhere on this path.
        # "alff_m1_z": same z recipe on the controlled ROI-first recomputation
        # ALFF_func_proc/method1/alff_roi_first.npz. EXACT 954-subject cohort:
        # CMU_b_0050669 and Leuven_1_0050706 (zero-ROI func_preproc) are
        # DROPPED from the dataset in this mode -- never imputed, never
        # substituted from another source. len(dataset) == 954 here.
        # alff_paper stays FROZEN as the shared-[0,1] normalization control.
        assert node_feature_mode in ('alff', 'alff_pcc', 'alff_raw', 'alff_paper',
                                     'alff_new_z', 'alff_m1_z')
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
        return {'alff': 'data.pt',
                'alff_pcc': 'data_alff_pcc93.pt',
                'alff_raw': 'data_alff_raw.pt',
                'alff_paper': 'data_alff_paper.pt',
                'alff_new_z': 'data_alff_new_z.pt',
                'alff_m1_z': 'data_alff_m1_z.pt'}[self.node_feature_mode]

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

            raw.append((subject_id, x_alff, pcc_matrix, dyn_weight, y))

        return raw

    def _build_data_list(self, raw, alff_min=None, alff_max=None, pcc_min=None, pcc_max=None):
        data_list = []
        for subject_id, x_alff, pcc_matrix, dyn_weight, y in raw:
            if self.node_feature_mode in ('alff_raw', 'alff_paper', 'alff_new_z', 'alff_m1_z'):
                # Substitute raw ALFF by subject ID (never by position).
                x_alff = self._raw_alff_map[subject_id]
            if self.node_feature_mode in ('alff_new_z', 'alff_m1_z'):
                # Stage 6E: per-subject PER-BAND z across the 90 ROIs.
                mu = x_alff.mean(axis=0, keepdims=True)
                sd = x_alff.std(axis=0, keepdims=True)
                # Fail loudly on a degenerate band -- never silently repair.
                assert (sd > 1e-8).all(), f"{subject_id}: degenerate band std {sd}"
                x_alff = (x_alff - mu) / sd
                assert x_alff.shape == (90, 3), f"{subject_id}: shape {x_alff.shape}"
                assert np.isfinite(x_alff).all(), f"{subject_id}: non-finite after z"
                assert np.allclose(x_alff.mean(axis=0), 0.0, atol=1e-10), subject_id
                assert np.allclose(x_alff.std(axis=0), 1.0, atol=1e-6), subject_id
            if self.node_feature_mode == 'alff_paper':
                # A-GCL Sec 2.1: ONE scalar min and ONE scalar max over the
                # whole (90,3) -- shared across all three bands, per subject.
                lo = x_alff.min()
                hi = x_alff.max()
                assert hi > lo, f"{subject_id}: degenerate ALFF (max==min)"
                # exact division (no epsilon): the epsilon variant would make
                # the maximum land at 1 - ~2e-14 instead of exactly 1.0,
                # breaking the exact-endpoint asserts below; hi > lo is
                # guaranteed by the assert above, so the division is safe.
                x_alff = (x_alff - lo) / (hi - lo)
                assert x_alff.shape == (90, 3), f"{subject_id}: shape {x_alff.shape}"
                assert np.isfinite(x_alff).all(), f"{subject_id}: non-finite after [0,1] map"
                assert np.isclose(x_alff.min(), 0.0), f"{subject_id}: min {x_alff.min()} not ~0"
                assert np.isclose(x_alff.max(), 1.0), f"{subject_id}: max {x_alff.max()} not ~1"
            num_nodes = pcc_matrix.shape[0]

            if self.node_feature_mode == 'alff_pcc':
                # Signed [-1,1] rescale, matching pcc_scaled's own treatment one line
                # below -- not [0,1]. Raw ALFF (node_feature_mode='alff') is already
                # signed (z-scored per subject per band in Notebook 3); squashing it
                # to [0,1] here made every ROI-pair dot product v_i.v_j >= 0, so
                # mij_source='alff''s M_ij = sigmoid(v_i.v_j) (Eq. 4) could only ever
                # read >= 0.5 -- structurally unable to express "dissimilar" for any
                # pair, in every run using this feature mode.
                alff_scaled = 2 * (x_alff - alff_min) / (alff_max - alff_min + 1e-12) - 1
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

        if self.node_feature_mode in ('alff_raw', 'alff_paper', 'alff_new_z', 'alff_m1_z'):
            import numpy as _np
            repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
            if self.node_feature_mode == 'alff_m1_z':
                d = _np.load(osp.join(repo_root, 'ALFF_func_proc', 'method1',
                                      'alff_roi_first.npz'), allow_pickle=True)
                expected_n = 954
            else:
                d = _np.load(osp.join(repo_root, 'alff_new', 'non_combat',
                                      'alff_new.npz'), allow_pickle=True)
                expected_n = 956
            # File-level integrity: exactly the expected cohort, no duplicate
            # IDs; the alff_new file additionally carries per-subject ok flags.
            assert len(d['file_ids']) == expected_n
            assert len(set(d['file_ids'].tolist())) == len(d['file_ids'])
            if 'ok' in d:
                assert d['ok'].all()
            # No nan_to_num: silent repair would hide a corrupted source. The
            # data is asserted finite instead -- fail loudly, never patch.
            self._raw_alff_map = {}
            for i, sid in enumerate(d['file_ids']):
                arr = d['alff'][i].astype(_np.float64)
                assert arr.shape == (90, 3), f"{sid}: raw ALFF shape {arr.shape}"
                assert _np.isfinite(arr).all(), f"{sid}: non-finite raw ALFF"
                self._raw_alff_map[sid] = arr
            npz_ids = set(d['file_ids'].tolist())
            dataset_ids = {sid for sid, *_ in raw_all}
            if self.node_feature_mode == 'alff_m1_z':
                # EXACT 954 cohort: the controlled func_preproc recomputation
                # cannot produce valid AAL90 features for these two subjects
                # (one all-zero ROI each). They are DROPPED from the dataset in
                # this mode -- never imputed, never mixed from another source.
                expected_missing = {'CMU_b_0050669', 'Leuven_1_0050706'}
                assert npz_ids <= dataset_ids, (
                    f"alff_m1_z: npz-only subjects {sorted(npz_ids - dataset_ids)[:5]}")
                assert dataset_ids - npz_ids == expected_missing, (
                    f"alff_m1_z: unexpected missing set "
                    f"{sorted(dataset_ids - npz_ids)}")
                raw_all = [r for r in raw_all if r[0] in npz_ids]
                assert len(raw_all) == 954, f"alff_m1_z: {len(raw_all)} subjects after drop"
            else:
                # Exact subject-set equality (order-free by design: retrieval
                # goes through self._raw_alff_map[subject_id], never by position).
                assert npz_ids == dataset_ids, (
                    f"{self.node_feature_mode}: subject-set mismatch -- "
                    f"npz-only={sorted(npz_ids - dataset_ids)[:5]}, "
                    f"dataset-only={sorted(dataset_ids - npz_ids)[:5]}")

        if self.node_feature_mode == 'alff_pcc':
            all_alff = np.stack([r[1] for r in raw_all])   # [N, 90, 3]
            all_pcc = np.stack([r[2] for r in raw_all])     # [N, 90, 90]
            alff_min, alff_max = float(all_alff.min()), float(all_alff.max())
            pcc_min, pcc_max = float(all_pcc.min()), float(all_pcc.max())
            data_list = self._build_data_list(raw_all, alff_min, alff_max, pcc_min, pcc_max)
        else:
            data_list = self._build_data_list(raw_all)

        torch.save(self.collate(data_list),
                   osp.join(self.processed_dir, self.processed_file_names))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
