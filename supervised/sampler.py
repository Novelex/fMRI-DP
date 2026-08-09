"""Class-balanced batch sampler for Supervised Contrastive Learning.

Plain DataLoader(shuffle=True) gives no guarantee about class composition per
batch -- a batch could end up almost all one class, starving SupCon's
positive sets (same-class peers) for whichever subjects land in an
imbalanced batch. This sampler guarantees every batch is as close to an even
class split as batch_size allows, cycling through each class's own shuffled
pool independently (reshuffling a class's pool when it runs out, rather than
stopping early), so batch composition never depends on how the classes
happened to interleave in the underlying dataset order.
"""
import numpy as np
from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, seed=123):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.seed = seed
        # Incremented every __iter__ call (DataLoader calls this once per
        # epoch) so each epoch gets a genuinely different shuffle, not the
        # same fixed sequence repeated 300 times -- while still being
        # reproducible run-to-run for a given seed.
        self._epoch = 0
        self.classes = np.unique(self.labels)
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        # Drop the remainder, matching drop_last=False's absence being a
        # non-issue here since batches are built directly at fixed size.
        self.num_batches = len(self.labels) // batch_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1
        pools = {c: rng.permutation(idx).tolist() for c, idx in self.class_indices.items()}
        n_classes = len(self.classes)
        per_class = self.batch_size // n_classes
        remainder = self.batch_size - per_class * n_classes
        for _ in range(self.num_batches):
            batch = []
            for i, c in enumerate(self.classes):
                take = per_class + (1 if i < remainder else 0)
                if len(pools[c]) < take:
                    pools[c] = rng.permutation(self.class_indices[c]).tolist()
                batch.extend(pools[c][:take])
                pools[c] = pools[c][take:]
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
