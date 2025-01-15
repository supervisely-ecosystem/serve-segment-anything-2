from torch.utils.data.sampler import Sampler
import supervisely as sly


class BatchSampler(Sampler):
    """Ensures that only images with the same number of labels are located within one batch"""

    def __init__(self, sly_items, sly_project_meta, batch_size):
        self.sly_items = sly_items
        self.sly_project_meta = sly_project_meta
        self.batch_size = batch_size

    def group_items_by_labels_count(self):
        n_labels_to_idx = {}
        for idx, sly_item in enumerate(self.sly_items):
            ann_path = sly_item.ann_path
            ann = sly.Annotation.load_json_file(ann_path, self.sly_project_meta)
            n_labels = len(ann.labels)
            if n_labels in n_labels_to_idx:
                n_labels_to_idx[n_labels].append(idx)
            else:
                n_labels_to_idx[n_labels] = [idx]
        self.indices = []
        for n_labels, idx in n_labels_to_idx.items():
            batched_indices = [
                idx[i : i + self.batch_size]
                for i in range(0, len(idx), self.batch_size)
            ]
            self.indices.extend(batched_indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        self.group_items_by_labels_count()
        return len(self.indices)
