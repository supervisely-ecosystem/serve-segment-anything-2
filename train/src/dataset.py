from torch.utils.data import Dataset
import supervisely as sly
from src.utils import get_multilabel_mask, resize_image_and_mask


class SAM2Dataset(Dataset):
    def __init__(self, sly_items, sly_project_meta):
        self.sly_items = sly_items
        self.sly_project_meta = sly_project_meta

    def __len__(self):
        return len(self.sly_items)

    def __getitem__(self, idx):
        sly_item = self.sly_items[idx]
        img_path = sly_item.img_path
        ann_path = sly_item.ann_path
        img_np = sly.image.read(img_path)
        ann = sly.Annotation.load_json_file(ann_path, self.sly_project_meta)
        mask = get_multilabel_mask(img_np, ann)
        img_np, mask = resize_image_and_mask(img_np, mask)
        return img_np, mask, ann_path
