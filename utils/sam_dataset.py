import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as coco_mask


class CustomSAMDataset(Dataset):
    def __init__(self, coco_data, image_dir, processor):
        self.coco_data = coco_data
        self.image_dir = image_dir
        self.processor = processor
        self.image_ids = [img["id"] for img in coco_data["images"]]
        self.categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        self.num_categories = len(self.categories)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = next(
            img for img in self.coco_data["images"] if img["id"] == image_id
        )
        image_path = f"{self.image_dir}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")

        # Get annotations for this image
        annotations = [
            ann for ann in self.coco_data["annotations"] if ann["image_id"] == image_id
        ]

        # Create separate masks for each category
        masks = {
            cat_id: np.zeros((image_info["height"], image_info["width"]), dtype=bool)
            for cat_id in self.categories.keys()
        }

        for ann in annotations:
            rle = coco_mask.frPyObjects(
                ann["segmentation"], image_info["height"], image_info["width"]
            )
            mask = coco_mask.decode(rle)
            if len(mask.shape) > 2:
                mask = mask.sum(axis=2) > 0
            masks[ann["category_id"]] |= mask

        # Convert masks dictionary to a tensor
        mask_tensor = torch.stack([torch.from_numpy(mask) for mask in masks.values()])

        # Prepare inputs for the model
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add ground truth masks
        inputs["ground_truth_masks"] = mask_tensor

        return inputs
