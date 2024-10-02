import torch
import monai
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torch.optim import Adam
from transformers import SamProcessor, SamModel

from utils.sam_dataset import CustomSAMDataset
from utils.common import read_json


class SAMTrainer:
    def __init__(
        self,
        model_name,
        coco_data,
        image_dir,
        batch_size=2,
        learning_rate=1e-5,
        num_epochs=10,
    ):
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name)
        self.dataset = CustomSAMDataset(coco_data, image_dir, self.processor)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = Adam(
            self.model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0
        )
        self.loss_fn = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.num_epochs = num_epochs

    def freeze_encoders(self):
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    def prepare_model(self):
        self.freeze_encoders()
        self.model.to(self.device)
        self.model.train()

    def train(self):
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss.item()
            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(self.dataloader)}"
            )

    def train_step(self, batch):
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = self.model(
            pixel_values=inputs["pixel_values"], multimask_output=False
        )
        predicted_masks, ground_truth_masks = self.prepare_masks(
            outputs.pred_masks, inputs["ground_truth_masks"]
        )
        loss = self.loss_fn(predicted_masks, ground_truth_masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def prepare_masks(self, predicted_masks, ground_truth_masks):
        # Convert boolean tensor to float
        ground_truth_masks = ground_truth_masks.float()

        if ground_truth_masks.shape[-2:] != predicted_masks.shape[-2:]:
            ground_truth_masks = interpolate(
                ground_truth_masks, size=predicted_masks.shape[-2:], mode="nearest"
            )
        num_categories = ground_truth_masks.shape[1]
        predicted_masks = predicted_masks.squeeze(2).repeat(1, num_categories, 1, 1)
        return predicted_masks, ground_truth_masks

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def main():
    model_name = "facebook/sam-vit-base"
    coco_file_path = "/home/dhanush/SIM/data/coco_sam.json"
    image_dir = "/home/dhanush/SIM/data/rgb_10"

    coco_data = read_json(coco_file_path)

    trainer = SAMTrainer(model_name, coco_data, image_dir)
    trainer.prepare_model()
    trainer.train()
    trainer.save_model("sam_finetuned_model.pth")


if __name__ == "__main__":
    main()
