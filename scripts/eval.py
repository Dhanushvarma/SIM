import torch
import monai
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torch.optim import Adam
from transformers import SamProcessor, SamModel

from utils.sam_dataset import CustomSAMDataset
from utils.common import read_json
from utils.vis_utils import run_inference_and_visualize


def main():
    image_dir = "/home/dhanush/SIM/data/rgb_10"
    coco_file = "/home/dhanush/SIM/data/coco_sam.json"
    output_dir = "results"
    coco_data = read_json(coco_file)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    finetuned_model_path = "/home/dhanush/SIM/checkpoints/sam_finetuned_model.pth"


    # Load your fine-tuned model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.load_state_dict(torch.load(finetuned_model_path, weights_only=True))
    model.to('cpu')

    # Create dataset (use the updated CustomSAMDataset class)
    dataset = CustomSAMDataset(coco_data, image_dir, processor)

    # Run inference and visualize
    run_inference_and_visualize(model, dataset, 'cpu', output_dir, num_samples=20)

if __name__ == "__main__":
    main()