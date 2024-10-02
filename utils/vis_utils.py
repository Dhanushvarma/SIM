import torch
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torch.nn.functional import interpolate


def run_inference_and_visualize(model, dataset, device, output_dir, num_samples=5):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    to_pil = ToPILImage()

    # Get category information from the dataset
    categories = dataset.categories
    num_categories = len(categories)
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_categories]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        # Move inputs to the correct device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            # Forward pass
            outputs = model(pixel_values=inputs["pixel_values"], multimask_output=False)

        # Get predicted mask
        predicted_masks = outputs.pred_masks.squeeze().sigmoid().cpu().numpy() > 0.5

        # Ensure predicted_masks has the correct number of channels
        if predicted_masks.ndim == 2:
            predicted_masks = np.repeat(
                predicted_masks[np.newaxis, :, :], num_categories, axis=0
            )

        # Get ground truth masks
        ground_truth_masks = inputs["ground_truth_masks"].squeeze().cpu().numpy()

        # Resize ground truth to match prediction if necessary
        if ground_truth_masks.shape[-2:] != predicted_masks.shape[-2:]:
            ground_truth_masks = (
                interpolate(
                    inputs["ground_truth_masks"].float(),
                    size=predicted_masks.shape[-2:],
                    mode="nearest",
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # Get original image
        original_image = to_pil(inputs["pixel_values"].squeeze().cpu())

        # Create color-coded masks
        ground_truth_colored = np.zeros((*ground_truth_masks.shape[1:], 3))
        predicted_colored = np.zeros((*predicted_masks.shape[1:], 3))

        for j, (gt_mask, pred_mask) in enumerate(
            zip(ground_truth_masks, predicted_masks)
        ):
            color = np.array(mcolors.to_rgb(colors[j]))
            ground_truth_colored += np.outer(gt_mask, color).reshape(*gt_mask.shape, 3)
            predicted_colored += np.outer(pred_mask, color).reshape(*pred_mask.shape, 3)

        # Normalize the colored masks
        ground_truth_colored = np.clip(ground_truth_colored, 0, 1)
        predicted_colored = np.clip(predicted_colored, 0, 1)

        # Visualize and save
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(ground_truth_colored)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")
        axs[2].imshow(predicted_colored)
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{i}.png"))
        plt.close(fig)

    # Create and save legend
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")
    for j, (cat_id, cat_name) in enumerate(categories.items()):
        ax.add_patch(plt.Rectangle((0, j), 0.3, 0.8, fc=colors[j]))
        ax.text(0.35, j + 0.4, cat_name, fontsize=12, va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "legend.png"))
    plt.close(fig)

    print(f"Visualizations saved in {output_dir}")
