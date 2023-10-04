import torch
import torchvision
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline
import accelerate

device = "mps" if torch.has_mps else "cpu"

# Load the pipeline
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device);

# Sample an image
image_pipe().images[0]

from torch import nn
# The random starting point for a batch of 4 images
x = torch.randn(4, 3, 256, 256).to(device)

# Set the number of timesteps lower
image_pipe.scheduler.set_timesteps(num_inference_steps=30)

# Loop through the sampling timesteps
for i, t in enumerate(image_pipe.scheduler.timesteps):

    # Get the prediction given the current sample x and the timestep t
    with torch.no_grad():
        noise_pred = image_pipe.unet(x, t)["sample"]

    # Calculate what the updated sample should look like with the scheduler
    scheduler_output = image_pipe.scheduler.step(noise_pred, t, x)

    # Update x
    x = scheduler_output.prev_sample

    # Occasionally display both x and the predicted denoised images
    if i % 10 == 0 or i == len(image_pipe.scheduler.timesteps) - 1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
        axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
        axs[0].set_title(f"Current x (step {i})")

        pred_x0 = scheduler_output.pred_original_sample
        grid = torchvision.utils.make_grid(pred_x0, nrow=4).permute(1, 2, 0)
        axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
        axs[1].set_title(f"Predicted denoised images (step {i})")
        plt.show()