import torch.utils.data
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

image_size = 64

preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# Create a dataloader
batch_size = 32


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

# Loading a single batch of images
batch = next(iter(train_dataloader))
print('Shape:', batch['images'].shape,
      '\nBounds:', batch['images'].min().item(), 'to', batch['images'].max().item())

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(batch['images'][:4]):
    axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
    axs[i].set_axis_off()
fig.show()