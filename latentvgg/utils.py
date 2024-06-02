import numpy as np
from PIL import Image

import torchvision
from torchvision import transforms

def pil_to_tensor(pil):
    image_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return image_transforms(pil).unsqueeze(0)

def tensor_to_np(tensors):
    tensors = (tensors / 2 + 0.5).clamp(0, 1) * 255
    tensors = tensors.detach().cpu()
    tensors = torchvision.utils.make_grid(tensors, nrow=4).permute(1, 2, 0).numpy().astype(np.uint8)
    images = Image.fromarray(tensors)
    return images

def tensor_to_latent(vae, tensor):
    return vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor

def latent_to_tensor(vae, latent):
    return vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
