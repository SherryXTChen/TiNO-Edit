from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms

from diffusers import DiffusionPipeline
import clip

from model import LatentCLIP
from utils import pil_to_tensor, tensor_to_latent
from dataset import ImageDataset

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # get sd to generate training images
    model_id = '../stable_diffusion/stable-diffusion-xl-base-1-0-local'
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.upcast_vae()
    vae = pipe.vae.to(device)
    vae.enable_xformers_memory_efficient_attention()

    # get clip to calculate image feature gt
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    clip_preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        clip_preprocess.transforms[-1]
    ])
    clip_model = clip_model.eval().to(device)

    data_root = '../../../datasets/LaionArt/data'
    val_set = ImageDataset(data_root, split='val')

    loss_func = lambda x, y: (1 - nn.CosineSimilarity()(x, y)).mean()

    loss_total = 0
    count = 0

    for tensor in tqdm(val_set):
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            gt = clip_model.encode_image(clip_preprocess(tensor))

        model = LatentCLIP()
        model = model.to(torch.float32)
        model.load_pretrained('./ckpts/checkpoint-0000150000.ckpt')
        model = model.to(device).eval()

        with torch.no_grad():
            latent = tensor_to_latent(vae, tensor)
            pred = model(latent)
        loss_total += loss_func(gt, pred).item()
        count  += 1
        if count >= 1000:
            break

    print(loss_total / count)