from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from diffusers import DiffusionPipeline

from model import LatentVGG
from utils import tensor_to_latent
from dataset import ImageDataset

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # get sd to generate training images
    model_id = '../stable_diffusion/stable-diffusion-xl-base-1-0-local'
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.upcast_vae()
    vae = pipe.vae.to(device)
    vae.enable_xformers_memory_efficient_attention()

    print('start loading data')
    data_root = '../../../datasets/LaionArt/data'
    val_set = ImageDataset(data_root, split='val')
    print('end loading data')

    # get vgg to calculate image feature gt
    vgg_model = torchvision.models.vgg16(weights='DEFAULT').eval().to(device)
    vgg_preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # define out model
    latent_vgg_model = LatentVGG().eval().to(device)
    loss_func = lambda x, y: F.l1_loss(x, y, reduction='mean')

    @torch.no_grad()
    def get_gt_vgg(tensors):
        return vgg_model(vgg_preprocess(tensors))
    
    # def loss_func_inter(tensors, latents):
    #     vgg_block_list = []
    #     vgg_block_list.append(vgg_model.features[:4].eval())
    #     vgg_block_list.append(vgg_model.features[4:9].eval())
    #     vgg_block_list.append(vgg_model.features[9:16].eval())
    #     vgg_block_list.append(vgg_model.features[16:23].eval())

    #     for bl in vgg_block_list:
    #         for p in bl.parameters():
    #             p.requires_grad = False
    #     vgg_block_list = nn.ModuleList(vgg_block_list)

    #     latent_vgg_block_list = []
    #     latent_vgg_block_list.append(latent_vgg_model.latent_vgg.features[:4])
    #     latent_vgg_block_list.append(latent_vgg_model.latent_vgg.features[4:9])
    #     latent_vgg_block_list.append(latent_vgg_model.latent_vgg.features[9:16])
    #     latent_vgg_block_list.append(latent_vgg_model.latent_vgg.features[16:23])
    #     latent_vgg_block_list = nn.ModuleList(latent_vgg_block_list)

    #     vgg_features = []
    #     y = tensors
    #     with torch.no_grad():
    #         for _, block in enumerate(vgg_block_list):
    #             y = block(y)
    #             vgg_features.append(y)

    #     latent_vgg_features = []
    #     x = latent_vgg_model.transform(latents)
    #     for _, block in enumerate(latent_vgg_block_list):
    #         x = block(x)
    #         latent_vgg_features.append(x)

    #     loss = 0
    #     for gt, pred in zip(vgg_features, latent_vgg_features):
    #         loss = loss + F.l1_loss(gt, pred, reduction='mean')
    #     return loss
    

    data_root = '../../../datasets/LaionArt/data'
    val_set = ImageDataset(data_root, split='val')

    loss_total = 0
    total = 1000

    for i in tqdm(range(total)):
        tensor = val_set[i].unsqueeze(0).to(device)
        with torch.no_grad():
            gt = get_gt_vgg(tensor)

        with torch.no_grad():
            latent = tensor_to_latent(vae, tensor)
            pred = latent_vgg_model(latent)
        loss_total += loss_func(gt, pred)

    print(loss_total / total)