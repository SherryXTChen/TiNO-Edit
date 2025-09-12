

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel
from latentclip.model import LatentCLIP
from latentvgg.model import LatentVGG

#------------------------------------------------------------------------------------------------------------------

# VGG perceptual loss in Stable Diffusion latent domain
class LatentVGGLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        latent_vgg_model = LatentVGG()
        latent_vgg_model.load_pretrained('./latentvgg/checkpoint.ckpt') # should run this in the root folder

        blocks = []
        blocks.append(latent_vgg_model.latent_vgg.features[:4])
        blocks.append(latent_vgg_model.latent_vgg.features[4:9])
        blocks.append(latent_vgg_model.latent_vgg.features[9:16])
        blocks.append(latent_vgg_model.latent_vgg.features[16:23])
        self.blocks = nn.ModuleList(blocks).to(target.device).eval()
        self.blocks.requires_grad = False

        self.target_feat_list = []
        y = target.float()
        with torch.no_grad():
            for _, block in enumerate(self.blocks):
                y = block(y)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                self.target_feat_list.append((y, gram_y))

    def forward(self, input, feature_layers=[0, 1, 2, 3], style_layers=[]):   
        loss = 0.0
        x = input.float()
        for i, block in enumerate(self.blocks):
            x = block(x)
            y, gram_y = self.target_feat_list[i]
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                loss += F.l1_loss(gram_x, gram_y)
        return loss

# CLIP loss in Stable Diffusion latent domain
class LatentCLIPDirectionalLoss(nn.Module):
    def __init__(self, image1, text1, text2):
        super().__init__()
        self.latent_clip_model = LatentCLIP().to(image1.device).eval()
        self.latent_clip_model.load_pretrained('./latentclip/checkpoint.ckpt')
        self.latent_clip_model.requires_grad = False

        clip_model_id = "openai/clip-vit-base-patch32"
        clip_processor = CLIPProcessor.from_pretrained(clip_model_id) # , device=self.device)
        clip_model = CLIPModel.from_pretrained(clip_model_id)

        with torch.no_grad():
            text1_feat = clip_model.get_text_features(**clip_processor(text=[text1], return_tensors="pt"))
            text2_feat = clip_model.get_text_features(**clip_processor(text=[text2], return_tensors="pt"))
            
            self.text_diff = (text2_feat - text1_feat).to(image1.device, image1.dtype)
            self.image1_feat = self.latent_clip_model(image1).to(image1.dtype)

            del clip_processor, clip_model

    def forward(self, image2):
        image2_feat = self.latent_clip_model(image2).to(image2.dtype)
        return 1 - F.cosine_similarity(image2_feat - self.image1_feat, self.text_diff)
    

# KL divergence loss
def kl_to_standard_normal(img):
    img_flat = img.view(-1)
    mu = img_flat.mean()
    sigma = img_flat.std(unbiased=False)

    # Avoid numerical instability
    sigma_sq = sigma ** 2 + 1e-4
    kl = 0.5 * (mu ** 2 + sigma_sq - torch.log(sigma_sq) - 1.0)
    return kl