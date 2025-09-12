import os
import numpy as np
from PIL import Image
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# from transformers import BlipProcessor, BlipForConditionalGeneration
# from transformers import AutoProcessor, AutoModelForCausalLM
from diffusers import Transformer2DModel
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# from diffusers.models.attention import GEGLU
# from diffusers.models.lora import LoRACompatibleLinear

# from groundingdino.util.inference import Model
# from segment_anything_hq import sam_model_registry, SamPredictor
from clip_interrogator import Config, Interrogator

#------------------------------------------------------------------------------------------------------------------

# get basename of file path without extension
def get_basename(path):
    return os.path.basename(os.path.splitext(path)[0])

# pad image to square shape with white pixels by default
def resize_image(image, size=512):
    width, height = image.size
    if width >= height:
        new_width = size
        new_height = int(height * (size / width))
    else:
        new_height = size
        new_width = int(width * (size / height))
    return image.resize((new_width, new_height))

def pad_image_to_square(image, color=(255, 255, 255), return_offset=False):
    width, height = image.size
    new_size = max(width, height)
    padded_image = Image.new("RGB", (new_size, new_size), color)
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2
    padded_image.paste(image, (x_offset, y_offset))
    if return_offset:
        return padded_image, (x_offset, y_offset)
    return padded_image

#------------------------------------------------------------------------------------------------------------------

def freeze_pipeline(pipeline):
    pipeline.vae.eval()
    for p in pipeline.vae.parameters():
        p.requires_grad = False

    pipeline.unet.eval()
    for p in pipeline.unet.parameters():
        p.requires_grad = False

    pipeline.text_encoder.eval()
    for p in pipeline.text_encoder.parameters():
        p.requires_grad = False

    try:
        pipeline.text_encoder_2.eval()
        for p in pipeline.text_encoder_2.parameters():
            p.requires_grad = False
    except:
        pass

def pil_to_tensor(pil):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return image_transforms(pil).unsqueeze(0)

def tensor_to_pil(tensor, normalize=True):
    if normalize:
        tensor = (tensor / 2 + 0.5)
    tensor = tensor.clamp(0, 1) * 255
    np_pil = tensor.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)[0]
    return Image.fromarray(np_pil)

def tensor_to_latent(pipe, tensor):
    return pipe.vae.encode(tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

def latent_to_tensor(pipe, latent):
    return pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
        
#------------------------------------------------------------------------------------------------------------------

def get_token(pipe, sentence, padding='max_length'):
        tokenizer = pipe.tokenizer
        inputs = tokenizer(
            [sentence], max_length=tokenizer.model_max_length, padding=padding, truncation=True, return_tensors='pt'
        ).input_ids[0]
        return inputs

def get_token_index(pipe, sentence, sub_sentence):
    assert sub_sentence in sentence, f'{sub_sentence} not a substring of {sentence}'
    sentence_tokens = get_token(pipe, sentence)
    sub_tokens = get_token(pipe, sub_sentence, padding=False)[1:-1]
    l = len(sub_tokens)

    for start in range(1, len(sentence_tokens)-l, 1):
        if torch.abs(sentence_tokens[start:start+l] - sub_tokens).max() < 1e-7:
            return [x for x in range(start, start+l)]

class ImageCaptionEstimator(nn.Module) :
    def __init__(self): 
        super().__init__()
        config = Config()
        config.blip_offload = True
        config.chunk_size = 2048
        config.flavor_intermediate_count = 512
        config.blip_num_beams = 64
        self.model = Interrogator(config)
        self.requires_grad = False

    def forward(self, image):
        generated_text = self.model.interrogate(image, max_flavors=4)
        return generated_text

#------------------------------------------------------------------------------------------------------------------

def crop_tensor_by_mask(tensor, mask, return_xy=False):
    nonzero_indices = torch.nonzero(mask[0,0])
    x1, y1 = tuple(nonzero_indices.min(dim=0)[0].tolist())
    x2, y2 = tuple(nonzero_indices.max(dim=0)[0].tolist())
    crop = tensor[..., x1:x2+1, y1:y2+1]
    if return_xy:
        return crop, torch.tensor([[y1, x1, y2, x2]])
    else:
        return crop

class SegMaskExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.seg_processor = CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
        self.seg_model = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined')
        self.seg_model.eval().to(device)

    def forward(self, image, text):
        inputs = self.seg_processor(text=text, images=[image], padding='max_length', return_tensors='pt').to(self.seg_model.device)
        outputs = self.seg_model(**inputs)
        preds = outputs.logits

        while len(preds.shape) < 4:
            preds = preds.unsqueeze(0)
        preds = F.interpolate(preds, size=(image.height, image.width), mode='bilinear')

        out = nn.Sigmoid()(preds)
        out = (out - out.min()) / (out.max() - out.min())
        out = (out > out.mean()).float().expand(-1, 3, -1, -1)
        return out
  