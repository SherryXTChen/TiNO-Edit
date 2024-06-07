import torch
import torch.nn as nn
from torchvision import transforms
import clip

class LatentCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model, _ = clip.load("ViT-B/32")       
        self.dtype_type = torch.float32
        self.latent_clip = clip_model.visual.to(self.dtype_type)

        self.transform = transforms.Compose([
            transforms.Resize((63, 63), interpolation=transforms.InterpolationMode.BICUBIC),
        ])
        self.latent_clip.conv1 = torch.nn.Conv2d(4, 768, kernel_size=(9, 9), stride=(9, 9), bias=False)

    def forward(self, latents):
        return self.latent_clip(self.transform(latents).to(self.dtype_type))

    def save_pretrained(self, path):
        torch.save(self.latent_clip.state_dict(), path)

    def load_pretrained(self, path):
        self.latent_clip.load_state_dict(torch.load(path))

if __name__ == '__main__':
    model = LatentCLIP()
    print(model.latent_clip)