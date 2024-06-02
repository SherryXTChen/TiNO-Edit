import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

class LatentVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_vgg = torchvision.models.vgg16(weights='DEFAULT')
        self.latent_vgg.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, latents):
        return self.latent_vgg(latents)

    def save_pretrained(self, path):
        torch.save(self.latent_vgg.state_dict(), path)

    def load_pretrained(self, path):
        self.latent_vgg.load_state_dict(torch.load(path))
