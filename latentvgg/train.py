import os
import argparse
from tqdm import tqdm
import random

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import transformers
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import diffusers
from diffusers import DiffusionPipeline

from model import LatentVGG
from dataset import ImageDataset
from utils import tensor_to_latent, tensor_to_np

#------------------------------------------------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logging_dir',
        type=str,
        default='./logs',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='./ckpts'
    )
    parser.add_argument(
        '--tracker_project_name',
        type=str,
        default='latentvgg',
    )
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=100000,
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--log_freq',
        type=int,
        default=1000
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    logging_dir = args.logging_dir
    out_dir = args.out_dir
    accelerator_project_config = ProjectConfiguration(out_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='no',
        log_with='wandb',
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        accelerator.init_trackers(args.tracker_project_name)

    # # get sd to generate training images
    model_id = '../stable_diffusion/stable-diffusion-xl-base-1-0-local'
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.upcast_vae()
    vae = pipe.vae.to(accelerator.device)
    vae.enable_xformers_memory_efficient_attention()

    print('start loading data')
    data_root = '../../../datasets/LaionArt/data'
    train_set = ImageDataset(data_root, split='train')
    val_set = ImageDataset(data_root, split='val')
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4
    )
    print('end loading data')

    # get clip to calculate image feature gt
    vgg_model = torchvision.models.vgg16(weights='DEFAULT')
    vgg_preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vgg_model = vgg_model.eval().to(accelerator.device)

    @torch.no_grad()
    def get_gt_vgg(tensors):
        return vgg_model(vgg_preprocess(tensors))

    # define out model
    latent_vgg_model = LatentVGG()
    optimizer = torch.optim.AdamW(latent_vgg_model.parameters(), lr=args.learning_rate)
    loss_func = lambda x, y: F.l1_loss(x, y, reduction='mean')
 
    # training parameters
    max_train_steps = args.max_train_steps
    num_epoch = max_train_steps // len(train_loader) + 1
    batch_size = args.batch_size
    log_freq = args.log_freq
    total_batch_size = batch_size * accelerator.num_processes

    latent_vgg_model, optimizer, train_loader = accelerator.prepare(latent_vgg_model, optimizer, train_loader)

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    for e in range(num_epoch):
        for tensors in train_loader:
            with accelerator.accumulate(latent_vgg_model):
                with torch.no_grad():
                    latents = tensor_to_latent(vae, tensors)
                    gt = get_gt_vgg(tensors)
            
                pred = latent_vgg_model(latents)
                loss = loss_func(gt, pred)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            logs = {'train_loss': loss.item()}

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % log_freq == 0:
                        save_path = os.path.join(out_dir, 'checkpoint-' + str(global_step).zfill(10) + '.ckpt')
                        accelerator.unwrap_model(latent_vgg_model).save_pretrained(save_path)

                        with torch.no_grad():
                            val_tensors = val_set[random.choice([i for i in range(len(val_set))])]
                            val_tensors = val_tensors.unsqueeze(0).to(accelerator.device)
                            val_latents = tensor_to_latent(vae, val_tensors)
                            val_gt = get_gt_vgg(val_tensors)
                            val_pred = latent_vgg_model(val_latents)
                            val_loss = loss_func(val_gt, val_pred)
                            logs['val_loss'] = val_loss.item()
                        
                        images = tensor_to_np(tensors)
                        tracker = accelerator.get_tracker("wandb")
                        tracker.log({"training": wandb.Image(images, caption='inputs')})

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    save_path = os.path.join(out_dir, 'checkpoint.ckpt')
    accelerator.unwrap_model(latent_vgg_model).save_pretrained(save_path)
                        
            