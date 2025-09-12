import os
import argparse
from PIL import Image
import random

from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput

from diffusers.utils import deprecate, is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps, retrieve_latents
from diffusers import StableDiffusionImg2ImgPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from loss import LatentVGGLoss, LatentCLIPDirectionalLoss, kl_to_standard_normal
from scheduling_ddim import TiNOEditDDIMScheduler
from scheduling_pndm import TiNOEditPNDMScheduler

def project_decreasing(x):
    with torch.no_grad():
        x[-1] = torch.clamp(x[-1], min=0, max=max(0, x[-2]-10))
        for i in range(len(x)-2, 1, -1):
            x[i] = torch.clamp(x[i], min=min(x[i+1]+10, x[i-1]-10), max=max(x[i+1]+10, x[i-1]-10))
        x[0] = torch.clamp(x[0], max=1000)
    return x

class TiNOEditStrokeGuidedImageEditingPipeline(StableDiffusionImg2ImgPipeline):
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                    image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                    )

                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        return init_latents
    
    def __call__(
        self,
        original_prompt: Union[str, List[str]] = None,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask: PipelineImageInput = None,
        strength: float = 0.5,
        num_inference_steps: Optional[int] = 10,
        num_optimization_steps: Optional[int] = 10,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self.scheduler.timesteps = self.scheduler.timesteps.to(device, prompt_embeds.dtype)
        timesteps = timesteps.to(device, prompt_embeds.dtype)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        init_latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        init_latents = init_latents.to(noise.dtype)

        # locate editing region
        mask = F.interpolate(mask, size=init_latents.shape[-2:], mode="bilinear").to(device, dtype=prompt_embeds.dtype)

        # enable gradient
        timesteps.requires_grad = True
        noise.requires_grad = True

        # loss function and optimizer
        vgg_loss_func = LatentVGGLoss(init_latents * mask)
        clip_loss_func =  LatentCLIPDirectionalLoss(init_latents * mask, original_prompt, prompt)
        
        optimizer_timesteps = torch.optim.AdamW([timesteps], lr=1, eps=1e-4)
        optimizer_noise = torch.optim.AdamW([noise], lr=0.005, eps=1e-4)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_optimization_steps) as progress_bar:
            for _ in range(num_optimization_steps):
                optimizer_timesteps.zero_grad()
                optimizer_noise.zero_grad()

                # get latents
                latents = self.scheduler.add_noise(init_latents, noise, latent_timestep) * mask + init_latents * (1 - mask)

                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0] * mask + init_latents * (1 - mask)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    if XLA_AVAILABLE:
                        xm.mark_step()

                loss_dict = {
                    "vgg_loss": vgg_loss_func(latents * mask),
                    "clip_loss": clip_loss_func(latents * mask),
                }
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer_timesteps.step()
                optimizer_timesteps.step()
                project_decreasing(timesteps)

                if type(self.scheduler) == TiNOEditPNDMScheduler:
                    self.scheduler.ets = []
                    self.scheduler.counter = 0
                    self.scheduler.cur_model_output = 0

                # print({k:v.detach() for k,v in loss_dict.items()}, timesteps.detach())
                progress_bar.update()

        latents = latents.detach()
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()
           
        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    

# program argument and argument value preprocessing
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_image_path",
        type=str,
        default="examples/2_input.jpg",
        help="path to the original image",
    )
    parser.add_argument(
        "--user_input_image_path",
        type=str,
        default="examples/2_user.jpg",
        help="path to the user input image",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="examples/2_output.jpg",
        help="path to save the output image",
    )
    parser.add_argument(
        "--original_prompt",
        type=str,
        default="Snoopy and Charlie",
        help="Description of the original image",
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        default="Snoopy and Charlie next to an apple tree",
        help="Description of the target image"
    )
    parser.add_argument(
        "--num_optimization_steps",
        type=int,
        default=10,
        help="Number of optimization steps, each is a full reversed diffusion process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="seed for reproducing results"
    )
    
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = random.randint(0, 1e10)
    torch.manual_seed(args.seed)
    return args
  

if __name__ == "__main__":
    args = parse_arguments()
    
    # load stable diffusion
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    device = "cuda"
    pipe = TiNOEditStrokeGuidedImageEditingPipeline.from_pretrained(model_id , safety_checker=None, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()

    # initialize scheduler
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config["timestep_spacing"] = "trailing"
    del scheduler_config["skip_prk_steps"]
    pipe.scheduler = TiNOEditDDIMScheduler(**scheduler_config)

    # get editing region
    original_image = Image.open(args.original_image_path)
    user_input_image = Image.open(args.user_input_image_path)

    image1_array = np.array(original_image)
    image2_array = np.array(user_input_image)
    difference = np.abs(image1_array - image2_array)
    gray_difference = np.mean(difference, axis=2)
    threshold_value = 30
    binary_mask = gray_difference > threshold_value
    tensor_mask = torch.from_numpy(binary_mask).float()
    mask = tensor_mask.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)

    # run optimization
    negative_prompt = 'out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,'
    image = pipe(
        original_prompt=args.original_prompt,
        prompt=args.target_prompt, 
        negative_prompt=negative_prompt,
        image=user_input_image,
        mask=mask,
        generator=torch.manual_seed(args.seed),
    ).images[0]
    image.save(args.output_image_path)
