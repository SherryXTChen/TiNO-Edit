# Copyright 2025 Zhejiang University Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim


from typing import Tuple, Union

import torch

from diffusers import PNDMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput

class TiNOEditPNDMScheduler(PNDMScheduler):
    def get_alphas_cumprod(self, t):
        if self.config.beta_schedule == "linear":
            start = self.config.beta_start
            end = self.config.beta_end
            get_alpha_func = lambda beta: 1 - beta
        elif self.config.beta_schedule == 'scaled_linear':
            start = self.config.beta_start**0.5
            end = self.config.beta_end**0.5
            get_alpha_func = lambda beta: 1 - beta**2
        else:
            raise NotImplementedError(f"{self.config.beta_schedule} does is not implemented for {self.__class__}")
        step = (end - start) / (self.config.num_train_timesteps - 1)

        alphas_cumprod = torch.tensor(1.0).to(t.device)
        while t >= 0:
            alphas_cumprod = alphas_cumprod * get_alpha_func(start + t * step)
            t = t - 1
        return alphas_cumprod

    def _get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        # See formula (9) of PNDM paper https://huggingface.co/papers/2202.09778
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation

        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        # alpha_prod_t = self.alphas_cumprod[timestep]
        # alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t = self.get_alphas_cumprod(timestep)
        alpha_prod_t_prev = self.get_alphas_cumprod(prev_timestep) if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.config.prediction_type == "v_prediction":
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        elif self.config.prediction_type != "epsilon":
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `v_prediction`"
            )

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        )

        return prev_sample

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        # self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # timesteps = timesteps.to(original_samples.device)

        # sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        #     sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        a = self.get_alphas_cumprod(timesteps)
        sqrt_alpha_prod = a ** 0.5

        # sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = (1 - a) ** 0.5

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps