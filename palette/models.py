from .guided_diffusion.unet import UNet
from diffusion import DiffusionModel
import torch

class PaletteModel(UNet, DiffusionModel):
    def forward(self, cond_image: torch.Tensor, noisy_image: torch.Tensor, gamma: torch.Tensor):
        noise_prediction = super().forward(torch.cat([cond_image, noisy_image], dim=1), gamma)
        return noise_prediction
    
    @torch.no_grad()
    def refinement_step(self, predicted_gt_image_t: torch.Tensor, cond_image: torch.Tensor, alpha_t: torch.Tensor, gamma_t: torch.Tensor):
        ### use paper Saharia et al. directly
        ## prepare
        # use the denoise function (UNet) to predict the noise noise_prediction
        predicted_noise_field = self(cond_image, predicted_gt_image_t, gamma_t)
        # find the coefficient of noise_prediction
        noise_prediction_coefft = (1-alpha_t) / torch.sqrt(1-gamma_t)
        # find mu_t, the mean of the final output distribution
        mu_t = (1/torch.sqrt(alpha_t))*(predicted_gt_image_t - noise_prediction_coefft*predicted_noise_field)
        # find sigma_t, the standard deviation of the final output distribution
        sigma_t = torch.sqrt(1-alpha_t)

        ## generate a sample image, by sampling from Gaussian distribution N(mu_theta, sigma_theta^2)
        # generate new Gaussian noise from a Z-dist
        epsilon = torch.randn_like(predicted_gt_image_t)
        # scale using X = sigma*Z + mu; output is refined prediction
        predicted_gt_image_tm1 = sigma_t*epsilon + mu_t
        return predicted_gt_image_tm1