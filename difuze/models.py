import torch
import tqdm

from . import networks
from . import support

# define a placeholder class for diffusion models, demonstrating the necessity of a refinement_step method
class Diffusion(torch.nn.Module):
    @torch.no_grad()
    def refinement_step(self, predicted_gt_image_batch_t, cond_image_batch, alpha_t, gamma_t):
        raise AttributeError('Must define a refinement step!')
    
    @torch.no_grad()
    def infer_one_batch(self, cond_image_batch: torch.Tensor, mask: torch.BoolTensor, inference_noise_schedule: support.NoiseSchedule) -> torch.Tensor:
        """Sample from the neural network over a single batch of images

        cond_image_batch: the batch of conditioned images
        mask: a boolean array of pixels to ignore in predictions (NOT YET IMPLEMENTED)
        """

        # start with a noise field, inserting the cond_image_batch where the mask is present
        predicted_gt_image_batch = torch.randn_like(cond_image_batch)*mask + cond_image_batch*(1-mask)
        # iteratively apply the refinement step to denoise and renoise the image
        # note: t runs from T to 1
        for i in tqdm.tqdm(range(len(inference_noise_schedule))):
            # calculate the current t value from i
            t = len(inference_noise_schedule) - (i+1)
            # actually predict an image, and apply the mask
            predicted_gt_image_batch = self.refinement_step(
                predicted_gt_image_batch_t=predicted_gt_image_batch,
                cond_image_batch=cond_image_batch,
                alpha_t=torch.tensor(inference_noise_schedule.alphas[t], device=cond_image_batch.device),
                gamma_t=torch.tensor(inference_noise_schedule.gammas[t], device=cond_image_batch.device)
            )*mask + cond_image_batch*(1-mask) # apply masking
        return predicted_gt_image_batch


class Palette(networks.guided_diffusion.UNet, Diffusion):
    def forward(self, cond_image_batch: torch.Tensor, noisy_image_batch: torch.Tensor, gamma: torch.Tensor):
        noise_prediction = super().forward(torch.cat([cond_image_batch, noisy_image_batch], dim=1), gamma)
        return noise_prediction
    
    @torch.no_grad()
    def refinement_step(self, predicted_gt_image_batch_t: torch.Tensor, cond_image_batch: torch.Tensor, alpha_t: torch.Tensor, gamma_t: torch.Tensor):
        ### use paper Saharia et al. directly
        ## prepare
        # use the denoise function (UNet) to predict the noise noise_prediction
        predicted_noise_field = self(cond_image_batch, predicted_gt_image_batch_t, gamma_t)
        # find the coefficient of noise_prediction
        noise_prediction_coefft = (1-alpha_t) / torch.sqrt(1-gamma_t)
        # find mu_t, the mean of the final output distribution
        mu_t = (1/torch.sqrt(alpha_t))*(predicted_gt_image_batch_t - noise_prediction_coefft*predicted_noise_field)
        # find sigma_t, the standard deviation of the final output distribution
        sigma_t = torch.sqrt(1-alpha_t)

        ## generate a sample image, by sampling from Gaussian distribution N(mu_theta, sigma_theta^2)
        # generate new Gaussian noise from a Z-dist
        epsilon = torch.randn_like(predicted_gt_image_batch_t)
        # scale using X = sigma*Z + mu; output is refined prediction
        predicted_gt_image_batch_tm1 = sigma_t*epsilon + mu_t
        return predicted_gt_image_batch_tm1