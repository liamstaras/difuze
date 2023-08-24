import torch
import torch.utils.data
import tqdm

from collections import OrderedDict

from . import log
from . import support
from . import models

# a class containing the main algorithms for training any diffusion model
class InferenceFramework:
    def __init__(
        self,
        device: str,
        model: models.Diffusion,
        checkpoint_state_dict: dict,
        inference_dataloader: torch.utils.data.DataLoader,
        inference_noise_schedule: support.NoiseSchedule,
        data_logger: log.DataLogger
    ):
        ## properties specified as arguments
        self.device = device
        self.model = model.to(device)
        self.inference_dataloader = inference_dataloader
        self.inference_noise_schedule = inference_noise_schedule
        self.data_logger = data_logger

        ## load states from checkpoint
        self.epoch_number = checkpoint_state_dict['epoch_number']
        self.model.load_state_dict(checkpoint_state_dict['model_state_dict'])
        self.recent_rms_metrics = checkpoint_state_dict['recent_rms_metrics']
        self._initial_learning_rate = checkpoint_state_dict['initial_lr']
        self.training_batch_size = checkpoint_state_dict['batch_size']

        ## summarize configuration
        self.data_logger.message('\n'.join((
            "==== DIFFUSION MODEL INFERENCE ====",
            ":: begin checkpoint configuration summary",
            " --- loss function: {loss_fn}",
            " --- optimizer: {optim}",
            " --- initial learning rate: {lr}",
            " --- batch size: {batch}",
            ":: end checkpoint configuration summary"
            )).format(
                loss_fn = '--',
                optim = '--',
                lr = self._initial_learning_rate,
                batch = self.training_batch_size
            )
        )

    @torch.no_grad()
    def infer_all_data(self):
        """Sample from the neural network over a single iteration of the inference dataloader
        """

        # loop over the inference data, showing tqdm progress bar and tracking the index
        for i, data in enumerate(tqdm.tqdm(self.inference_dataloader)):
            gt_image_batch, cond_image_batch, mask = data
            # infer the image
            predicted_gt_image_batch = self.infer_one_batch(cond_image_batch.to(self.device), mask.to(self.device))
            
            # log the each visual from the batch
            for j in range(len(predicted_gt_image_batch)):
                visuals = OrderedDict((
                    ('Cond', cond_image_batch[j].squeeze()),
                    ('Pred', predicted_gt_image_batch[j].squeeze()),
                    ('GT', gt_image_batch[j].squeeze())
                ))
                for visual_name in visuals:
                    self.data_logger.tensor(
                        series_name = 'Inference/Output/'+visual_name,
                        tensor = visuals[visual_name],
                        index = i
                    )
        
  
    @torch.no_grad()
    def infer_one_batch(self, cond_image_batch: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Sample from the neural network over a single batch of images, and run metrics

        cond_image_batch: the batch of conditioned images
        mask: a boolean array of pixels to ignore in predictions (NOT YET IMPLEMENTED)
        """
        
        # place the model into evaluation mode
        self.model.eval()
        # carry out inference to predict the ground truth
        predicted_gt_image_batch = self.model.infer_one_batch(cond_image_batch, mask, self.inference_noise_schedule)
        
        return predicted_gt_image_batch