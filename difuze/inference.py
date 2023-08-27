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
        training_hostname = checkpoint_state_dict['save_hostname']
        training_timestamp = checkpoint_state_dict['save_timestamp']
        self.epoch_number = checkpoint_state_dict['epoch_number']
        self.model.load_state_dict(checkpoint_state_dict['model_state_dict'])
        self.recent_rms_metrics = checkpoint_state_dict['recent_rms_metrics']
        self.configuration = checkpoint_state_dict['configuration']

        ## summarize configuration
        self.data_logger.message('\n'.join((
            "==== DIFFUSION MODEL INFERENCE ====",
            ":: begin checkpoint configuration summary",
            " --- hostname: {hostname}",
            " --- save timestamp: {timestamp}",
            " --- loss function: {loss_fn}",
            " --- optimizer: {optim}",
            " --- initial learning rate: {lr}",
            " --- (training) batch size: {train_batch}",
            ":: end checkpoint configuration summary",
            ":: begin inference configuration summary",
            " --- (inference) batch size: {inf_batch}",
            " --- refinement steps: {ref_steps}"
            )).format(
                hostname = training_hostname,
                timestamp = training_timestamp,
                loss_fn = self.configuration['loss_function'],
                optim = self.configuration['optimizer'],
                lr = self.configuration['initial_lr'],
                train_batch = self.configuration['batch_size'],
                inf_batch = self.inference_dataloader.batch_size,
                ref_steps = len(self.inference_noise_schedule)
            )
        )

    @torch.no_grad()
    def infer_all_data(self):
        """Sample from the neural network over a single iteration of the inference dataloader
        """

        # loop over the inference data, showing tqdm progress bar and tracking the index
        for i, data in enumerate(tqdm.tqdm(self.inference_dataloader)):
            gt_image_batch, cond_image_batch, mask, actual_index = data
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
                        series_name = 'Inference/Output/',
                        tag = visual_name,
                        tensor = visuals[visual_name],
                        index = actual_index[j]
                    )
        
  
    @torch.no_grad()
    def infer_one_batch(self, cond_image_batch: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Sample from the neural network over a single batch of images, and run metrics

        cond_image_batch: the batch of conditioned images
        mask: a boolean array of pixels to ignore in predictions (NOT YET IMPLEMENTED)
        """
        
        # place the model into validation mode
        self.model.eval()
        # carry out inference to predict the ground truth
        predicted_gt_image_batch = self.model.infer_one_batch(cond_image_batch, mask, self.inference_noise_schedule)
        
        return predicted_gt_image_batch
