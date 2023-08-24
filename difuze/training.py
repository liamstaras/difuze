import torch
import torch.utils.data
import numpy as np
import random
import tqdm

from collections import OrderedDict

from . import log
from . import support
from . import models
from . import metrics

# a class containing the main algorithms for training any diffusion model
class TrainingFramework:
    def __init__(
        self,
        device: str,
        model: models.Diffusion,
        optimizer: torch.optim.Optimizer,
        loss_scheduler,
        training_dataloader: torch.utils.data.DataLoader,
        training_noise_schedule: support.NoiseSchedule,
        training_loss_function: torch.nn.modules.loss._Loss,
        evaluation_dataloader: torch.utils.data.DataLoader,
        inference_noise_schedule: support.NoiseSchedule,
        evaluation_metrics: list[metrics.TorchMetric],
        data_logger: log.DataLogger,

        state_dict_to_load: dict | None = None,
        metric_scheduler = None
    ):
        ## properties specified as arguments
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_scheduler = loss_scheduler
        self.training_dataloader = training_dataloader
        self.training_noise_schedule = training_noise_schedule
        self.training_loss_function = training_loss_function
        self.evaluation_dataloader = evaluation_dataloader
        self.inference_noise_schedule = inference_noise_schedule
        self.evaluation_metrics = evaluation_metrics
        self.data_logger = data_logger

        self.metric_scheduler = metric_scheduler

        self.epoch_number = 0

        ## summarize configuration
        self.data_logger.message('\n'.join((
            "==== DIFFUSION MODEL TRAINING ====",
            ":: begin configuration summary",
            " --- loss function: {loss_fn}",
            " --- optimizer: {optim}",
            " --- initial learning rate: {lr}",
            " --- batch size: {batch}",
            ":: end configuration summary"
            )).format(
                loss_fn = self.training_loss_function.__class__.__name__,
                optim = self.optimizer.__class__.__name__,
                lr = self.optimizer.param_groups[-1]['lr'],
                batch = self.training_dataloader.batch_size
            )
        )

        if state_dict_to_load is not None:
            self.data_logger.message('Loading state from dict...')
            self.state = state_dict_to_load

        self._initial_learning_rate = self.optimizer.param_groups[-1]['lr']


    def train_single_epoch(self, log_every: int) -> torch.Tensor:
        """Train the neural network over a full iteration of the training dataloader

        log_every: the number of iterations between each logging event
        """

        # update the epoch number
        self.epoch_number += 1

        # add logging messages
        self.data_logger.message('This is epoch '+str(self.epoch_number), also_print=True)
        self.data_logger.message('Beginning training...', also_print=True)

        # zero loss counters
        running_loss = 0.
        current_loss = 0.
        # loop over the training data, showing tqdm progress bar and tracking the index
        # use tqdm to show a progress bar, to which we can affix the current loss rate
        pbar = tqdm.tqdm(self.training_dataloader, postfix='current learning rate: --------, current loss: ------')
        # using zero indexing is annoying for mean calc, so start from 1
        for i, data in enumerate(pbar, start=1):
            # extract the images from the loaded data
            gt_image_batch, cond_image_batch, mask = data

            # add the loss to the cumulative total
            current_loss = self.train_one_batch(gt_image_batch.to(self.device), cond_image_batch.to(self.device), mask.to(self.device))

            # display the current learning rate and loss on the progress bat
            pbar.set_postfix_str('current learning rate: {:.2e}, current loss: {:.4f}'.format(float(self.optimizer.param_groups[-1]['lr']), float(current_loss)))

            running_loss += current_loss
            # check if we are at a logging iteration
            if i % log_every == 0:
                # find the mean loss over the past logging group
                mean_loss = running_loss/log_every

                # determine the global index for logging purposes - equal to the number of images seen by the model
                global_index = ((self.epoch_number-1)*len(self.training_dataloader) + i)*self.training_dataloader.batch_size

                # write to the log
                self.data_logger.scalar(
                    series_name = 'Train/'+self.training_loss_function.__class__.__name__,
                    y_value = mean_loss,
                    status_message = 'Epoch {: >3d}, iteration {: >6d}. Current learning rate is {:.2e}'.format(
                        self.epoch_number,
                        i,
                        self.optimizer.param_groups[-1]['lr']
                    ),
                    tensorboard_x_value = global_index,
                )

                # zero the running counter
                running_loss = 0.

        # return the most recent loss
        return mean_loss


    def train_one_batch(self, gt_image_batch: torch.Tensor, cond_image_batch: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Train the neural network over a single batch of images
        
        gt_image_batch: the batch of ground truth images
        cond_image_batch: the batch of conditioned images
        mask: a boolean array of pixels to ignore in predictions (NOT YET IMPLEMENTED)
        """

        # place the model into training mode
        self.model.train()
        # reset optimizer gradients
        self.optimizer.zero_grad()
        # randomly pick t, and sample the corresponding gamma
        t = random.randrange(len(self.training_noise_schedule))
        gamma = torch.tensor(self.training_noise_schedule.gammas[t], device=self.device)
        # generate N(0,1) noise field
        noise_field = torch.randn_like(gt_image_batch, device=self.device)
        # add noise to the gt_image_batch at an appropriate level
        noisy_image_batch = torch.sqrt(gamma)*gt_image_batch + torch.sqrt(1-gamma)*noise_field
        # return predict the noise field using the nn
        predicted_noise_field = self.model(cond_image_batch, noisy_image_batch, gamma)
        # calculate loss
        loss = self.training_loss_function(noise_field, predicted_noise_field)
        loss.backward()
        # adjust learning weights
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def evaluate_single_epoch(self, epoch_number: int) -> float:
        """Sample from the neural network over a single iteration of the evaluation dataloader, and run metrics

        epoch_number: the index of the current epoch, for logging purposes
        """
        # inform the user that evaluation is about to commence
        self.data_logger.message('This is an evaluation epoch. Beginning evalution...', also_print=True)
        
        # make ordered dict to store lists of metric results
        all_metric_results = OrderedDict(
            (metric.name, []) for metric in self.evaluation_metrics
        )
        # loop over the evaluation data, showing tqdm progress bar and tracking the index
        for i, data in enumerate(tqdm.tqdm(self.evaluation_dataloader)):
            gt_image_batch, cond_image_batch, mask = data
            # get the current metric results
            predicted_gt_image_batch, metric_results = self.evaluate_one_batch(gt_image_batch.to(self.device), cond_image_batch.to(self.device), mask.to(self.device))
            # loop over all metrics, and add the result for each image in the batch to the list for this epoch
            for key in metric_results:
                all_metric_results[key].extend(metric_results[key])
        # create a new OrderedDict to store the mean metric results, by dividing through by the length of the dataloader
        mean_metric_results = OrderedDict(
            # use nanmean to avoid polluting the mean with any stray NaNs
            (key, np.nanmean(all_metric_results[key])) for key in metric_results
        )

        # calculate an RMS score
        rms_metrics = np.sqrt(np.mean(tuple(
            mean_metric_results[metric_name]**2 for metric_name in mean_metric_results
        )))
        mean_metric_results['All_Metrics_RMS'] = rms_metrics

        self.data_logger.message('Metric results:', also_print=True)

        # step the metric scheduler if present
        if self.metric_scheduler is not None:
            self.metric_scheduler.step(rms_metrics)

        # log all metric scores
        for metric_name in mean_metric_results:
            self.data_logger.scalar(
                series_name = 'Evaluation/Metric/'+metric_name,
                y_value = mean_metric_results[metric_name],
                tensorboard_x_value = epoch_number,
                also_print = True
            )
        
        # log the FINAL visual from the FINAL batch
        final_visuals = OrderedDict((
            ('Cond', cond_image_batch[-1].squeeze()),
            ('Pred', predicted_gt_image_batch[-1].squeeze()),
            ('GT', gt_image_batch[-1].squeeze())
        ))
        for visual_name in final_visuals:
            self.data_logger.tensor(
                series_name = 'Evaluation/Visual/'+visual_name,
                tensor = final_visuals[visual_name],
                index = epoch_number,
            )
        
        # return the RMS score for determining the best epoch
        return rms_metrics
    
    @torch.no_grad()
    def evaluate_one_batch(self, gt_image_batch: torch.Tensor, cond_image_batch: torch.Tensor, mask: torch.BoolTensor) -> tuple[torch.Tensor, OrderedDict[str, tuple[float]]]:
        """Sample from the neural network over a single batch of images, and run metrics
        
        gt_image_batch: the batch of ground truth images
        cond_image_batch: the batch of conditioned images
        mask: a boolean array of pixels to ignore in predictions (NOT YET IMPLEMENTED)
        """
        
        # place the model into evaluation mode
        self.model.eval()
        # carry out inference to predict the ground truth
        predicted_gt_image_batch = self.model.infer_one_batch(cond_image_batch, mask, self.inference_noise_schedule)
        # run evaluation metrics on the image
        metric_results = OrderedDict(
            (metric.name, metric(output=predicted_gt_image_batch, target=gt_image_batch)) for metric in self.evaluation_metrics
        )
        return predicted_gt_image_batch, metric_results
    
    @torch.no_grad()
    @property
    def state(self) -> dict:
        # build a state dict containing all important parts of the framework
        state_dict = {
            'epoch_number': self.epoch_number,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_scheduler_state_dict': self.loss_scheduler.state_dict(),
            'recent_rms_metrics': self.recent_rms_metrics,
            'best_rms_metrics': self.best_rms_metrics,
            'initial_lr': self._initial_learning_rate,
            'batch_size': self.training_dataloader.batch_size
        }

        # add the metric scheduler if it exists
        if self.metric_scheduler is not None:
            state_dict['metric_scheduler_state_dict'] = self.metric_scheduler.state_dict()

        return state_dict
    
    @torch.no_grad()
    @state.setter
    def state(self, state_dict: dict):
        # load important properties from a state dict
        self.epoch_number = state_dict['epoch_number']
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.loss_scheduler.load_state_dict(state_dict['loss_scheduler_state_dict'])
        self.recent_rms_metrics = state_dict['recent_rms_metrics']
        self._initial_learning_rate = state_dict['initial_lr']  
    

    def main_training_loop(self, log_every: int = 100, eval_every: int = 1, save_every: int = 1) -> None:
        """Run the training and evaluation cycle for the model

        log_every: the number of training iterations between each logging event
        eval_every: the number of epochs between each evaluation event
        save_every: the number of epochs between each forced save event; note that a best RMS evaluation score triggers saving anyway
        """

        # keep track of the epoch number and the best metric score
        self.best_rms_metrics = np.inf # we are trying to minimize this quantity

        while True:
            

            # actually run training
            self.train_single_epoch(log_every)

            # keep track of whether we have saved the model this epoch
            saved = False

            # check if we are at an evaluation epoch
            if self.epoch_number % eval_every == 0:
                # actually run evaluation, storing the RMS of all the metrics
                self.recent_rms_metrics = self.evaluate_single_epoch()

                # see if this is the best epoch
                if self.recent_rms_metrics <= self.best_rms_metrics:
                    # notify the user
                    self.data_logger.message('This is the new best epoch!!', also_print=True)

                    # update the tracker
                    self.best_rms_metrics = self.recent_rms_metrics

                    # save this model as the new best
                    self.data_logger.message('Saving new best model...')
                    self.data_logger.state_dict(
                        epoch_number = self.epoch_number,
                        state_dict = self.state,
                        best = True
                    )

                    # keep track of the fact that we've already saved
                    saved = True
            
            # see if we're at a save epoch
            if self.epoch_number % save_every == 0:
                # inform the user
                self.data_logger.message('This is a save epoch.', also_print=True)

                # check if we've already saved
                if saved:
                    self.data_logger.message('However, the model has already been saved this epoch. Resuming training.', also_print=True)
                else:
                    # save the model
                    self.data_logger.message('Saving model...', also_print=True)
                    self.data_logger.state_dict(
                        epoch_number = self.epoch_number,
                        state_dict = self.state,
                        best = False
                    )
                    self.data_logger.message('Resuming training.', also_print=True)

            # consult the scheduler for learning rate change
            self.loss_scheduler.step()
