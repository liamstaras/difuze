import torch
import random
import tqdm
from data import Saver
from support import NoiseSchedule, TorchMetric
from torch.utils.data import DataLoader
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
import numpy as np
from pprint import pformat
import os

# define a placeholder class for diffusion models, demonstrating the necessity of a refinement_step method
class DiffusionModel(torch.nn.Module):
    def refinement_step(self, predicted_gt_image_batch_t, cond_image_batch, alpha_t, gamma_t):
        raise AttributeError('Must define a refinement step!')

# a class containing the main algorithms for any diffusion model
class DiffusionFramework:
    def __init__(
        self,
        device: str,
        model: DiffusionModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        training_dataloader: DataLoader,
        training_noise_schedule: NoiseSchedule,
        training_loss_function: torch.nn.modules.loss._Loss,
        evaluation_dataloader: DataLoader,
        inference_noise_schedule: NoiseSchedule,
        evaluation_metrics: list[TorchMetric],
        base_path: str,
        summary_writer = None,
        save_functions: list[Saver] = [],
        visual_function: Callable[[torch.Tensor], np.ndarray] = lambda tensor: tensor.cpu().numpy()
    ):
        ## properties specified as arguments
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_dataloader = training_dataloader
        self.training_noise_schedule = training_noise_schedule
        self.training_loss_function = training_loss_function
        self.evaluation_dataloader = evaluation_dataloader
        self.inference_noise_schedule = inference_noise_schedule
        self.evaluation_metrics = evaluation_metrics
        self.base_path = base_path
        self.summary_writer = summary_writer
        self.save_functions = save_functions
        self.visual_function = visual_function

        ## initialize path and subdirectories
        self.log_path = os.path.join(self.base_path, 'logfile.log')
        self.image_base_path = os.path.join(self.base_path, 'output')
        self.model_base_path = os.path.join(self.base_path, 'models')
        self.loss_function_name = self.training_loss_function.__class__.__name__
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.image_base_path, exist_ok=True)
        os.makedirs(self.model_base_path, exist_ok=True)


    def train_single_epoch(self, epoch_number: int, log_every: int) -> torch.Tensor:
        """Train the neural network over a full iteration of the training dataloader

        epoch_number: the index of the current epoch, for logging purposes
        log_every: the number of iterations between each logging event
        """

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
            # display the current loss
            pbar.set_postfix_str('current learning rate: {:.2e}, current loss: {:.4f}'.format(float(self.optimizer.param_groups[-1]['lr']), float(current_loss)))
            running_loss += current_loss
            # check if we are at a logging iteration
            if i % log_every == 0:
                # find the mean loss over the past logging group
                mean_loss = running_loss/log_every
                # write to the log
                self.write_log_line(
                    'Epoch {: >3d}, iteration {: >6d}. Current learning rate is {:.2e}. Mean loss: {:.5f}.'.format(
                        epoch_number,
                        i,
                        self.optimizer.param_groups[-1]['lr'],
                        mean_loss
                    )
                )
                # determine the global index for logging purposes
                global_index = (epoch_number-1)*len(self.training_dataloader) + i
                # write to tensorboard (if initialized)
                self.log_scalar('train/'+self.loss_function_name, mean_loss, global_index)
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

    def evaluate_single_epoch(self, epoch_number: int) -> OrderedDict:
        """Sample from the neural network over a single iteration of the evaluation dataloader, and run metrics

        epoch_number: the index of the current epoch, for logging purposes
        """
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
        # log the FINAL visual from each batch
        self.log_visuals('Evaluation', epoch_number, cond_image_batch[-1], predicted_gt_image_batch[-1], gt_image_batch[-1], mask[-1])
        return mean_metric_results

    def evaluate_one_batch(self, gt_image_batch: torch.Tensor, cond_image_batch: torch.Tensor, mask: torch.BoolTensor) -> tuple[torch.Tensor, OrderedDict]:
        """Sample from the neural network over a single batch of images, and run metrics
        
        gt_image_batch: the batch of ground truth images
        cond_image_batch: the batch of conditioned images
        mask: a boolean array of pixels to ignore in predictions (NOT YET IMPLEMENTED)
        """
        
        # place the model into evaluation mode
        self.model.eval()
        # carry out inference to predict the ground truth
        predicted_gt_image_batch = self.infer_one_batch(cond_image_batch, mask)
        # run evaluation metrics on the image
        metric_results = OrderedDict(
            (metric.name, metric(output=predicted_gt_image_batch, target=gt_image_batch)) for metric in self.evaluation_metrics
        )
        return predicted_gt_image_batch, metric_results


    def infer_one_batch(self, cond_image_batch: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Sample from the neural network over a single batch of images

        cond_image_batch: the batch of conditioned images
        mask: a boolean array of pixels to ignore in predictions (NOT YET IMPLEMENTED)
        """

        # start with a noise field, inserting the cond_image_batch where the mask is present
        predicted_gt_image_batch = torch.randn_like(cond_image_batch)*mask + cond_image_batch*(1-mask)
        # iteratively apply the refinement step to denoise and renoise the image
        # note: t runs from T to 1
        for i in tqdm.tqdm(range(len(self.inference_noise_schedule))):
            # calculate the current t value from i
            t = len(self.inference_noise_schedule) - (i+1)
            # actually predict an image, and apply the mask
            predicted_gt_image_batch = self.model.refinement_step(
                predicted_gt_image_batch_t=predicted_gt_image_batch,
                cond_image_batch=cond_image_batch,
                alpha_t=torch.tensor(self.inference_noise_schedule.alphas[t], device=self.device),
                gamma_t=torch.tensor(self.inference_noise_schedule.gammas[t], device=self.device)
            )*mask + cond_image_batch*(1-mask) # apply masking
        return predicted_gt_image_batch
    
    def save(self, epoch_number: int, best: bool = False) -> None:
        """Save the state_dict of the model to an automatically generated path

        epoch_number: the index of the current epoch
        best: whether "_BEST" should be added to the name of the file
        """

        # add "_BEST" if this was the best epoch so far
        _best = '_BEST' if best else ''
        # generate output name
        name = 'model_{}{}'.format(epoch_number, _best)
        # save model to file
        torch.save(self.model.state_dict(), os.path.join(self.model_base_path, name))
    
    def write_log_line(self, line: str, date_time: bool = True, also_print: bool = False) -> None:
        """Add a line to the log file

        line: the text to add
        date_time: whether the date and time should be included in the log message
        also_print: whether to additionally display the output on the screen
        """

        if date_time:
            line = datetime.now().strftime('%Y%m%d_%H%M%S: ') + line
        if also_print: print(line)
        with open(self.log_path, 'a') as log_file:
            log_file.write(line+'\n')
    
    def log_scalar(self, series_name: str, y_value: float, x_value: float) -> None:
        """Add a scalar to TensorBoard

        series_name: the name of the series to add to
        y_value: the y coordinate of the scalar to add
        x_value: the x coordinate of the scalar to add
        """

        # if a tensorboard instance has been passed, use it to log the scalar
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(series_name, y_value, x_value)
    
    def log_visuals(self, series_name: str, index: int, cond_image: torch.Tensor, predicted_gt_image: torch.Tensor, gt_image: torch.Tensor, mask: torch.Tensor) -> None:
        """Write visuals to tensorboard and files

        series_name: the name of the series to add to
        index: the index of the current image set
        cond_image: a single conditioned image
        predicted_gt_image: a single predicted ground truth image
        gt_image: a single ground truth image
        """

        # remove any excess axes to avoid problems with processing
        cond_image_out = cond_image.squeeze()
        predicted_gt_image_out = predicted_gt_image.squeeze()
        gt_image_out = gt_image.squeeze()

        # use TensorBoard if it's enabled; make sure to run the visual_function to turn outputs into images
        if self.summary_writer is not None:
            self.summary_writer.add_image_batch(series_name+'/Conditioned', self.visual_function(cond_image_out), index)
            self.summary_writer.add_image_batch(series_name+'/Predicted', self.visual_function(predicted_gt_image_out), index)
            self.summary_writer.add_image_batch(series_name+'/Ground Truth', self.visual_function(gt_image_out), index)
        
        # generate a friendly file name stem
        name = '{}_{}_{:d}'.format(series_name, '{}', index)

        # save outputs to files, using each writer
        for writer in self.save_functions:
            writer(cond_image_out, self.image_base_path, name=name.format('Cond'))
            writer(predicted_gt_image_out, self.image_base_path, name=name.format('Pred'))
            writer(gt_image_out, self.image_base_path, name=name.format('GT'))

    def main_training_loop(self, log_every: int = 100, eval_every: int = 1, save_every: int = 1) -> None:
        """Run the training and evaluation cycle for the model

        log_every: the number of training iterations between each logging event
        eval_every: the number of epochs between each evaluation event
        save_every: the number of epochs between each forced save event; note that a best RMS evaluation score triggers saving anyway
        """

        # keep track of the epoch number and the best metric score
        epoch_number = 1 # zero indexing is confusing for the user
        best_rms_metrics = np.inf # we are trying to minimize this quantity

        while True:
            # add logging messages
            self.write_log_line('Begin epoch '+str(epoch_number), also_print=True)
            self.write_log_line('Beginning training...', also_print=True)

            # actually run training
            self.train_single_epoch(epoch_number, log_every)

            # keep track of whether we have saved the model this epoch
            saved = False

            # check if we are at an evaluation epoch
            if epoch_number % eval_every == 0:
                # notify the user that evaluation is about to commence
                self.write_log_line('This is an evaluation epoch. Beginning evalution...', also_print=True)

                # actually run evaluation, and store the results for each metric in an OrderedDict
                mean_metric_results = self.evaluate_single_epoch(epoch_number)

                # log the results using pretty print
                self.write_log_line('Mean evaluation results follow:', also_print=True)
                self.write_log_line(pformat(mean_metric_results), also_print=True)

                # calculate an RMS score
                rms_metrics = np.sqrt(np.mean(tuple(
                    mean_metric_results[metric_name]**2 for metric_name in mean_metric_results
                )))

                # display the RMS score
                self.write_log_line('The RMS value is {:.4f}'.format(rms_metrics), also_print=True)

                # write all metric scores to tensorboard
                for metric_name in mean_metric_results:
                    self.log_scalar('Evaluation/'+metric_name, mean_metric_results[metric_name], epoch_number)
                self.log_scalar('Evaluation/All_Metrics_RMS', rms_metrics, epoch_number)

                # see if this is the best epoch
                if rms_metrics <= best_rms_metrics:
                    # notify the user
                    self.write_log_line('This is the new best epoch!!', also_print=True)

                    # update the tracker
                    best_rms_metrics = rms_metrics

                    # save this model as the new best
                    self.write_log_line('Saving new best model...')
                    self.save(epoch_number, best=True)

                    # keep track of the fact that we've already saved
                    saved = True
            
            # see if we're at a save epoch
            if epoch_number % save_every == 0:
                # inform the user
                self.write_log_line('This is a save epoch.')

                # check if we've already saved
                if saved:
                    self.write_log_line('However, the model has already been saved this epoch. Resuming training.')
                else:
                    # save the model
                    self.write_log_line('Saving model...')
                    self.save(epoch_number, best=False)
                    self.write_log_line('Resuming training.', also_print=True)
            
            # increase the epoch number
            epoch_number += 1
            # consult the scheduler for learning rate change
            self.scheduler.step()
