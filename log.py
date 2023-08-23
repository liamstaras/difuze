import os
from datetime import datetime
from socket import gethostname
from collections.abc import Callable
from data import Saver
import torch

class DataLogger:
    def __init__(
            self,
            base_directory_override: str = None,
            use_tensorboard: bool = False,
            add_timestamps: bool = True,
            visual_function: Callable[[torch.Tensor], torch.Tensor] = lambda tensor: tensor.cpu(),
            save_functions: list[Saver] = []
        ):
        ## establish base path
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if base_directory_override is None:
            self.base_directory = os.path.join('runs', timestamp+'_'+gethostname())
        else:
            self.base_directory = base_directory_override
        
        ## initialize path and subdirectories
        self.log_path = os.path.join(self.base_directory, 'logfile.log')
        self.output_base_directory = os.path.join(self.base_directory, 'output')
        self.model_base_directory = os.path.join(self.base_directory, 'models')

        os.makedirs(self.base_directory, exist_ok=True)
        os.makedirs(self.output_base_directory, exist_ok=True)
        os.makedirs(self.model_base_directory, exist_ok=True)

        ## initialize Tensorboard if required
        if use_tensorboard:
            from torch.utils.tensorboard.writer import SummaryWriter
            self.tensorboard_base_directory = os.path.join(self.base_directory, 'tensorboard')
            self.summary_writer = SummaryWriter(log_dir=self.tensorboard_base_directory)
        else:
            self.summary_writer = None
        
        ## other parameters
        self.add_timestamps = add_timestamps
        self.visual_function = visual_function
        self.save_functions = save_functions
    
    def message(self, line: str, also_print: bool = True) -> None:
        """Add a line to the log file

        line: the text to add
        date_time: whether the date and time should be included in the log message
        """

        # add a timestamp if required
        if self.add_timestamps:
            line = datetime.now().strftime('%Y%m%d_%H%M%S: ') + line

        # print the output if desired
        if also_print:
            print(line)

        # write the output to the logfile
        with open(self.log_path, 'a') as log_file:
            log_file.write(line+'\n')
    
    def scalar(
            self,
            series_name: str,
            y_value: torch.Tensor,
            status_message: str = '',
            tensorboard_x_value: torch.Tensor = None,
            also_print: bool = False
        ) -> None:
        """Add a scalar to both the log file and Tensorboard, and optionally print the output
        """

        # add to tensorboard if required
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(series_name, y_value, tensorboard_x_value)
        
        # additionally, write to the regular log
        self.message('{} :: {}: {}'.format(status_message, series_name, float(y_value)), also_print=also_print)
    
    def tensor(
            self,
            series_name: str,
            tensor: torch.Tensor,
            index: int,
            add_to_tensorboard: bool = True,
            save_locally: bool = True,
            filename_format: str = '_{:0>6d}'
        ) -> None:
        """Add an output tensor to Tensorboard and/or save it locally
        """

        if self.summary_writer is not None and add_to_tensorboard:
            self.summary_writer.add_image(series_name, self.visual_function(tensor), index)
        
        if save_locally:
            # build the path from index and series name
            output_name = os.path.join(series_name, filename_format.format(index))

            # loop through each writer
            for saver in self.save_functions:
                # save the data using the current writer
                saver(tensor, self.output_base_directory, output_name)
    
    def model(self, epoch_number: int, model: torch.nn.Module, best: bool = False) -> None:
        """Save the state_dict of the model to an automatically generated path

        epoch_number: the index of the current epoch
        best: whether "_BEST" should be added to the name of the file
        """

        # add "_BEST" if this was the best epoch so far
        _best = '_BEST' if best else ''
        # generate output name
        name = 'model_{}{}'.format(epoch_number, _best)
        # save model to file
        torch.save(model.state_dict(), os.path.join(self.model_base_directory, name))
