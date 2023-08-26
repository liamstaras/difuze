import torch
import os

from datetime import datetime
from socket import gethostname
from collections.abc import Callable

from . import data

class DataLogger:
    def __init__(
            self,
            base_directory_override: str = None,
            use_tensorboard: bool = False,
            timestamp_format: str = '%Y-%m-%d_%H-%M-%S',
            visual_function: Callable[[torch.Tensor], torch.Tensor] = lambda tensor: tensor.cpu(),
            save_functions: list[data.Saver] = []
        ):
        ## establish base path
        timestamp = datetime.now().strftime(timestamp_format)
        if base_directory_override is None:
            self.base_directory = os.path.join('runs', timestamp+'_'+gethostname())
        else:
            self.base_directory = base_directory_override
        
        ## initialize path and subdirectories
        self.log_path = os.path.join(self.base_directory, 'logfile.log')
        self.output_base_directory = os.path.join(self.base_directory, 'output')
        self.model_base_directory = os.path.join(self.base_directory, 'models')

        ## initialize Tensorboard if required
        if use_tensorboard:
            from torch.utils.tensorboard.writer import SummaryWriter
            self.tensorboard_base_directory = os.path.join(self.base_directory, 'tensorboard')
            self.summary_writer = SummaryWriter(log_dir=self.tensorboard_base_directory)
        else:
            self.summary_writer = None
        
        ## other parameters
        self.timestamp_format = timestamp_format
        self.visual_function = visual_function
        self.save_functions = save_functions
    
    def message(self, lines: str, also_print: bool = True) -> None:
        """Add a line to the log file

        lines: the text to add
        date_time: whether the date and time should be included in the log message
        """

        # make base directory if needed
        os.makedirs(self.base_directory, exist_ok=True)

        # split lines
        lines_list = lines.split('\n')

        # add a timestamp
        time_stamp = datetime.now().strftime(self.timestamp_format)

        # write the first line with a timestamp
        self._write_line(time_stamp+lines_list[0], self.log_path, also_print)

        # write the remaining lines without timestamps
        for line in lines_list[1:]:
            self._write_line(' '*(len(time_stamp)-3)+'=> '+line, self.log_path, also_print)

        
    @staticmethod
    def _write_line(line: str, log_path: str, also_print: bool) -> None:
        # print the output if desired
        if also_print:
            print(line)
        # write the output to the logfile
        with open(log_path, 'a') as log_file:
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
            tag: str,
            tensor: torch.Tensor,
            index: int,
            add_to_tensorboard: bool = True,
            save_locally: bool = True,
            filename_format: str = '{series}{index:0>8d}_{tag}'
        ) -> None:
        """Add an output tensor to Tensorboard and/or save it locally
        """

        if self.summary_writer is not None and add_to_tensorboard:
            self.summary_writer.add_image(series_name, self.visual_function(tensor), index)
        
        if save_locally:
            # make the output directory if it doesn't already exist
            os.makedirs(self.output_base_directory, exist_ok=True)

            # build the path from index and series name
            output_name = os.path.normpath(filename_format.format(index=index, series=series_name, tag=tag))

            # loop through each writer
            for saver in self.save_functions:
                # save the data using the current writer
                saver(tensor, self.output_base_directory, output_name)
    
    def state_dict(self, epoch_number: int, state_dict: dict, best: bool = False) -> None:
        """Save a state_dict to an automatically generated path

        epoch_number: the index of the current epoch
        best: whether "_BEST" should be added to the name of the file
        """

        # make the model directory if it doesn't already exist
        os.makedirs(self.model_base_directory, exist_ok=True)

        # add "_BEST" if this was the best epoch so far
        _best = '_BEST' if best else ''
        # generate output name
        name = 'model_{}{}'.format(epoch_number, _best)
        # save model to file
        torch.save(state_dict, os.path.join(self.model_base_directory, name))
