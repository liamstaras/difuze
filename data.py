import torch.utils.data as data
import numpy as np
import os
import torch
from collections.abc import Callable

class NpyDataset(data.Dataset):
    def __init__(self, data_path: str, gt_index: int, cond_index: int, mask_index=None, start_index=0., stop_index=1.):
        self._data = np.load(data_path, mmap_mode='r')
        self._gt_index = gt_index
        self._cond_index = cond_index
        self._mask_index = mask_index
        self._single_shape = self._data[0,0].shape
        # make sure we have a colour channel axis, even if it is only of length 1
        if len(self._single_shape) == 2: self._single_shape = (1, *self._single_shape)
        self._blank_mask = torch.ones(self._single_shape)
        round_index = lambda index: round(index*len(self._data)) if type(index) is not int else index
        self._indices = np.arange(round_index(start_index), round_index(stop_index))
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # retrive the current data item
        item = self._data[self._indices[index]]
        # extract gt, cond and mask images from input
        gt_image = torch.tensor(item[self._gt_index].reshape(self._single_shape), dtype=torch.float32)
        cond_image = torch.tensor(item[self._cond_index].reshape(self._single_shape), dtype=torch.float32)
        if self._mask_index is not None:
            mask_image = torch.tensor(item[self._mask_index], dtype=torch.bool)
        else:
            mask_image = self._blank_mask
        return gt_image, cond_image, mask_image
    def __len__(self):
        return len(self._indices)

class Saver:
    def __init__(self, process_fn: Callable[[torch.Tensor], np.ndarray] = lambda tensor: tensor.cpu().numpy()):
        self.process_fn = process_fn
    def __call__(self, tensor, base_path, name):
        output_array = self.process_fn(tensor.detach())
        self._save(output_array, base_path, name)
    @staticmethod
    def _save(array, base_path, name):
        raise AttributeError('Must define how to save an array!')

class TifSaver(Saver):
    @staticmethod
    def _save(array, base_path, name):
        from PIL import Image
        output_path = os.path.join(base_path, name+'.tif')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray((array*255).astype(np.uint8).transpose((1,2,0))).save(output_path)

class NpySaver(Saver):
    @staticmethod
    def _save(array, base_path, name):
        output_path = os.path.join(base_path, name+'.npy')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, array)