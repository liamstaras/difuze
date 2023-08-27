import torch

class Metric:
    def __call__(self, predicted_gt_image: torch.Tensor, gt_image: torch.Tensor) -> float:
        raise AttributeError('Must define how to call metric!')
    @property
    def name(self):
        raise AttributeError('Must define name of metric!')