import torch
import numpy as np
from collections.abc import Callable

def load_flist(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=int)

class NoiseSchedule:
    def __init__(self, T_steps, beta_min, beta_max, spacing_function=np.linspace):
        self.T_steps = T_steps
        self.betas = spacing_function(beta_min, beta_max, T_steps)
        self.alphas = 1.-self.betas
        self.gammas = np.cumprod(self.alphas, axis=0)
    def __getitem__(self, index):
        return self.alphas(index)
    def __len__(self):
        return self.T_steps

class TorchMetric:
    def __init__(self, metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.metric_fn = metric_fn
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:
        return self.metric_fn(output.squeeze(), target.squeeze())
    @property
    def name(self):
        return self.metric_fn.name+'->'+self.__class__.__name__

class NumpyMetric(TorchMetric):
    def __init__(self, metric_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        super().__init__(metric_fn=metric_fn)

    @torch.no_grad()
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:
        return self.metric_fn(
            output.detach().cpu().numpy().squeeze(), 
            target.detach().cpu().numpy().squeeze()
        )

