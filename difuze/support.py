import numpy as np

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
