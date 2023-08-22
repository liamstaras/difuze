import numpy as np
import math

class Series:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
    def interpolate(self, new_x):
        return Series(new_x, np.interp(new_x, self.x, self.y))
    def mean(self):
        return self._mean(self.x, self.y)
    def rms(self):
        return np.sqrt(self._mean(self.x, self.y**2))
    @staticmethod
    def _mean(x, y):
        area = np.trapz(y, x)
        x_range = np.ptp(x)
        return area/x_range


class Statistic:
    def __init__(self):
        pass
    def __call__(self, input: np.ndarray) -> Series:
        raise AttributeError('Must define function for generating output!')
    @property
    def name(self):
        return self.__class__.__name__

class MeanSeries:
    # x: x values, y_mean: y values, y_std_err: standard error in the mean for each y value
    def __init__(self, x, y_mean, y_std, sample_size):
        self.x = x
        self.y_mean = y_mean
        self.y_std = y_std
        self.sample_size = sample_size
    @property
    def y_std_err(self):
        return self.y_std/np.sqrt(self.sample_size)

def density_series(input: np.ndarray, bins, data_range=None):
    histogram, bin_edges = np.histogram(input, bins=bins, range=data_range)
    bin_means = (bin_edges[:-1]+bin_edges[1:])/2
    return Series(bin_means, histogram)

@np.vectorize
def relative_diff(output, target):
    if output==target==0:
        return 0
    else:
        return (output-target)/np.sqrt(output**2 + target**2)