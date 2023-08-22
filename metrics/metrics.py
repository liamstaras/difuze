import numpy as np
from .support import Statistic, Series, relative_diff

class Metric:
    def __init__(self, statistic: Statistic):
        self.statistic = statistic
    def __call__(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise AttributeError('Must define function for generating output!')
    @property
    def name(self):
        return self.statistic.name+'->'+self.__class__.__name__

class RelativeDifference(Metric):
    def __call__(self, output, target):
        output_series = self.statistic(output)
        target_series = self.statistic(target)
        combined_x = np.union1d(output_series.x, target_series.x)
        output_interp = output_series.interpolate(combined_x)
        target_interp = target_series.interpolate(combined_x)
        diff_series = Series(combined_x, relative_diff(output_interp.y, target_interp.y))
        return diff_series.rms()