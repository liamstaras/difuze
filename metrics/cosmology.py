from .support import Statistic, Series, density_series
import numpy as np

class PixelCounts(Statistic):
    def __init__(self, bins=100, data_range=(-0.5,3.5)):
        self.bins = bins
        self.data_range = data_range
    def __call__(self, input):
        return density_series(np.exp(input)-1, bins=self.bins, data_range=self.data_range)

class PeakCounts(Statistic):
    def __init__(self, bins=100, data_range=(-2,2)):
        self.bins = bins
        self.data_range = data_range
    def __call__(self, input):
        input = np.exp(input) - 1
        peak_values = []
        for x in range(0, input.shape[0]):
            for y in range(0, input.shape[1]):
                surrounding = (
                    (x-1, y-1), (x+0, y-1), (x+1, y-1),
                    (x-1, y+0),             (x+1, y+0),
                    (x-1, y+1), (x+0, y+1), (x+1, y+1))
                peak = True
                for coord_pair in surrounding:
                    coord_pair = (coord_pair[0] % 128, coord_pair[1] % 128)
                    if input[x,y] <= input[coord_pair[0], coord_pair[1]]:
                        peak = False
                        break
                if peak:
                    peak_values.append(input[x,y])
        return density_series(peak_values, bins=self.bins, data_range=self.data_range)

class PowerSpectrumNBK(Statistic):
    def __init__(self, box_size=(1000,1000), kmin=1e-5, kmax=0.3, dk=None):
        self.box_size = box_size
        self.kmin = kmin
        self.kmax = kmax
        self.dk = dk
        from nbodykit.lab import FFTPower, ArrayMesh
        self._FFTPower = FFTPower
        self._ArrayMesh = ArrayMesh

    def __call__(self, field: np.ndarray):
        field_mesh = self._ArrayMesh(np.exp(field)-1, BoxSize=self.box_size)
        r_2d = self._FFTPower(field_mesh, mode='1d', kmin=self.kmin, kmax=self.kmax, dk=self.dk)
        return Series(
            x = np.real(r_2d.power['k']),
            y = np.real(r_2d.power['power'])
        )

class PowerSpectrumPB(Statistic):
    def __init__(self, box_size=(1000,1000), kmin=1e-5, kmax=0.3, dk=1e-2):
        self.box_size = box_size
        self.kmin = kmin
        self.kmax = kmax
        self.dk = dk
        from powerbox import get_power
        self._get_power = get_power
    def __call__(self, field: np.ndarray):
        spec, k = self._get_power(np.exp(field)-1, self.box_size, bins=np.arange(self.kmin, self.kmax, self.dk))
        return Series(
            x = k,
            y = spec
        )

def PowerSpectrum(*args, **kwargs):
    try: return PowerSpectrumNBK(*args, **kwargs)
    except ImportError: return PowerSpectrumPB(*args, **kwargs)