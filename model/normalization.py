import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        ep   = 1e-5
        mean = np.mean(x)
        var  = np.var(x)
        x_norm = (x - mean) / np.sqrt(var + ep)
        res  = x_norm * gamma + beta
        return np.round(res, 5)

