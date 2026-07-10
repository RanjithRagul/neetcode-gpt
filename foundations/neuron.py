import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        res = z = np.matmul(x, w) + b
        if activation == "sigmoid":
            res = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            res = max(0, z)
        return round(float(res), 5)