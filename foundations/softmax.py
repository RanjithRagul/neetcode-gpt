import numpy as np
from numpy.typing import NDArray

class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # my amplifier / sum of all amplifier
        # e power 1 = 2.7
        shifted = z - np.max(z)
        exps = np.exp(shifted)
        return np.round(exps/ np.sum(exps), 4)      
