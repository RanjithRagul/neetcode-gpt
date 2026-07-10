import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:

        raw_score = np.dot(x, w) + b # z
        prediction = 1.0 / (1.0 + np.exp(-raw_score)) # sigmoid
        mistake = prediction - y_true # error
        how_easy_is_it_to_change = prediction * (1.0 - prediction) # sigmoid_derv
        final_correction = mistake * how_easy_is_it_to_change # delta
        weight_update = np.round(final_correction * x, 5) # dL_dW
        bias_update = round(float(final_correction), 5) # dL_dB
        return weight_update, bias_update