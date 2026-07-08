class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # new = old - steps * slope
        # f(x) = x**2
        # f'(x) = 2x
        res = init
        for _ in range(iterations):
            res -= learning_rate * (2 * res)
        return round(res, 5)
