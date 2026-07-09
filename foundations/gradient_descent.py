class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # new = old - (steps * slope)
        # f(x) = x**2
        # f'(x) = 2x
        grad = init
        for _ in range(iterations):
            derive = 2 * grad
            grad = grad - learning_rate * derive
        return round(grad, 5)