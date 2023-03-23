import sympy as sp

t = sp.Symbol('t')

def get_cubic_polynomial_weights(start: int, end: int, time: int) -> list[float]:
    """
    start: theta0
    end: thetaf
    """
    a0: float = start
    a1: float = 0
    a2: float = 3 * (end - start) / time**2
    a3: float = -2 * (end - start) / time**3
    return [a0, a1, a2, a3]

def theta(time, weight: list[float]) -> float:
    return sp.N(weight[0] + weight[1] * time \
            + weight[2] * time**2 + weight[3] * time**3, 3)

# test = get_cubic_polynomial_weights(15, 75, 3)
# sp.pprint(theta(t, test))
