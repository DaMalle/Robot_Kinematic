import sympy as sp

class Tragectory_planner:
    def __init__(self) -> None:
        pass

    def get_weights(self, theta0: float, thetaf: float, time, theta0_diff: float = 0, thetaf_diff: float = 0) -> list[float]:
        """Weights for a cubic polynomial"""
        a0: float = theta0
        a1: float = theta0_diff
        a2: float = (3 * (thetaf - theta0) / time**2) - (2 * theta0_diff / time) - thetaf_diff / time
        a3: float = (-2 * (thetaf - theta0) / time**3) + (thetaf_diff + theta0_diff) / time**2

        return [a0, a1, a2, a3]

    def print_theta(self, theta0: float, thetaf: float, time, theta0_diff: float = 0, thetaf_diff: float = 0) -> None:
        weight = self.get_weights(theta0, thetaf, time, theta0_diff, thetaf_diff)
        time = sp.Symbol('t')
        sp.pprint(sp.N(weight[0] + weight[1] * time + weight[2] * time**2 + weight[3] * time**3, 3))

    def print_derivative_theta(self, theta0: float, thetaf: float, time, theta0_diff: float = 0, thetaf_diff: float = 0) -> None:
        weight = self.get_weights(theta0, thetaf, time, theta0_diff, thetaf_diff)
        time = sp.Symbol('t')
        sp.pprint(sp.N(weight[1] + 2 * weight[2] * time + 3 * weight[3] * time**2, 3))


if __name__ == "__main__":
    t = Tragectory_planner()
    t.print_theta(15, 75, 3, 0, 10)
    t.print_derivative_theta(15, 75, 3, 0, 0)
