from dataclasses import dataclass
from enum import Enum, auto, unique
import sympy as sp


@unique
class Theta(Enum):
    base = auto()
    derivative = auto()
    double_derivative = auto()


@dataclass
class Weights:
    a0: float
    a1: float
    a2: float
    a3: float


@dataclass
class Conditions:
    t: float
    theta0: float
    thetaf: float
    theta0_diff: float = 0
    thetaf_diff: float = 0


class Tragectory_planner:
    def __init__(self, conditions: Conditions):
        self.conditions = conditions
        self.weights = self.get_weights(conditions)

    def get_weights(self, conditions: Conditions) -> Weights:
        """Returns weights for a cubic polynomial"""
        return Weights(
            a0=conditions.theta0,
            a1=conditions.theta0_diff,
            a2=(3 * (conditions.thetaf
                - conditions.theta0) / conditions.t**2)
            - (2 * conditions.theta0_diff / conditions.t)
            - conditions.thetaf_diff / conditions.t,
            a3=(-2 * (conditions.thetaf
                - conditions.theta0) / conditions.t**3)
            + (conditions.thetaf_diff
                + conditions.theta0_diff) / conditions.t**2
        )

    def print_theta(self, theta: Theta, time: float | None = None) -> None:
        """Prints theta-angle for specific time or symbolic time"""
        a = self.weights
        tf = time if time else sp.Symbol('t')

        if theta is Theta.double_derivative:
            sp.pprint(sp.N(2 * a.a2 + 6 * a.a3 * tf, 3))
        elif theta is Theta.derivative:
            sp.pprint(sp.N(a.a1 + 2 * a.a2 * tf + 3 * a.a3 * tf**2, 3))
        else:
            sp.pprint(sp.N(a.a0 + a.a1 * tf + a.a2 * tf**2 + a.a3 * tf**3, 3))


if __name__ == "__main__":
    t = Tragectory_planner(Conditions(5, 15, 1, 0, 15.5))
