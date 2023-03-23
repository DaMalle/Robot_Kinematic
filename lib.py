from dataclasses import dataclass
import sympy as sp


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
    time: float | None = None


class Tragectory_planner:
    def __init__(self) -> None:
        pass

    def get_weights(self, segment: Conditions) -> Weights:
        """Weights for a cubic polynomial"""
        return Weights(
            a0=segment.theta0,
            a1=segment.theta0_diff,
            a2=(3 * (segment.thetaf - segment.theta0) / segment.t**2) \
                - (2 * segment.theta0_diff / segment.t) \
                - segment.thetaf_diff / segment.t,
            a3=(-2 * (segment.thetaf - segment.theta0) / segment.t**3) \
                + (segment.thetaf_diff + segment.theta0_diff) / segment.t**2
        )

    def print_theta(self, segment: Conditions) -> None:
        a = self.get_weights(segment)
        if segment.time == None: time = sp.Symbol('t') 
        else: time = segment.time
        sp.pprint(sp.N(a.a0 + a.a1 * time + a.a2 * time**2 + a.a3 * time**3, 3))

    def print_derivative_theta(self, segment: Conditions) -> None:
        a = self.get_weights(segment)
        if segment.time == None: time = sp.Symbol('t') 
        else: time = segment.time
        sp.pprint(sp.N(a.a1 + 2 * a.a2 * time + 3 * a.a3 * time**2, 3))

    def print_double_derivative_theta(self, segment: Conditions) -> None:
        a = self.get_weights(segment)
        if segment.time == None: time = sp.Symbol('t') 
        else: time = segment.time
        sp.pprint(sp.N(2 * a.a2 + 6 * a.a3 * time, 3))

if __name__ == "__main__":
    t = Tragectory_planner()
    tf = sp.Symbol('tf')
    t.print_theta(Conditions(5, 15, 1, 0, 15.5, 0.876))
    t.print_theta(Conditions(15.5, 40, 1, 15.5, 0))
