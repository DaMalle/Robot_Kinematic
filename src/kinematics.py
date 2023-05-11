from dataclasses import dataclass
import sympy as sp
import numpy as np
import math


@dataclass
class Weights:
    a0: float
    a1: float
    a2: float
    a3: float


@dataclass
class Conditions:
    """Conditions uses radians for yaw, pitch and roll"""
    t: float
    theta0: float
    thetaf: float
    theta0_diff: float = 0
    thetaf_diff: float = 0
    time: float | None = None


class TragectoryPlanner:
    def __init__(self) -> None:
        pass

    def get_weights(self, segment: Conditions) -> Weights:
        """Returns weights for a cubic polynomial"""
        return Weights(
            a0=segment.theta0,
            a1=segment.theta0_diff,
            a2=(3 * (segment.thetaf - segment.theta0) / segment.t**2)
            - (2 * segment.theta0_diff / segment.t)
            - segment.thetaf_diff / segment.t,
            a3=(-2 * (segment.thetaf - segment.theta0) / segment.t**3)
            + (segment.thetaf_diff + segment.theta0_diff) / segment.t**2
        )

    def print_weights(self, segment: Conditions) -> None:
        """Prints weights for a cubic polynomial"""
        a0 = sp.N(segment.theta0, 3)
        a1 = sp.N(segment.theta0_diff, 3)
        a2 = sp.N((3 * (segment.thetaf - segment.theta0) / segment.t**2)
                  - (2 * segment.theta0_diff / segment.t)
                  - segment.thetaf_diff / segment.t, 3)
        a3 = sp.N((-2 * (segment.thetaf - segment.theta0) / segment.t**3)
                  + (segment.thetaf_diff
                  + segment.theta0_diff) / segment.t**2, 3)

        sp.pprint(f"a0: {a0} a1:{a1} a2: {a2} a3 {a3}")

    def print_theta(self, segment: Conditions) -> None:
        """Prints theta-angle for specific time or symbolic time"""
        a = self.get_weights(segment)
        time = segment.time if segment.time else sp.Symbol('t')
        s = sp.N(a.a0 + a.a1 * time + a.a2 * time**2 + a.a3 * time**3, 3)
        sp.pprint(s)

    def print_derivative_theta(self, segment: Conditions) -> None:
        """Prints derivative of theta-angle for specific time or symbolic time
        """
        a = self.get_weights(segment)
        time = segment.time if segment.time else sp.Symbol('t')
        sp.pprint(sp.N(a.a1 + 2 * a.a2 * time + 3 * a.a3 * time**2, 3))

    def print_double_derivative_theta(self, segment: Conditions) -> None:
        """Prints double derivative of theta-angle
        for specific time or symbolic time
        """
        a = self.get_weights(segment)
        time = segment.time if segment.time else sp.Symbol('t')
        sp.pprint(sp.N(2 * a.a2 + 6 * a.a3 * time, 3))


def is_rotation_matrix(matrix: np.ndarray) -> bool:
    """Returns true if numpy array is (close enough) to an identity matrix"""
    return np.allclose(matrix.T @ matrix, np.identity(3)) and np.allclose(np.linalg.det(matrix), 1)

def to_degrees(number: float) -> float:
    """Converts radians to degrees"""
    return number*180/math.pi

def to_radians(number: float) -> float:
    """Converts degrees to radians"""
    return number*math.pi/180

if __name__ == "__main__":
    Xs = 289.48
    Ys = 334.78
    Zs = 818.21
    Rs = 55.04
    Ps = -20.95
    Yaws = 139.63

    Xe = 561.11
    Ye = -71.56
    Ze = 1010.43
    Re = -8.28
    Pe = -55.20
    Yawe = 139.95

    """tf = (((Xe-Xs)**2 + (Ye - Ys)**2 + (Ze - Zs)**2)**0.5)/100
    t = TragectoryPlanner()
    # tf = sp.Symbol('tf')
    t.print_weights(Conditions(tf, Xs, Xe))
    t.print_weights(Conditions(tf, Ys, Ye))
    t.print_weights(Conditions(tf, Zs, Ze))
    t.print_weights(Conditions(tf, Rs*math.pi/180, Re*math.pi/180))
    t.print_weights(Conditions(tf, Ps*math.pi/180, Pe*math.pi/180))
    t.print_weights(Conditions(tf, Yaws*math.pi/180, Yawe*math.pi/180))"""

    print(is_rotation_matrix(np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])))
