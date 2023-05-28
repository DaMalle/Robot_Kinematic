from dataclasses import dataclass
import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv


@dataclass
class Weights:
    a0: float
    a1: float
    a2: float
    a3: float


@dataclass
class Conditions:
    """Conditions uses radians for yaw, pitch and roll"""
    t: float                  # endtime or tf
    theta0: float             # start angle for joint
    thetaf: float             # end angle for joint
    theta0_diff: float = 0    # start velocity
    thetaf_diff: float = 0    # end velocity
    time: float | None = None # used to define for specifik time


class Trajectory_cubic_polynomials:
    """Trajectory generation with cubic poly nomials WITHOUT any viapoints
    Used to make ptp/point-to-point/jmove/jointmove.
    not linear
    """
    def __init__(self) -> None:
        pass

    def get_weights(self, segment: Conditions) -> Weights:
        """Returns weights for a cubic polynomial"""
        return Weights(
            a0=segment.theta0,
            a1=segment.theta0_diff,
            a2=(3 * (segment.thetaf - segment.theta0) / segment.t**2) - (2 * segment.theta0_diff / segment.t) - segment.thetaf_diff / segment.t,
            a3=(-2 * (segment.thetaf - segment.theta0) / segment.t**3) + (segment.thetaf_diff + segment.theta0_diff) / segment.t**2
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

    def graph_theta(self, segment: Conditions) -> None:
        a = self.get_weights(segment)
        x = np.linspace(0, segment.t, 1000, endpoint=True)
        y = a.a0 + a.a1 * x + a.a2 * x**2 + a.a3 * x**3
        plt.plot(x, y)
        plt.show()

    def graph_derivative_theta(self, segment: Conditions) -> None:
        a = self.get_weights(segment)
        x = np.linspace(0, segment.t, 1000, endpoint=True)
        y = a.a1 + 2 * a.a2 * x + 3 * a.a3 * x**2
        plt.plot(x, y)
        plt.show()

    def graph_double_derivative_theta(self, segment: Conditions) -> None:
        a = self.get_weights(segment)
        x = np.linspace(0, segment.t, 1000, endpoint=True)
        y = 2 * a.a2 + 6 * a.a3 * x
        plt.plot(x, y)
        plt.show()

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

class Trajectory_parabolic_blends:
    def __init__(self, theta0: float, thetaf: float, tf: float, acceleration: float) -> None:
        self.a = acceleration
        self.theta0 = theta0
        self.thetaf = thetaf
        self.tf = tf
        self.tb = (tf/2) - math.sqrt((self.a**2) * (tf**2) - 4 * self.a * (thetaf - theta0)) / (2 * self.a)

    def is_valid_acceleration(self) -> bool:
        return self.a >= 4 * (self.thetaf - self.theta0) / self.tf**2
    
    def get_blendtime(self) -> float:
        return self.tb

    def get_theta(self, t: float) -> float:
        """Computes a trapezoidal trajectory, which has a linear motion segment with
        parabolic blends. Then prints it"""
        if t > self.tf - self.tb:
            return self.thetaf - 0.5 * self.a * ((self.tf-t)**2)
        if t > self.tb:
            return (0.5 * self.a * (self.tb**2) + self.theta0) + self.a * self.tb * (t-self.tb)
        return self.theta0 + 0.5 * self.a * (t**2)
    
    def graph_theta(self) -> None:
        x = np.linspace(0, self.tf, 1000, endpoint=True)
        y = [self.get_theta(t) for t in x]
        plt.plot(x, y)
        plt.show()

def T_mdh(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Returns transformation matrix for modified denavit-hartenberg"""
    s = math.sin
    c = math.cos
    return np.array([
        [c(theta), -s(theta), 0, a],
        [s(theta) * c(alpha), c(theta) * c(alpha), -s(alpha), -s(alpha) * d],
        [s(theta) * s(alpha), c(theta) * s(alpha), c(alpha), c(alpha) * d],
        [0, 0, 0, 1]
    ])

def rotX(angle: float, degrees: bool = True) -> np.ndarray:
    if degrees: angle = to_radians(angle)
    return np.array([
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)]
    ])

def rotY(angle: float, degrees: bool = True) -> np.ndarray:
    if degrees: angle = to_radians(angle)
    return np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)]
    ])

def rotZ(angle: float, degrees: bool = True) -> np.ndarray:
    if degrees: angle = to_radians(angle)
    return np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])

def is_rotation_matrix(matrix: np.ndarray) -> bool:
    """Returns true if numpy array is (close enough) to an identity matrix"""
    return np.allclose(matrix.T @ matrix, np.identity(3)) and np.allclose(np.linalg.det(matrix), 1)

def to_degrees(number: float) -> float:
    """Converts radians to degrees"""
    return number * 180/math.pi

def to_radians(number: float) -> float:
    """Converts degrees to radians"""
    return number * math.pi/180

def to_euler_angles(rotation_matrix: np.ndarray, Conventions: str, degrees: bool = True) -> np.ndarray:
    """
    euler angles are intrinsic whereas fixed-angles are extrinsic
    example of converntions: zyx, xyx, zyz etc (all combinations of x y z = 12)
    """
    r = R.from_matrix(rotation_matrix)
    return r.as_euler(Conventions.upper(), degrees=degrees)

def to_fixed_angless(rotation_matrix: np.ndarray, Conventions: str, degrees: bool = True) -> np.ndarray:
    """fixed-angles are extrinsic. fixed: xyz = euler zyx"""
    r = R.from_matrix(rotation_matrix)
    return r.as_euler(Conventions.lower(), degrees=degrees)

def to_angle_axis(rotation_matrix: np.ndarray) -> np.ndarray:
    """rotvec is the same as angle-axis
    to go the other way use
    """
    r = R.from_matrix(rotation_matrix)
    return r.as_rotvec()

def to_quaternions(rotation_matrix: np.ndarray) -> np.ndarray:
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()


# jacobians

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    
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

    #tf = (((Xe-Xs)**2 + (Ye - Ys)**2 + (Ze - Zs)**2)**0.5)/100
    #weigths = Trajectory_cubic_polynomials().graph_double_derivative_theta(Conditions(1, 15, 75))
    # tf = sp.Symbol('tf')
    """t.print_weights(Conditions(tf, Xs, Xe))
    t.print_weights(Conditions(tf, Ys, Ye))
    t.print_weights(Conditions(tf, Zs, Ze))
    t.print_weights(Conditions(tf, Rs*math.pi/180, Re*math.pi/180))
    t.print_weights(Conditions(tf, Ps*math.pi/180, Pe*math.pi/180))
    t.print_weights(Conditions(tf, Yaws*math.pi/180, Yawe*math.pi/180))"""

    """print(is_rotation_matrix(np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])))"""
    para = Trajectory_parabolic_blends(15, 75, 10, 3)
    print(para.get_blendtime())
    print(para.get_theta(0))
    #para.graph_theta()

    # from euler to rotation matrix. upper-case for intrinsic and lower for extrinsic. angles given in the order of the string
    #                          z    y   x
    eul = R.from_euler('ZYX', [30, -30, 45], degrees=True)
    #                                 x    y   z
    fixed_axis = R.from_euler('xyz', [45, -30, 30], degrees=True)
    #print(fixed_axis.as_matrix())
    #print(eul.as_matrix())

    # example of cubic polyunomial with viapoint. total time is 8 seconds (3 and 5) and with velocity of 10 in the viapoint
    Trajectory_cubic_polynomials().print_theta(Conditions(3, 15, 75, 0, 10))
    Trajectory_cubic_polynomials().print_theta(Conditions(5, 75, 35, 10, 0))


    ############################################################################
    Trajectory_cubic_polynomials().graph_theta(Conditions(1, 25, 15, 0, -30))

    print(Trajectory_parabolic_blends(-50, 140, 8, 20).get_theta(0.5))
    Trajectory_parabolic_blends(-50, 140, 8, 20).graph_theta()
