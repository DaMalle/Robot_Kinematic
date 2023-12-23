import sympy as sp
# import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

#  Used to declare variable, which is dependent on another variable (fx time t)
#  t = sp.symbols("t")
#  theta = sp.Function("theta", real=True)(t)
# from sympy.physics.vector import dynamicsymbols

#  moment of inertia (2D):


def parallel_theorem(I_CoM, m, d):
    """
    d = distance from CoM to point of rotation
    I_CoM = moment of inertia at CoM
    m = mass
    """
    return I_CoM + m * d**2


def MOI_cylindrical_shell(m, r):
    """
    m = mass
    r = radius
    """
    return m * r**2


def MOI_cylinder(m, r):
    return (m * r**2)/2


def MOI_uniform_rod_end(m, L):
    """
    m = mass
    L = length of rod
    """
    return (m * L**2) / 3


def MOI_uniform_rod_center(m, L):
    """
    m = mass
    L = length of rod
    """
    return (m * L**2) / 12


def MOI_disk(m, r):
    return (m * r**2) / 2


def MOI_uniform_solid_sphere(m, r):
    return 2 * (m * r**2) / 5


def MOI_sphere_shell(m, r):
    return 2 * (m * r**2) / 3

# general functions


def rotX(angle: float, degrees: bool = True) -> sp.Matrix:
    if degrees:
        angle = to_radians(angle)

    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(angle), -sp.sin(angle)],
        [0, sp.sin(angle), sp.cos(angle)]
    ])


def rotY(angle: float, degrees: bool = True) -> sp.Matrix:
    if degrees:
        angle = to_radians(angle)

    return sp.Matrix([
        [sp.cos(angle), 0, sp.sin(angle)],
        [0, 1, 0],
        [-sp.sin(angle), 0, sp.cos(angle)]
    ])


def rotZ(angle: float, degrees: bool = True) -> sp.Matrix:
    if degrees:
        angle = to_radians(angle)
    return sp.Matrix([
        [sp.cos(angle), -sp.sin(angle), 0],
        [sp.sin(angle), sp.cos(angle), 0],
        [0, 0, 1]
    ])


def to_degrees(number: float) -> float:
    """Converts radians to degrees"""
    return number * 180/sp.pi


def to_radians(number: float) -> float:
    """Converts degrees to radians"""
    return number * sp.pi/180


def to_rotation_matrix(euler, seq="xyz", degrees: bool = False):
    """fixed-angles are extrinsic. fixed: xyz = euler zyx"""
    r = R.from_euler(seq, euler, degrees)
    return r.as_matrix()


# static analysis
# ## steps
# seperate into bodyparts.
# make free body diagrams for bodyparts without gravity
# setup equations asssuming the robot arm has to stand still

# tau for a rr-robot is equal to the transpose of the jacobian * force at the end-effector
# tau = j.t * f
# f = j.-t * tau

# sp.pprint(sp.Matrix([u, v, w]).jacobian(sp.Matrix([x, y, theta])))


#  #################### dynamic motion equation 2d ###################
#  Linear: CoM = center of mass
#  total_force = mass * acceleration_at_CoM
#  Rotation:
#  total_torque_at_CoM = moment_of_inertia_at_CoM * angular_acceleration

#  ## steps
#  first divide into separate bodyparts
#  insert values into the 2 equations from above
#  isolate for tau and F (remember that some variables are vectors)


def kinetic_energy(mass, velocity, moment_of_intertia, angular_velocity):
    """Can be used for translational, rotation and general motion
    if fixed axis rotation only: i is MOI about rotation point
    if general motion (both): i and v is about center of mass
    """
    i = moment_of_intertia
    omega = angular_velocity
    m = mass
    v = velocity
    translational = (m * v**2) / 2
    fixed_axis_rotational = (i * omega**2) / 2
    return translational + fixed_axis_rotational


def potential_energy_gravitational(mass, height_of_body_CoM):
    h = height_of_body_CoM
    m = mass
    g = -9.81  # not sure
    return m * g * h


def potential_energy_spring(k, x):
    return (k * x**2) / 2


def lagrangian(T_total, V_total):
    # T = kinetic_energy, V = potential_energy. Either total or per link
    # langrangian is the sum of the kinetic energy minus the potential energy
    L = T_total - V_total
    return L


def lagrangian_general_motion(L, theta, theta_diff, t):
    return sp.diff(sp.diff(L, theta_diff), t) - sp.diff(L, theta)


#  #################### recursive newtonian method ###################
# consists of outward and inward recursion:
# outward: calculates accelerations from base link to ending of last link
# inward: calculates force, start from last link to base link
# ## steps
# see functions below.


def get_angular_velocity(theta_i_diff, omega_before=sp.Matrix([0, 0, 0]), z_axis_i=sp.Matrix([0, 0, 1])):
    """angular velocity aka omega
    omega_i = omega_i-1 + theta_i_diff
    note: theta_i_diff is the relative angular velocity of link i with respect to link i-1
    """
    return omega_before + theta_i_diff * z_axis_i


def get_velocity(omega_before, s_before, v_before=sp.Matrix([0, 0, 0])):
    return v_before + omega_before.cross(s_before)


def get_angular_velocity_translational(omega_before=sp.Matrix([0, 0, 0])):
    return omega_before


def get_velocity_CoM_translational(v_before, omega_before, r_before_to_ci, d_diff):
    """
    v_before = velocity i-1
    omega_before = angular velocity i-1
    r_before_to_ci = position vector from i-1 to ci
    d_diff = translational velocity
    """
    return v_before + omega_before.cross(r_before_to_ci) + d_diff


def get_velocity_CoM(v_i, s_ci, omega_i):
    """
    v_i = velocity in current link
    s_ci = vector from start to CoM
    omega_i = angular_velocity
    """
    return v_i + omega_i.cross(s_ci)


def get_angular_acceleration(angular_acceleration_before, theta_i_diff_diff, z_axis_i=sp.Matrix([0, 0, 1])):
    return angular_acceleration_before + theta_i_diff_diff*z_axis_i


def get_acceleration_link(
    angular_acceleration_before,
    s_before,
    angular_velocity_before,
    acceleration_before=0,
):
    temp1 = sp.Matrix([0, 0, angular_acceleration_before]).cross(s_before)
    temp2 = sp.Matrix([0, 0, angular_velocity_before]).cross(s_before)
    return (
        acceleration_before
        + temp1
        + sp.Matrix([0, 0, angular_velocity_before]).cross(temp2)
    )


def get_acceleration_link_CoM(v_i, angular_acceleration, s_c_before, acceleration_before=0):
    return (
        acceleration_before
        + sp.Matrix([0, 0, angular_acceleration]).cross(s_c_before)
        + v_i.cross(v_i.cross(s_c_before))
    )


# inward recursion:


def get_force_i_CoM(mass_i, acceleration_CoM):
    return mass_i * acceleration_CoM


def get_torque_i_CoM(inertia_i_CoM, angular_acceleration_i_CoM):
    return inertia_i_CoM * angular_acceleration_i_CoM


def get_force(force_i_CoM, f_after):
    """Remember: if it is f1,2 which should be found, input f2,3"""
    return force_i_CoM + f_after


def get_torque_2d(s, f_after, torque_i_CoM, torque_after, force_i_CoM):
    """2D only, but 3d vectors"""
    s_CoM = s / 2
    return (
        torque_i_CoM
        + torque_after
        + s_CoM.cross(force_i_CoM)
        + s.cross(f_after)
    )


# 3d inertia tensor


def get_simpel_inertia_tensor(I_x, I_y, I_z):
    """The I's are the inertia around each axis"""
    return sp.Matrix([
        [I_x, 0, 0],
        [0, I_y, 0],
        [0, 0, I_z]
    ])


def get_global_inertia_tensor(local_inertia_tensor, rotation_matrix):
    return rotation_matrix @ local_inertia_tensor @ rotation_matrix.T


#  #################### dynamic motion equation 3d ###################
#  Linear: CoM = center of mass
#  total_force = mass * acceleration_at_CoM
#  Rotation:
#  total_torque_at_CoM = inertia_tensor * angular_acceleration + angular_velocity cross (inertia_tensor * angular_velocity)


def get_torque_3d(I, omega, omega_diff):
    return I @ omega_diff + omega.cross(I @ omega)

#  ## angular rate vs angular velocity: velocity is a vector and rate is value


#  multiple forces on a body


def n_total(f1, f2, f3, f4):
    """ can be more, can be less. I choose 4 to represent a drone"""
    L = 0.5
    s1: sp.Expr = sp.Matrix([L*sp.cos(0), L*sp.sin(0), 0])
    s2 = sp.Matrix([L*sp.cos(sp.pi/2), L*sp.sin(sp.pi/2), 0])
    s3 = sp.Matrix([L*sp.cos(sp.pi), L*sp.sin(sp.pi), 0])
    s4 = sp.Matrix([L*sp.cos(sp.pi*1.5), L*sp.sin(sp.pi*1.5), 0])

    n1 = s1.cross(f1)
    n2 = s2.cross(f2)
    n3 = s3.cross(f3)
    n4 = s4.cross(f4)
    return n1 + n2 + n3 + n4


def f_total(f1, f2, f3, f4, m):
    """ can be more, can be less. I choose 4 + gravity to represent a drone"""
    g = sp.Matrix([0, 0, -9.81])
    f5 = m*g
    return f1 + f2 + f3 + f4 + f5


def get_kinetic_energy_3d(m_i, v_ci, omega_i, I_i):
    """
    I_i = inertia tensor (3x3 matrix) for link i
    """
    temp1 = m_i * v_ci / 2
    temp2 = omega_i / 2
    return temp1.dot(v_ci) + temp2.dot(I_i @ omega_i)


def get_potential_energy_3d(m_i, r_ci):
    g = 9.81
    temp1 = m_i * g * sp.Matrix([0, 0, 1])
    return temp1.dot(r_ci)


def lagrangian_3d(total_potential, total_kinetic):
    """total meaning the sum of all the energies"""
    return total_kinetic - total_potential


# Parallel robots


def geometric_relations() -> None:
    """
    from point O in the middle to P at the end-effector.

    use f1, f2 to get algebraic solution (theta1, theta2)
    """
    # OP = OA + AB + BP
    # OP = OC + CD + DP

    # l3 = ||AP - AB||
    # l4 = ||CP - CD||

    # f1: (-d/2 + L1 * cos(theta1)-x)**2 + (L1 * sin(theta1)-y)**2 - L3**2 = 0
    # f2: (-d/2 + L2 * cos(theta2)-x)**2 + (L2 * sin(theta2)-y)**2 - L4**2 = 0
    return


def get_theta1_parallel_geometric(y, x, d, alpha1):
    theta1 = sp.atan(y/x+d/2) - alpha1
    return theta1


def get_theta2_parallel_geometric(y, x, d, alpha2):
    theta2 = sp.atan(y/x+d/2) - alpha2
    return theta2


def get_forward_parallel(theta1, theta2, L1, L2, L3, L4, d):
    """returns P(x, y)"""
    x, y = sp.symbols("x, y")
    eq1 = (-d/2 + L1 * sp.cos(theta1)-x)**2 + (L1 * sp.sin(theta1)-y)**2 - L3**2
    eq2 = (d/2 + L2 * sp.cos(theta2)-x)**2 + (L2 * sp.sin(theta2)-y)**2 - L4**2

    P = sp.solve([eq1, eq2], [x, y], dict=True)
    return P


def get_inverse_parallel(x, y, L1, L2, L3, L4, d):
    """returns theta1, theta2"""
    theta1, theta2 = sp.symbols("theta_1, theta_2")
    eq1 = (-d/2 + L1 * sp.cos(theta1)-x)**2 + (L1 * sp.sin(theta1)-y)**2 - L3**2
    eq2 = (d/2 + L2 * sp.cos(theta2)-x)**2 + (L2 * sp.sin(theta2)-y)**2 - L4**2

    angles = sp.solve([eq1, eq2], [theta1, theta2], dict=True)
    return angles


def get_j_a_parallel(x, y, L1, L2, theta1, theta2, d):
    J_a = sp.Matrix([
        [x+d/2-L1*sp.cos(theta1), y-L1*sp.sin(theta1)],
        [x-d/2-L2*sp.cos(theta2), y-L2*sp.sin(theta2)]
    ])
    return J_a


def get_j_b_parallel(x, y, L1, L2, theta1, theta2, d):
    jb1 = L1 * sp.sin(theta1)*x+((L1*sp.sin(theta1)*d)/2)-L1*sp.cos(theta1)*y
    jb2 = L2 * sp.sin(theta2)*x-((L2*sp.sin(theta2)*d)/2)-L2*sp.cos(theta2)*y
    J_b = sp.Matrix([
        [jb1, 0],
        [0, jb2]
    ])
    return J_b


def get_J_parallel(x, y, L1, L2, theta1, theta2, d):
    J_a = get_j_a_parallel(x, y, L1, L2, theta1, theta2, d)
    J_b = get_j_b_parallel(x, y, L1, L2, theta1, theta2, d)
    return -J_a.inv() @ J_b


def get_x_dot_parallel(x, y, L1, L2, theta1, theta2, d, theta1_dot, theta2_dot): # velocity analysis parallel
    J = get_J_parallel(x, y, L1, L2, theta1, theta2, d)
    theta_dot = sp.Matrix([theta1_dot, theta2_dot])
    x_dot = J @ theta_dot
    return x_dot


def get_passive_joints_angular_velocity_parallel(L1, L2, L3, L4, theta1, theta1_dot, theta2_dot, phi1, phi2):
    """
    phi1 = theta1 - theta3
    phi2 = theta2 + theta4
    phi_dot = angular_velocities
    """
    theta_dot = sp.Matrix([theta1_dot, theta2_dot])
    J_a = sp.Matrix([
        [-L1 * sp.sin(theta1), L2 * sp.sin(theta1)],
        [L1 * sp.cos(theta1), -L2 * sp.cos(theta1)]
    ])
    J_p = sp.Matrix([
        [-L3 * sp.sin(phi1), L4 * sp.sin(phi2)],
        [L3 * sp.cos(phi1), -L4 * sp.cos(phi2)]
    ])
    angular_velocities = - J_p.inv() @ J_a @ theta_dot
    return angular_velocities


def get_velocitites_moving_links_parallel(x_dot, phi1_dot, phi2_dot, s_c3, s_c4):
    v_c3 = x_dot + sp.Matrix([0, 0, phi1_dot]).cross(s_c3)
    v_c4 = x_dot + sp.Matrix([0, 0, phi2_dot]).cross(s_c4)
    return v_c3, v_c4


def get_equation_motion_parallel():
    """lagrangian with only kinetic energy"""
    return L


def get_theta3_and_4(x, y, L1, L2, L3, L4, theta1, theta2, d):
    theta3, theta4 = sp.symbols("theta_3, theta_4")
    OC = sp.Matrix([d/2, 0])
    OA = sp.Matrix([-d/2, 0])
    AB = sp.Matrix([L1*sp.cos(theta1), L1*sp.sin(theta1)])
    CD = sp.Matrix([L2*sp.cos(theta2), L2*sp.sin(theta2)])
    OP = sp.Matrix([x, y])

    # theta 4:
    x1, y1 = OP - CD - OC
    phi2 = sp.N(to_degrees(sp.atan2(y1, x1)))
    theta4 = phi2 - theta2

    # theta 3
    x2, y2 = OP - OA - AB
    phi1 = sp.N(to_degrees(sp.atan2(y2, x2)))
    theta3 = theta1 - phi1

    return theta3, theta4


def forward_singularities_parallel(theta1=0, theta2=0, theta3=0, theta4=0):
    """theta2 + theta4 - theta1 - theta3 = 0, pi"""
    phi1 = theta1 + theta3
    phi2 = theta2 + theta4
    value = phi2 - phi1
    return value == 0 or value == sp.pi


def inverse_singularities_parallel(theta3, theta4):
    """
    theta3 = 0, pi
    or
    theta4 = 0, pi
    """
    return theta3 == 0 or theta3 == sp.pi or theta4 == 0 or theta4 == sp.pi


if __name__ == "__main__":
    # run code here
    # r = to_rotation_matrix(sp.Matrix([0, 0, 0]))
    # local = sp.Matrix([-1.1, -0.35, 0])
    # print(get_global_inertia_tensor(local, r))
    # I = sp.Matrix([
    #     [5.864, 0, -12.806],
    #     [0, 109.262, 0],
    #     [-12.806, 0, 110.161]
    # ])
    # omega = sp.Matrix([0, 0, 6])
    # omega_diff = sp.Matrix([0, 0, 0])
    # print(get_torque_3d(I, omega, omega_diff))
    # omega = sp.Matrix([3, 1, 2])
    # print(get_torque_3d(I, omega, omega_diff))
    # omega_diff = sp.Matrix([0, 0, 3])
    # print(get_torque_3d(I, omega, omega_diff))

    # R1 = rotZ(120)
    # R2 = R1 @ rotX(90) @ rotZ(30)
    # omega1 = get_angular_velocity(0, sp.Matrix([0, 0, 1.2]))
    # omega2 = R2 @ sp.Matrix([0, 0, 3]) + omega1
    # s2 = R2 @ sp.Matrix([3, 0, 0])
    # s1 = sp.Matrix([0, 0, 2])
    #
    #
    # # print(omega2)
    # v2 = sp.Matrix([0, 0, 0])
    # v_c2 = get_velocity_CoM(v2, s2/2, omega2)
    # print(s2/2)
    # print(v_c2)
    m1 = 10
    m2 = 6
    l1 = 3
    l2 = 3
    I1 = get_simpel_inertia_tensor(6, 6, 4)
    I2 = get_simpel_inertia_tensor(2, 3, 3)

    t = sp.symbols("t")
    theta1 = sp.Function("theta_1", real=True)(t)
    theta2 = sp.Function("theta_2", real=True)(t)
    theta1_diff = theta1.diff(t)
    theta2_diff = theta2.diff(t)
    R1 = rotZ(theta1, False)
    R2 = R1 @ rotX(sp.pi/2, False) @ rotZ(theta2, False)

    s1 = sp.Matrix([0, 0, l1])
    s_c1 = s1/2
    s2 = R2 @ sp.Matrix([3, 0, 0])
    s_c2 = s2/2

    r_c1 = s_c1
    r_c2 = s_c2  # from base to s_c2

    I1_g = get_global_inertia_tensor(I1, R1)
    I2_g = get_global_inertia_tensor(I2, R2)

    omega1 = sp.Matrix([0, 0, theta1_diff])
    v1 = sp.Matrix([0, 0, 0])
    v_c1 = sp.Matrix([0, 0, 0])
    K1 = get_kinetic_energy_3d(m1, v_c1, omega1, I1_g)
    P1 = get_potential_energy_3d(m1, r_c1)

    z1 = sp.Matrix([0, 0, 1])
    z2 = R2 @ z1
    omega2 = get_angular_velocity(theta2_diff, omega1, z2)
    v2 = get_velocity(omega1, s1)
    v_c2 = get_velocity_CoM(v2, s_c2, omega2)

    P2 = get_potential_energy_3d(m2, r_c2)
    K2 = get_kinetic_energy_3d(m2, v_c2, omega2, I2_g)
    L = lagrangian_3d(P1 + P2, K1 + K2)
    tau1 = lagrangian_general_motion(L, theta1, theta1_diff, t)
    tau2 = lagrangian_general_motion(L, theta2, theta2_diff, t)

    t1 = 5*sp.cos(sp.pi*t)
    t2 = 3*sp.cos(2*sp.pi*t)
    t1d = t1.diff(t)
    t2d = t2.diff(t)
    t1dd = t1.diff(t, t)
    t2dd = t2.diff(t, t)

    tau1 = tau1.subs([(theta1.diff(t,t), t1dd), (theta2.diff(t,t), t2dd), (theta1.diff(t), t1d), (theta2.diff(t), t2d), (theta1, t1), (theta2, t2)])
    tau2 = tau2.subs([(theta1.diff(t,t), t1dd), (theta2.diff(t,t), t2dd), (theta1.diff(t), t1d), (theta2.diff(t), t2d), (theta1, t1), (theta2, t2)])
    sp.pprint(sp.N(tau1.subs(t, 0.2)))
    sp.pprint(sp.N(tau2.subs(t, 0.2)))
    # k = sp.symbols("k")
    # sp.pprint(sp.solve(sp.cos(k), 1))
    #
    # x, y, z = sp.symbols("x, y, z")
    # u = x**2 + y - z
    # v = x*2+y**4+z**2
    # w = x*2+y**4+z**2
    # sp.pprint(sp.Matrix([u, v, w]).jacobian([x, y, z]))
    # sp.pprint(get_inverse_parallel(0.1, 0.3, 0.3, 0.3, 0.35, 0.35, 0.3))
    # P_total = get_forward_parallel(theta1=0.8, theta2=0.5, L1=0.3, L2=0.3, L3=0.35, L4=0.35, d=0.3)
    # P1 = P_total[0]
    # P2 = P_total[1]
    # x, y = P1.values()
    # sp.pprint(get_x_dot_parallel(x, y, 0.3, 0.3, 0.8, 0.5, 0.3, 1.8, 1.5))
    # x, y = P2.values()
    # sp.pprint(get_x_dot_parallel(x, y, 0.3, 0.3, 0.8, 0.5, 0.3, 1.8, 1.5))
    # P_total = get_forward_parallel(theta1=2.43, theta2=0.62, L1=0.3, L2=0.3, L3=0.35, L4=0.35, d=0.1)
    # sp.pprint(sp.N(to_degrees(2.43)))
    # sp.pprint(sp.N(to_degrees(0.62)))
    # x, y = P_total[1].values()
    # sp.pprint(P_total[1])
    # get_theta3_and_4(x, y, L1=0.3, L2=0.3, L3=0.35, L4=0.35, theta1=2.43, theta2=0.62, d=0.1)
