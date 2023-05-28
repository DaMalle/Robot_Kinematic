from kinematics import *
np.set_printoptions(precision=3, suppress=True)

##
## husk: bruges der grader eller radianer
## 

# write code here:
Trajectory_cubic_polynomials()
a = sp.Symbol('a')
sp.pprint(sp.diff(sp.diff(-2*a+10)))