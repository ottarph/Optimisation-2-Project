from fenics import *
from state_equation import *
from adjoint_equation import *

T = 1.0                   # final time
num_steps = 30            # number of time steps
delta_t = T / num_steps   # time step size

L = 4
B = 2

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(L, B), int(L*10), int(B*10))
V = FunctionSpace(mesh, "Lagrange", 1)


rho = 1
c = 1
k = 1

""" ----------------- State equation ----------------- """


tol = 1e-6
g_const = 3
# Heat coefficient, function of time and space
g = Expression("x[1] <=  tol || x[1] > B - tol ? g_const : 0",
               degree=0, tol=tol, B=B, g_const=g_const, t=0)

y_0 = Expression("100 + 20*sin(x[0]/L)", degree=2, L=L)


# Control, function of time
w = Expression("4 + 0.1*t", degree=1, t=0)


Y, T = state(w, V, y_0, g, rho, c, k, delta_t, num_steps)


# Create PVD file for saving solution
stateFile = File('rollout/state_equation.pvd')

y = Function(V)
for (y_n, t_n) in zip(Y, T):

    y.assign(y_n)

    stateFile << (y, t_n)



""" ----------------- Adjoint equation ----------------- """


y_d_const = 10
y_d_func = Expression("y_d_const", degree=0, y_d_const=y_d_const)
y_d = interpolate(y_d_func, V)

P = adjoint(V, Y, T, y_d, g, rho, c, k, delta_t, num_steps)

# Create PVD file for saving solution
adjFile = File('rollout/adjoint_equation.pvd')

# Save adjoint equation solution to file
p = Function(V)
for p_n, t_n in zip(P, T):

    # Save to file
    p.assign(p_n)

    adjFile << (p, t_n)


