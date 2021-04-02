from fenics import *
from state_equation import *

T = 1.0            # final time
num_steps = 30     # number of time steps
delta_t = T / num_steps # time step size

L = 4
B = 2

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(L, B), int(L*10), int(B*10))
V = FunctionSpace(mesh, "Lagrange", 1)


rho = 1
c = 1
k = 1

""" ----------------- State equation ----------------- """

# Define variational problem
y = TrialFunction(V)
v = TestFunction(V)

tol = 1e-6
g_const = 3
g = Expression("x[1] <=  tol || x[1] > B - tol ? g_const : 0",
               degree=0, tol=tol, B=B, g_const=g_const)

y_0 = Expression("100 + 20*sin(x[0]/L)", degree=2, L=L)


# Control
w = 5


Y, T = state(w, V, y_0, g, rho, c, k, delta_t, num_steps)


# Create PVD file for saving solution
stateFile = File('rollout/state_equation.pvd')

y = Function(V)
for (y_n, t_n) in zip(Y, T):

    y.assign(y_n)

    stateFile << (y, t_n)



""" ----------------- Adjoint equation ----------------- """

p = TrialFunction(V)
h = TestFunction(V)

y_d_const = 10
y_d_0 = Expression("y_d_const", degree=0, y_d_const=y_d_const)
y_d = interpolate(y_d_0, V)

y = Function(V)

p_0 = Expression('0', degree=0)
p_n = interpolate(p_0, V)


a = ( rho*c * p * h + delta_t*k * inner(grad(p), grad(h)) )*dx + ( delta_t * g * p * h )*ds
L = ( rho*c * p_n * h + delta_t * (y - y_d) * h )*dx

# Create PVD file for saving solution
adjFile = File('rollout/adjoint_equation.pvd')

P = []

# Time-stepping
p = Function(V)
p.assign(p_n)
t = 0
p0 = Function(V)
p0.assign(p)
P.append(p0)
for n in range(num_steps):

    # Update current time
    t += delta_t

    y.assign(Y[-n])

    # Compute solution
    solve(a == L, p)

    # Keep solution to reverse later
    p_k = Function(V)
    p_k.assign(p)
    P.append(p_k)

    # Update previous solution
    p_n.assign(p)


# Reverse adjoint equation in time to get true adjoint
for p_n, t_n in zip(P[::-1], T):

    # Save to file
    p.assign(p_n)

    adjFile << (p, t_n)


