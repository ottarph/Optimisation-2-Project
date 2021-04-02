from fenics import *

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

# y_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
#                 degree=2, a=5)
y_0 = Expression("100 + 20*sin(x[0]/L)", degree=2, L=L)
y_n = interpolate(y_0, V)

# Control
w = 5


a = ( rho*c * y * v + delta_t*k * inner(grad(y), grad(v)) )*dx + ( delta_t * g * y * v )*ds
L = ( rho*c * y_n * v )*dx + ( delta_t * g * w * v )*ds


# Create PVD file for saving solution
stateFile = File('rollout/state_equation.pvd')

Y = []
T = []

# Time-stepping
y = Function(V)
y.assign(y_n)
t = 0
stateFile << (y, t)
y0 = Function(V)
y0.assign(y)
Y.append(y0)
T.append(t)
for n in range(num_steps):

    # Update current time
    t += delta_t

    # Save time used for y-step
    T.append(t)

    # Compute solution
    solve(a == L, y)

    # Save to file
    #stateFile << (y, t)

    # Keep solution for use in adjoint equation
    #Y.append(y.copy())
    y_k = Function(V)
    y_k.assign(y)
    Y.append(y_k)


    # Update previous solution
    y_n.assign(y)

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


