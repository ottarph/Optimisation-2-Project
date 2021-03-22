from fenics import *

T = 1.0            # final time
num_steps = 30     # number of time steps
delta_t = T / num_steps # time step size

L = 4
B = 2

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(L, B), int(L*10), int(B*10))
V = FunctionSpace(mesh, "Lagrange", 1)

tol = 1e-6

# Define variational problem
y = TrialFunction(V)
v = TestFunction(V)

g_const = 3
g = Expression("x[1] <=  tol & x[1] > B - tol ? g_const : 0",
               degree=0, tol=tol, B=B, g_const=g_const)

y_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
                 degree=2, a=5)
y_n = interpolate(y_0, V)

w = 5

rho = 1
c = 1

a = ( rho*c * y * v + delta_t * inner(grad(y), grad(v)) )*dx + ( delta_t * g * y * v )*ds
L = ( rho*c * y_n * v )*dx + ( delta_t * g * w * v )*ds


# Create PVD file for saving solution
pvdfile = File('rollout/solution.pvd')


# Time-stepping
y = Function(V)
y.assign(y_n)
t = 0
pvdfile << (y, t)
for n in range(num_steps):

    # Update current time
    t += delta_t

    # Compute solution
    solve(a == L, y)

    # Save to file and plot solution
    pvdfile << (y, t)

    # Update previous solution
    y_n.assign(y)



