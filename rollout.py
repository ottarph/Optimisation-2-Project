from fenics import *
from state_equation import *
from adjoint_equation import *
from cost_functional import *

T = 3.0                   # final time
num_steps = 50            # number of time steps
delta_t = T / num_steps   # time step size

L = 4
B = 2

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(L, B), int(L*10), int(B*10))
V = FunctionSpace(mesh, "Lagrange", 1)

""" Define new boundary-measure to split integral over Gamma_0, Gamma_1 """

#boundary_markers = FacetFunction("size_t", mesh)
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and ( near(x[1], 0, tol) or near(x[1], B, tol) )

bx0 = BoundaryX0()
bx0.mark(boundary_markers, 0)

class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and ( near(x[0], 0, tol) or near(x[0], L, tol) )

bx1 = BoundaryX1()
bx1.mark(boundary_markers, 1)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)



rho = 1
c = 100
k = 1

tol = 1e-6
g_const = 10
# Heat coefficient, function of time and space
g = Expression("x[1] <=  tol || x[1] > B - tol ? g_const : 0",
               degree=0, tol=tol, B=B, g_const=g_const, t=0)

g = Expression("g_const", degree=0, g_const=g_const, t=0)

y_0 = Expression("40 + 30*sin(pi*x[1]/B) + 10*cos(2*pi*x[0]/L)", degree=2, B=B, L=L)


# Control, function of time
W = []
w = Expression("t < 1.5 ? 0 : 100", degree=1, t=0)
for k in range(num_steps):
    w.t = k * delta_t
    w_k = interpolate(w, V)
    print(assemble(w_k * ds))
    W.append(w_k)


y_d_const = 10
y_d_func = Expression("y_d_const + 10*t", degree=0, y_d_const=y_d_const, t=0)

""" ----------------- State equation ----------------- """



Y, T = state(W, V, y_0, g, rho, c, k, delta_t, num_steps, ds)


# Create PVD file for saving solution
stateFile = File('rollout/state_equation.pvd')

y = Function(V)
for (y_n, t_n) in zip(Y, T):

    y.assign(y_n)

    stateFile << (y, t_n)

print(cost_functional(Y, W, T, y_d_func, delta_t, V))



""" ----------------- Adjoint equation ----------------- """


y_d_const = 10
y_d_func = Expression("y_d_const", degree=0, y_d_const=y_d_const)
y_d = interpolate(y_d_func, V)

P = adjoint(V, Y, T, y_d_func, g, rho, c, k, delta_t, num_steps, ds)

# Create PVD file for saving solution
adjFile = File('rollout/adjoint_equation.pvd')

# Save adjoint equation solution to file
p = Function(V)
for p_n, t_n in zip(P, T):

    # Save to file
    p.assign(p_n)

    adjFile << (p, t_n)


