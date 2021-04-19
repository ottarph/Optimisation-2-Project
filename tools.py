from fenics import *
from state_equation import *

def create_boundary_measure(mesh, L, B):
    """ Define new boundary-measure to split integral over Gamma_0, Gamma_1 """

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

    return ds


def create_test_problem(w_func, V, y_0, g, rho, c, k, delta_t, num_steps, ds):

    W = []
    t = 0
    for i in range(num_steps):
        w_func.t = t
        wi = interpolate(w_func, V)
        W.append(wi)
        t += delta_t

    Y, T = state(W, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

    return Y[-1]
