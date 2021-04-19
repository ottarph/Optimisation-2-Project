from fenics import *
from state_equation import *
from adjoint_equation import *
from cost_functional import *
from tools import *
import numpy as np


def gradient_descent(W_0, V, y_0, y_d_func, w_a, w_b, gamma, g, rho, c, k, delta_t, num_steps, ds,
         stop, max_iter, rel_stop=0.95, c1 = 0.5, tau=0.5, max_inner_iter=10):

    costs = []

    W = W_0
    Y, T = state(W, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

    last_cost = np.inf
    curr_cost = cost_functional(Y, W, T, y_d_func, delta_t, V, gamma)

    i = 0
    while curr_cost > stop and i < max_iter and curr_cost / last_cost < rel_stop:

        costs.append(curr_cost)

        i += 1
        print(f'step number {i}')
        print(f'cost = {curr_cost}')
        print()

        P = adjoint_eq(V, Y, T, y_d_func, g, rho, c, k, delta_t, num_steps, ds)

        grad = []
        for n, (t, p, w) in enumerate(zip(T, P, W)):
            g.t = t
            gg = interpolate(g, V)
            gn = Function(V)
            gn.vector()[:] = gg.vector()[:] * p.vector()[:] + gamma * w.vector()[:]
            grad.append(gn)

        grad_square_norm = 0

        for gn in grad:
            grad_square_norm += delta_t * norm(gn)**2


        """ Backtracking line search with Armijo condition """
        alpha = 1
        accept = False
        j = 0
        while accept is False and j < max_inner_iter:
            j += 1

            W_new = []
            for w, g in zip(W, grad):
                w_new = Function(V)
                w_new.vector()[:] = w.vector()[:] - alpha*g.vector()[:]
                W_new.append(w_new)

            Y_new, _ = state(W_new, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

            new_cost = cost_functional(Y_new, W_new, T, y_d_func, delta_t, V, gamma)

            if new_cost <= curr_cost + c1 * alpha * grad_square_norm:
                accept = True
            else:
                alpha *= tau

        print(f'j = {j}')

        """ Project onto admissible set """

        W_new_ad = []
        for w in W_new:
            w_ad = Function(V)
            w_ad.vector()[:] = np.maximum(w_a, np.minimum(w_b, w.vector()[:]))
            W_new_ad.append(w_ad)

        """ Compute solution and cost for new admissible control """

        Y_new_ad, _ = state(W_new_ad, V, y_0, g, rho, c, k, delta_t, num_steps, ds)
        new_cost_ad = cost_functional(Y_new_ad, W_new_ad, T, y_d_func, delta_t, V, gamma)

        """ Update loop variables """

        if new_cost_ad < curr_cost:
                
            last_cost = curr_cost
            curr_cost = new_cost_ad
            Y = Y_new_ad
            W = W_new_ad

    return W, costs


def main():

    """ Define domain and space """

    T = 3.0                   # final time
    num_steps = 10            # number of time steps
    delta_t = T / num_steps   # time step size

    L = 4
    B = 2

    fineness = 5

    # Create mesh and define function space
    mesh = RectangleMesh(Point(0, 0), Point(L, B), int(L*fineness), int(B*fineness))
    V = FunctionSpace(mesh, "Lagrange", 1)

    """ Define new boundary-measure to split integral over Gamma_0, Gamma_1 """
    ds = create_boundary_measure(mesh, L, B)
    
    """ Defining the cost functional """

    # Penalty constant for control
    gamma = 1

    # Target temperature function over time
    y_d_const = 30
    y_d_func = Expression("y_d_const + 5*t", degree=0, y_d_const=y_d_const, t=0)


    # Physical constants in equation
    rho = 1
    c = 100
    k = 1

    g_const = 10
    # Heat coefficient, function of time
    g = Expression("g_const", degree=0, g_const=g_const, t=0)

    # Initial condition
    y_0 = Expression("40 + 30*sin(pi*x[1]/B) + 10*cos(2*pi*x[0]/L)", degree=2, B=B, L=L)


    """ Define admissible set """

    w_a = 4  # Think of as 04 degrees Celsius water
    w_b = 90 # Think of as 90 degrees Celsius water


    # Control over boundary, assumed function of time
    W_0 = []
    w = Expression("t < 1.5 ? 4 : 90", degree=1, t=0)
    for i in range(num_steps):
        w.t = i * delta_t
        w_i = interpolate(w, V)
        W_0.append(w_i)

    stop = 0 # Don't stop because of low cost functional.
    max_iter = 10
    max_inner_iter = 10 # How many iterations to find a suitable step length

    W, costs = gradient_descent(W_0, V, y_0, y_d_func, w_a, w_b, gamma, g, rho, c, k, delta_t, num_steps, ds,
         stop, max_iter, rel_stop=0.95, c1 = 0.5, tau=0.5, max_inner_iter=max_inner_iter)

    fname = "temp.txt"
    save = True
    if save:
        np.savetxt(fname, costs)


    return


if __name__ == '__main__':
    main()
