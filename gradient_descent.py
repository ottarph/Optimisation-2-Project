from fenics import *
from state_equation import *
from adjoint_equation import *
from cost_functional import *
from tools import *
import numpy as np


def gradient_descent(W_0, V, y_0, y_d, w_a, w_b, gamma, g, rho, c, k, delta_t, num_steps, ds,
         stop, max_iter, rel_stop=0.95, c1=0.5, alpha_0=1, tau=0.5, max_inner_iter=10):

    costs = []

    W = W_0
    Y, T = state(W, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

    last_cost = np.inf
    curr_cost = cost_functional(Y, W, T, y_d, delta_t, V, gamma)

    i = 0
    while curr_cost > stop and i < max_iter and curr_cost / last_cost < rel_stop:

        costs.append(curr_cost)

        i += 1
        print()
        print(f'step number {i}')
        print(f'cost = {curr_cost}')


        P = adjoint_eq(V, Y, T, y_d, g, rho, c, k, delta_t, num_steps, ds)

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
        alpha = alpha_0
        accept = False
        j = 0
        while accept is False and j < max_inner_iter:
            j += 1

            W_new = []
            for w, gn in zip(W, grad):
                w_new = Function(V)
                w_new.vector()[:] = w.vector()[:] - alpha*gn.vector()[:]
                W_new.append(w_new)

            Y_new, _ = state(W_new, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

            new_cost = cost_functional(Y_new, W_new, T, y_d, delta_t, V, gamma)

            if new_cost <= curr_cost - c1 * alpha * grad_square_norm:
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
        new_cost_ad = cost_functional(Y_new_ad, W_new_ad, T, y_d, delta_t, V, gamma)

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

    """ Properties of system """
    # Physical constants in equation
    rho = 1
    c = 100
    k = 50

    g_const = 15
    # Heat coefficient, function of time
    g = Expression("g_const * exp(-pow(t - T/2, 2))", degree=2, g_const=g_const, T=T, t=0)

    # Initial condition
    y_0 = Expression("200 + 50*exp(-pow(x[0]-L/2, 2) - pow(x[1]-B/2, 2))", degree=2, L=L, B=B)
    

    """ Defining the cost functional """

    # Penalty constant for control
    gamma = 0.001

    # Target temperature at end time
    y_d_const = 30
    y_d_func = Expression("y_d_const + 5*t", degree=0, y_d_const=y_d_const, t=T)
    y_d = project(y_d_func, V)

    w_target_func = Expression("40", degree=1, t=0)

    y_d = create_test_problem(w_target_func, V, y_0, g, rho, c, k, delta_t, num_steps, ds)


    """ Define admissible set """

    w_a = 4  # Think of as 04 degrees Celsius water
    w_b = 90 # Think of as 90 degrees Celsius water


    # Control over boundary, assumed function of time
    W_0 = []
    w = Expression("10 + 9 * t", degree=1, t=0)
    for i in range(num_steps):
        w.t = i * delta_t
        w_i = interpolate(w, V)
        W_0.append(w_i)

    stop = 0 # Don't stop because of low cost functional.
    max_iter = 20
    max_inner_iter = 20 # How many iterations to find a suitable step length

    W, costs = gradient_descent(W_0, V, y_0, y_d, w_a, w_b, gamma, g, rho, c, k, delta_t, num_steps, ds,
         stop, max_iter, rel_stop=0.99, c1=0.5, alpha_0=10, tau=0.5, max_inner_iter=max_inner_iter)

    fname = "temp.txt"
    save_costs = True
    if save_costs:
        np.savetxt(fname, costs)


    save_start_state = True
    if save_start_state:
        Y, T = state(W_0, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

        oldStateFile = File('gradient_descent/old_state.pvd')

        yi = Function(V)
        for y_i, t_i in zip(Y, T):
            
            # Save to file
            yi.assign(y_i)
            
            oldStateFile << (yi, t_i)

        oldStateDiffFile = File('gradient_descent/old_state_diff.pvd')

        for y_i, t_i in zip(Y, T):

            # Save to file
            yi.assign(y_i - y_d)
            
            oldStateDiffFile << (yi, t_i)

    save_control = True
    if save_control:

        newControlFile = File('gradient_descent/new_control.pvd')

        wi = Function(V)
        for (w_i, t_i) in zip(W, T):

            wi.assign(w_i)

            newControlFile << (wi, t_i)


    save_end_state = True
    if save_end_state:

        Y, T = state(W, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

        newStateFile = File('gradient_descent/new_state.pvd')

        yi = Function(V)
        for y_i, t_i in zip(Y, T):
            
            # Save to file
            yi.assign(y_i)
            
            newStateFile << (yi, t_i)

        newStateDiffFile = File('gradient_descent/new_state_diff.pvd')

        for y_i, t_i in zip(Y, T):

            # Save to file
            yi.assign(y_i - y_d)
            
            newStateDiffFile << (yi, t_i)

    return


if __name__ == '__main__':
    main()
