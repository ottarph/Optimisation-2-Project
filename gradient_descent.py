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
