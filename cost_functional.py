from fenics import *
from state_equation import *
from adjoint_equation import *


def cost_functional(Y, W, T, y_d_func, delta_t, V, gamma):
    
    cost = 0
    for y, w, t in zip(Y, W, T):
        y_d_func.t = t
        y_d = interpolate(y_d_func, V)
        cost += 0.5*delta_t * ( assemble( (y - y_d)*(y - y_d)*dx ) + gamma*assemble( w*w*ds ) )

    return cost
