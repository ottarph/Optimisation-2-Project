from fenics import *
from state_equation import *
from adjoint_equation import *


def cost_functional(Y, W, T, y_d_func, delta_t, V):
    """ For now only the part concerning Y """
    
    cost = 0
    for y, w, t in zip(Y, W, T):
        y_d_func.t = t
        y_d = interpolate(y_d_func, V)
        cost += delta_t * ( assemble( (y - y_d)*(y - y_d)*dx ) + assemble( w*w*ds ) )

    return cost