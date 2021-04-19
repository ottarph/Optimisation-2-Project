from fenics import *


def cost_functional(Y, W, T, y_d, delta_t, V, gamma):
    
    cost = 0
    for w in W:
        cost += 0.5*delta_t*gamma * assemble( w*w*ds ) 

    y = Function(V)
    y.assign(Y[-1])
    cost += 0.5 * assemble( (y - y_d)*(y - y_d)*dx ) 

    return cost
