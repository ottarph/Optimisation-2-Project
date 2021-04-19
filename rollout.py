from fenics import *
from state_equation import *
from adjoint_equation import *
from cost_functional import *
from tools import *
import numpy as np

""" Control whether or not to save animations """
SAVE = False

""" Define domain and space """

T = 3.0                   # final time
num_steps = 50            # number of time steps
delta_t = T / num_steps   # time step size

L = 4
B = 2

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(L, B), int(L*10), int(B*10))
V = FunctionSpace(mesh, "Lagrange", 1)


""" Define new boundary-measure to split integral over Gamma_0, Gamma_1 """
ds = create_boundary_measure(mesh, L, B)


""" Defining the cost functional """

# Penalty constant for control
gamma = 1

# Target temperature function over time
y_d_const = 30
y_d_func = Expression("y_d_const + 5*t", degree=0, y_d_const=y_d_const, t=T)
y_d = project(y_d_func, V)


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
W = []
w = Expression("t < 1.5 ? 4 : 90", degree=1, t=0)
for k in range(num_steps):
    w.t = k * delta_t
    w_k = interpolate(w, V)
    W.append(w_k)



""" ----------------- State equation ----------------- """

Y, T = state(W, V, y_0, g, rho, c, k, delta_t, num_steps, ds)


if SAVE:
    # Create PVD file for saving solution
    stateFile = File('rollout/state_equation.pvd')
    
    y = Function(V)
    for (y_n, t_n) in zip(Y, T):
        
        y.assign(y_n)
        
        stateFile << (y, t_n)


print(cost_functional(Y, W, T, y_d_func, delta_t, V, gamma))



""" ----------------- Adjoint equation ----------------- """


P = adjoint_eq(V, Y, T, y_d, g, rho, c, k, delta_t, num_steps, ds)

if SAVE:
    # Create PVD file for saving adjoint
    adjFile = File('rollout/adjoint_equation.pvd')
    
    # Save adjoint equation solution to file
    p = Function(V)
    for p_n, t_n in zip(P, T):
        
        # Save to file
        p.assign(p_n)
        
        adjFile << (p, t_n)


""" ----------------- Compute gradient ----------------- """

grad = []
for k, (t, p, w) in enumerate(zip(T, P, W)):
    g.t = t
    gg = interpolate(g, V)
    gk = Function(V)
    gk.vector()[:] = gg.vector()[:] * p.vector()[:] + gamma * w.vector()[:]
    grad.append(gk)

if SAVE:
    # Create PVD file for saving gradient
    gradFile = File('rollout/gradient.pvd')
    
    # Save adjoint equation solution to file
    gk = Function(V)
    for g_k, t_k in zip(grad, T):
        
        # Save to file
        gk.assign(g_k)
        
        gradFile << (gk, t_k)


""" Compute solution for incremented control """

step_length = 1

W_new = []
for w, g in zip(W, grad):
    w_new = Function(V)
    w_new.vector()[:] = w.vector()[:] - step_length*g.vector()[:]
    W_new.append(w_new)

Y_new, _ = state(W_new, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

""" Compute cost functional for new control """

print(cost_functional(Y_new, W_new, T, y_d_func, delta_t, V, gamma))

if SAVE:
    """ Saving new control to file """
    
    # Create PVD file for saving new control
    controlFile = File('rollout/new_control.pvd')
    
    # Save control to file
    wk = Function(V)
    for w_k, t_k in zip(W_new, T):
        
        # Save to file
        wk.assign(w_k)
        
        controlFile << (wk, t_k)

if SAVE:
    """ Saving state with new control to file """
    
    # Create PVD file for saving new control
    newStateFile = File('rollout/new_state.pvd')
    
    # Save adjoint equation solution to file
    yk = Function(V)
    for y_k, t_k in zip(Y_new, T):
        
        # Save to file
        yk.assign(y_k)
        
        newStateFile << (yk, t_k)


""" Project new control onto the admissible set """

W_new_ad = []
for w in W_new:
    w_ad = Function(V)
    w_ad.vector()[:] = np.maximum(w_a, np.minimum(w_b, w.vector()[:]))
    W_new_ad.append(w_ad)

""" Compute solution for new admissible control """

Y_new_ad, _ = state(W_new_ad, V, y_0, g, rho, c, k, delta_t, num_steps, ds)

""" Compute cost functional for new admissible control """

print(cost_functional(Y_new_ad, W_new_ad, T, y_d_func, delta_t, V, gamma))

if SAVE:
    """ Saving new admissible control to file """
    
    # Create PVD file for saving new control
    adControlFile = File('rollout/new_ad_control.pvd')
    
    # Save control to file
    wk = Function(V)
    for w_k, t_k in zip(W_new_ad, T):
        
        # Save to file
        wk.assign(w_k)
        
        adControlFile << (wk, t_k)

if SAVE:
    """ Saving state with new admissible control to file """
    
    # Create PVD file for saving new control
    newAdStateFile = File('rollout/new_ad_state.pvd')
    
    # Save adjoint equation solution to file
    yk = Function(V)
    for y_k, t_k in zip(Y_new_ad, T):
        
        # Save to file
        yk.assign(y_k)
        
        newAdStateFile << (yk, t_k)
