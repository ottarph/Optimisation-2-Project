from fenics import *

def state(w, V, y_0, g, rho, c, k, delta_t, num_steps):

    """ ----------------- State equation ----------------- """

    # Define variational problem
    y = TrialFunction(V)
    v = TestFunction(V)

    y_n = interpolate(y_0, V)

    a = ( rho*c * y * v + delta_t*k * inner(grad(y), grad(v)) )*dx + ( delta_t * g * y * v )*ds
    L = ( rho*c * y_n * v )*dx + ( delta_t * g * w * v )*ds


    Y = []
    T = []

    # Time-stepping
    y = Function(V)
    y.assign(y_n)
    t = 0
    y0 = Function(V)
    y0.assign(y)
    Y.append(y0)
    T.append(t)
    for n in range(num_steps):

        # Update current time
        t += delta_t

        # Save time used for y-step
        T.append(t)

        # Update control and heat coefficient for new time
        w.t = t
        g.t = t

        # Compute solution
        solve(a == L, y)

        # Keep solution for use in adjoint equation
        y_k = Function(V)
        y_k.assign(y)
        Y.append(y_k)

        # Update previous solution
        y_n.assign(y)

    return Y, T
