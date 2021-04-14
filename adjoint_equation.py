from fenics import *


def adjoint_eq(V, Y, T, y_d_func, g, rho, c, k, delta_t, num_steps, ds):

    """ ----------------- Adjoint equation ----------------- """

    p = TrialFunction(V)
    h = TestFunction(V)


    y = Function(V)

    y_d_func.t = T[-1]
    y_d = interpolate(y_d_func, V)

    y.assign(Y[-1])

    p_0 = 1/(rho*c) * (y - y_d)
    
    p_n = Function(V)


    a = ( rho*c * p * h + delta_t*k * inner(grad(p), grad(h)) )*dx + \
        ( delta_t * g * p * h )*ds(0)

    L = ( rho*c * p_n * h )*dx


    P = []

    # Time-stepping
    p = Function(V)
    p.assign(p_0)
    p_n.assign(p)
    p0 = Function(V)
    p0.assign(p_0)
    P.append(p0)
    for n in range(1, num_steps+1):

        # Update current time
        t = T[-n]

        # Update heat coefficient for new time
        g.t = t

        # Compute solution
        solve(a == L, p)

        # Keep solution to reverse later
        p_k = Function(V)
        p_k.assign(p)
        P.append(p_k)

        # Update previous solution
        p_n.assign(p)

    return P[::-1]
