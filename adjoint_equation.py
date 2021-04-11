from fenics import *


def adjoint(V, Y, T, y_d_func, g, rho, c, k, delta_t, num_steps, ds):

    """ ----------------- Adjoint equation ----------------- """

    p = TrialFunction(V)
    h = TestFunction(V)


    y = Function(V)

    p_0 = Expression('0', degree=0)
    p_n = interpolate(p_0, V)

    y_d = interpolate(y_d_func, V)


    a = ( rho*c * p * h + delta_t*k * inner(grad(p), grad(h)) )*dx + \
        ( delta_t * g * p * h )*ds(0)

    L = ( rho*c * p_n * h + delta_t * (y - y_d) * h )*dx


    P = []

    # Time-stepping
    p = Function(V)
    p.assign(p_n)
    p0 = Function(V)
    p0.assign(p)
    P.append(p0)
    for n in range(1, num_steps+1):

        # Update current time
        t = T[-n]

        y.assign(Y[-n])

        y_d_func.t = t
        y_d.assign( interpolate(y_d_func, V) )

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
