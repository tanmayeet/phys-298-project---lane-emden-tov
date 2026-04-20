import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#defining Lane Emden System as two first order DE's
#changing xi to x and theta to u
#defining v as du/dx(dtheta/dxi in standard LE eq)
#have system of two DE, du/dx= v is first eq by defenition
def LE_system(x,S, n):
    u,v = S
    if x < 1e-10:
        return[v,-1/3]
    if u <0:
        u_cap = 0
    else:
        u_cap =u
    dSdx = [v,(-(2*v)/x)-u_cap**n]
    return(dSdx)
#need to handle two cases, if x is zero, we are dividing by zero. L'Hopital says we approach -1/3
#if u = 0, apparently, there is a chance we overshoot 0 and go negative, which is bad.(idek bruh, some guy on stack overflow said that was an issue apparenlty)
u_0 = 1.0
v_0 = 0.0
S_0 = [u_0, v_0]
#IC's as system

#handle leaving the surface
def surface(x, S, n):
    return(S[0])
surface.terminal = True
surface.direction = -1
#ask a real coder for a better explanation of this, but my understanding is that in the scipi library, and in contexgt of solving DE's, an "event" counts as when we change sign from + to - for a value, and therefor cross 0, and when S[0] = 0 is when u(x) = 0, and we want to stop there, so the surfcae.terminal =True tells the computer to stop when we reach that event, the event is recorded by the function surface


#setting x_max to 30 is arbitrary. For realisitic stars, it is usually around 10. This is due to the polyprotic index.
def solver(n, x_max=50):
    solution = solve_ivp(
        LE_system,
        (1e-6, x_max),
        S_0,
        args=(n,),
        method ="DOP853",
        events = surface,
        max_step = 0.01,
        rtol = 1e-10,
        atol = 1e-12
    )
    if len(solution.t_events[0]) > 0:
        x_end = solution.t_events[0][0]
        real_surface = True
    else:
        x_end = x_max
        real_surface = False

    x_vals = solution.t
    u = solution.y[0]
    dudx= solution.y[1]

    return x_vals, u, dudx, x_end, real_surface



rows = []
indices = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 10.0]
for n in indices:
    x_vals, u, dudx, x_end, real_surface = solver(n)
    if real_surface:
        index = np.argmin(np.abs(x_vals - x_end))
        rows.append({
            'n':             n,
            'x₁':            round(x_end, 5),
            '-x^2 du/dx' : round(-x_end**2 * dudx[index], 6)
        })
    else:
        rows.append({
            'n':             n,
            'x₁':            '∞',
            '-x^2 du/dx' : '—'
        })

df = pd.DataFrame(rows)
print(df)
#keep in mind that for any polytropic index values less than 5 but approaching 5, youll need to adjust tolerance by adjusting xmax