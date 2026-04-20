import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

u_0 = 1.0
v_0 = 0.0
S_0 = [u_0, v_0]

def surface(x, S, n):
    return(S[0])
surface.terminal = True
surface.direction = -1

def get_x_max(n):
    if n >= 4.9:
        return 10000
    elif n >= 4.75:
        return 2000
    elif n >= 4.5:
        return 200
    else:
        return 50

def solver(n, x_max=50):
    max_step = max(0.01, x_max / 50000)
    solution = solve_ivp(
        LE_system,
        (1e-6, x_max),
        S_0,
        args=(n,),
        method ="DOP853",
        events = surface,
        max_step = max_step,
        rtol = 1e-10,
        atol = 1e-12
    )
    print(f"n={n}, x_max={x_max}, t_events={solution.t_events[0]}, success={solution.success}")
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
    x_vals, u, dudx, x_end, real_surface = solver(n, x_max=get_x_max(n))
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
