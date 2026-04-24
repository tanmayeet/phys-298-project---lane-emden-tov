import pandas as pd
import numpy as np
import matplotlib as mpl
from scipy.integrate import solve_ivp

def TOV_sys(r,S):
    u,t = S
    dudr = (r**2)* (np.sinh(t) - t)
    dtdr = (-4/(r*(r-2*u)))*(np.sinh(t) - 2* np.sinh(t/2))*(((r**3)/3)*(np.sinh(t) + 8* np.sinh(t/2) + 3*t)+ u)/(np.cosh(t) - 4*np.cosh(t/2) +3)
    dSdx = [dudr, dtdr]
    return(dSdx)


mu_not = 1
c = 1
G= 1
h=1

#defining constants
#alpha = (1/np.pi)*((h/((mu_not)*(c)))^(3/2) )*(c/((mu_not * G)^(1/2)))
#beta = ((c^2)/G)*alpha

# IC's
fermi_p = 3
def t_not_comp(fermi_p):
    t_o = 4*np.arcsinh(fermi_p/(mu_not * c))
    return t_o


u_o = 0
t_o = t_not_comp(fermi_p)
r_o = 0
S_o = [u_o, t_o]


#surface event
def surf(r, S):
    return(S[1])
surf.terminal = True
surf.direction = -1

#solver
def solver(t0, xmax=100):
    t_o = t_not_comp(t0)
    S_o = [0, t_o]

    solution = solve_ivp(
        TOV_sys,
        (1e-6, xmax),
        S_o,
        method ="DOP853",
        events = surf,
        max_step = 0.01,
        rtol = 1e-10,
        atol = 1e-12
    )

    rvals = solution.t
    uvals = solution.y[0]
    tvals = solution.y[1]

    if len(solution.t_events[0]) > 0:
        r_surf = solution.t_events[0][0]
        mass = uvals[-1]
        real_surface = True
    else:
        r_surf = xmax
        mass = uvals[-1]
        real_surface = False



    return rvals, uvals, tvals, r_surf, mass, real_surface


temp_t = [0.2, 1, 2, 3]

rows = []
for t0 in temp_t:
    rvals, uvals, tvals, r_surf, mass, real_surface = solver(t0)
    rows.append({
        't' : t0,
        'r':  round(r_surf, 5),
        'mass':    round(mass, 6),
    })

df = pd.DataFrame(rows)
print(df)
