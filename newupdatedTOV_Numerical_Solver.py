import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# From Oppenheimer for neutron gas
a_cm = 1.36e6
a_km = a_cm / 1e5

b_g = 1.83e34
M_sun_g = 2.0e33
mass_unit_solar = b_g / M_sun_g

def TOV_sys(r, S):
    u, t = S

    A = np.sinh(t) - t
    B = np.sinh(t) - 8*np.sinh(t/2) + 3*t
    C = np.sinh(t) - 2*np.sinh(t/2)
    D = np.cosh(t) - 4*np.cosh(t/2) + 3

    dudr = r**2 * A

    dtdr = (-4 / (r * (r - 2*u))) * (C / D) * ((r**3 / 3) * B + u)

    return np.array([dudr, dtdr])

def rk4_step(f, r, S, h):
    k1 = f(r, S)
    k2 = f(r + h/2, S + h*k1/2)
    k3 = f(r + h/2, S + h*k2/2)
    k4 = f(r + h, S + h*k3)

    return S + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def solver_RK4(t0, h=0.001, rmax=100):
    eps = 1e-6

    # Near-center expansion for u
    u0 = (np.sinh(t0) - t0) * eps**3 / 3
    S = np.array([u0, t0])

    r = eps

    r_values = [r]
    u_values = [S[0]]
    t_values = [S[1]]

    while r < rmax and S[1] > 0:
        S_old = S.copy()
        r_old = r

        S = rk4_step(TOV_sys, r, S, h)
        r = r + h

        
        if S[1] <= 0:
            t_old = S_old[1]
            t_new = S[1]

            frac = t_old / (t_old - t_new)

            r_surface = r_old + frac*h
            u_surface = S_old[0] + frac*(S[0] - S_old[0])

            radius_km = r_surface * a_km
            mass_solar = u_surface * mass_unit_solar

            return (
                np.array(r_values),
                np.array(u_values),
                np.array(t_values),
                r_surface,
                u_surface,
                radius_km,
                mass_solar,
                True
            )

        r_values.append(r)
        u_values.append(S[0])
        t_values.append(S[1])

    r_surface = r
    u_surface = S[0]
    radius_km = r_surface * a_km
    mass_solar = u_surface * mass_unit_solar

    return (
        np.array(r_values),
        np.array(u_values),
        np.array(t_values),
        r_surface,
        u_surface,
        radius_km,
        mass_solar,
        False
    )

# Table like the Oppenheimer refpaper
t0_values = [1, 2, 3, 4]

rows = []
for t0 in t0_values:
    rvals, uvals, tvals, r_surface, u_surface, radius_km, mass_solar, surface_found = solver_RK4(t0)

    rows.append({
        "central t0": t0,
        "dimensionless radius": round(r_surface, 4),
        "radius km": round(radius_km, 2),
        "dimensionless mass": round(u_surface, 5),
        "mass solar": round(mass_solar, 3),
        "surface found": surface_found
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# Scan many t0 values to find approximate maximum mass
scan_t0 = np.linspace(0.1, 4.0, 80)

scan_rows = []
for t0 in scan_t0:
    rvals, uvals, tvals, r_surface, u_surface, radius_km, mass_solar, surface_found = solver_RK4(t0)

    if surface_found:
        scan_rows.append({
            "central t0": t0,
            "radius km": radius_km,
            "mass solar": mass_solar
        })

scan_df = pd.DataFrame(scan_rows)

print("\nNumber of scan points:", len(scan_df))

max_row = scan_df.loc[scan_df["mass solar"].idxmax()]

print("\nApproximate maximum mass:")
print("central t0 =", round(max_row["central t0"], 3))
print("radius km  =", round(max_row["radius km"], 3))
print("mass solar =", round(max_row["mass solar"], 3))

plt.figure()
plt.plot(scan_df["central t0"], scan_df["mass solar"], marker="o")
plt.xlabel("Central t0")
plt.ylabel("Mass in solar masses")
plt.title("RK4: Neutron Core Mass vs Central t0")
plt.grid(True)
plt.savefig("mass_vs_t0.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(scan_df["radius km"], scan_df["mass solar"], marker="o")
plt.xlabel("Radius in km")
plt.ylabel("Mass in solar masses")
plt.title("RK4: Mass-Radius Curve")
plt.grid(True)
plt.savefig("mass_radius_curve.png", dpi=300, bbox_inches="tight")
plt.show()
