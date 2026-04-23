import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lane_emden as le

plt.style.use("fivethirtyeight")

# Create a directory to save the graphs in
output_folder = "graphs"
os.makedirs(output_folder, exist_ok=True)

# --- Normalized Pressure Graph ---

# Polytropic indices
indices = [1.5, 3.0]
colors = ["blue", "green"]

for n, color in zip(indices, colors):
    # run the numerical solver
    x_vals, u, dudx, x_end, real_surface = le.solver(n)

    normalized_pressure = u ** (n + 1)

    plt.plot(x_vals, normalized_pressure, label=f"n = {n}", linewidth=2)

plt.title("Normalized Pressure vs. Dimensionless Radius (Lane-Emden)")
plt.xlabel(r"Dimensionless Radius ($\xi$)")
plt.ylabel(r"Normalized Pressure ($P / P_c$)")
plt.xlim(0, 8)
plt.ylim(0, 1.05)
plt.legend(["n = 1.5", "n = 3.0"])
filepath = os.path.join(output_folder, "pressure_vs_radius.png")
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.show()

# --- Normalized Density Graph ---

for n, color in zip(indices, colors):
    x_vals, u, dudx, x_end, real_surface = le.solver(n)

    normalized_density = u ** (n)

    plt.plot(x_vals, normalized_density, label=f"n = {n}", linewidth=2)

plt.title("Normalized Density vs. Dimensionless Radius (Lane-Emden)")
plt.xlabel(r"Dimensionless Radius ($\xi$)")
plt.ylabel(r"Normalized Density ($\rho / \rho_c$)")
plt.xlim(0, 8)
plt.ylim(0, 1.05)
plt.legend()
filepath = os.path.join(output_folder, "density_vs_radius.png")
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.show()

## -- TODO: Radius vs. Mass Graph --
