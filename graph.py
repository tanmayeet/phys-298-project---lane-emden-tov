import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lane_emden as le

indices = [1.5, 3.0]
colors = ["blue", "green"]

for n, color in zip(indices, colors):
    x_vals, u, dudx, x_end, real_surface = le.solver(n)

    u_clipped = np.clip(u, 0, None)

    normalized_pressure = u_clipped ** (n + 1)

    plt.plot(x_vals, normalized_pressure, color=color, linewidth=2)

plt.title("Normalized Pressure vs. Dimensionless Radius (Lane-Emden)")
plt.xlabel("Dimensionless radius")
plt.ylabel("Dimensionless normalized pressure")
plt.xlim(0, 8)
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(["n = 1.5", "n = 3.0"])
plt.show()
