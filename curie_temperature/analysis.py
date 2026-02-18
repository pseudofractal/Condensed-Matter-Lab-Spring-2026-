import os
import matplotlib.pyplot as plt
from utils import load_data, apply_vtotal_method, compute_capacitance
from plotting import plot_rt, plot_derivative, finalize_plot

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

FILES = {
    "SC1 (22 nF)": "SC1_22nF_raw_data.csv",
    "SC2 (39 nF)": "SC1_22nF_raw_data.csv",
    "SC3 (62 nF)": "SC3_62nF_raw_data.csv",
}

results = {}

for label, filename in FILES.items():

    path = os.path.join(DATA_DIR, filename)

    df = load_data(path)

    df, V_total_avg = apply_vtotal_method(df)
    df, C_ref = compute_capacitance(df)

    results[label] = df

    print(f"\n{label}")
    print("V_total_avg =", V_total_avg)
    print("Reference C =", C_ref)

# -----------------------
# Plot epsilon_r vs T
# -----------------------

plt.figure()

for label, df in results.items():
    plt.plot(df["Temperature_C"], df["epsilon_r"], label=label)

plt.xlabel("Temperature (°C)")
plt.ylabel("Relative Permittivity (εr)")
plt.title("εr vs Temperature (All Samples)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("epsilon_r_vs_T_all.png")
plt.show()
