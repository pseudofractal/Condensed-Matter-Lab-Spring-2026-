import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sc
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from uncertainties import ufloat

plt.rcParams.update(
  {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 200,
    "axes.labelsize": 11,
  }
)


def get_paths():
  """Generates and creates the necessary directory structure for the project."""
  base_path = Path(__file__).parent
  paths = {
    "data": base_path / "data",
    "plots": base_path / "plots",
    "final": base_path / "final",
    "config": base_path / "config.json",
  }
  for name, path in paths.items():
    if name != "config":
      path.mkdir(parents=True, exist_ok=True)
  return paths


def linear_func(x, m, c):
  """Standard linear model for curve fitting."""
  return m * x + c


def get_fit_metrics(x, y, popt, pcov):
  """Calculates R-squared and Relative Slope Uncertainty as measures of fit correctness."""
  residuals = y - linear_func(x, *popt)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((y - np.mean(y)) ** 2)
  r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

  slope_error = np.sqrt(pcov[0, 0])
  rel_error_pct = (slope_error / abs(popt[0])) * 100

  return r_squared, rel_error_pct


def fit_calibration(paths):
  """Fits the electromagnet current vs magnetic field data."""
  file_path = paths["data"] / "calibration.csv"
  if not file_path.exists():
    return None

  df = pd.read_csv(file_path)
  x, y = df["current_A"].to_numpy(), df["field_Gauss"].to_numpy() * 1e-4

  popt, pcov = curve_fit(linear_func, x, y)
  r2, _ = get_fit_metrics(x, y, popt, pcov)

  return ufloat(popt[0], np.sqrt(pcov[0, 0])), ufloat(popt[1], np.sqrt(pcov[1, 1])), r2


def analyze_susceptibility(paths, config, m_cal, c_cal):
  """Performs susceptibility calculations and fit quality assessment for each dataset."""
  results = []
  mu0, g = config["mu_0_SI"], config["g_m_s2"]
  chi_w_mass = config["chi_water_mass_SI"]

  for entry in config["experiments"]:
    file_path = paths["data"] / entry["file_name"]
    if not file_path.exists():
      continue

    df = pd.read_csv(file_path)
    density_lab = entry["density_g_mL"]
    rho_si = density_lab * 1000.0
    h0_m = entry["h0_cm"] * 0.01

    b_vals = np.array([(m_cal * i + c_cal).n for i in df["current_A"]])
    b0_t = df["B0_Gauss"].to_numpy() * 1e-4
    x_data = b_vals**2 - b0_t**2
    y_data_m = np.abs(df["h_observed_cm"].to_numpy() * 0.01 - h0_m)

    popt, pcov = curve_fit(linear_func, x_data, y_data_m)
    r2, rel_err = get_fit_metrics(x_data, y_data_m, popt, pcov)

    slope_u = ufloat(popt[0], np.sqrt(pcov[0, 0]))
    chi_vol = slope_u * (2 * mu0 * rho_si * g)
    chi_mass = (chi_vol / rho_si) - chi_w_mass

    results.append(
      {
        "file": entry["file_name"],
        "density": density_lab,
        "chi_vol": chi_vol,
        "chi_mass": chi_mass,
        "r2": r2,
        "rel_err": rel_err,
        "x": x_data,
        "y_mm": y_data_m * 1000,
        "fit_line_mm": linear_func(x_data, popt[0] * 1000, popt[1] * 1000),
      }
    )
  return results


def plot_quincke(paths, res):
  """Generates plots titled by density in mm units with scientific notation for field terms."""
  fig, ax = plt.subplots(figsize=(6, 4.5))

  ax.scatter(res["x"], res["y_mm"], color="black", marker="o", s=20, label="Observed")
  ax.plot(
    res["x"], res["fit_line_mm"], color="red", linestyle="-", linewidth=1.5, label="Linear Fit"
  )

  ax.set_ylabel(r"Displacement $\Delta h$ [mm]")
  ax.set_xlabel(r"Magnetic Field Term $(B^2 - B_0^2)$ [T$^2$]")

  ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

  info_text = f"$R^2 = {res['r2']:.4f}$\nFit Uncertainty: {res['rel_err']:.2f}%"
  ax.text(
    0.05,
    0.95,
    info_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="silver"),
  )

  ax.set_title(f"Density: {res['density']} g/mL")
  ax.legend(loc="lower right")

  plt.tight_layout()
  plt.savefig(paths["plots"] / f"plot_{Path(res['file']).stem}.png")
  plt.close()


def main():
  """Main execution flow for directory setup, calibration, analysis, and reporting."""
  paths = get_paths()
  if not paths["config"].exists():
    with open(paths["config"], "w") as f:
      json.dump(
        {
          "chi_water_mass_SI": -9.0e-9,
          "g_m_s2": sc.g,
          "mu_0_SI": sc.mu_0,
          "experiments": [{"file_name": "experiment_1.csv", "density_g_mL": 1.15, "h0_cm": 10.0}],
        },
        f,
        indent=2,
      )

  with open(paths["config"]) as f:
    config = json.load(f)

  cal = fit_calibration(paths)
  if not cal:
    print("Required file data/calibration.csv missing.")
    return
  m_cal, c_cal, cal_r2 = cal

  results = analyze_susceptibility(paths, config, m_cal, c_cal)

  with (paths["final"] / "report.txt").open("w") as f:
    f.write("QUINKE'S METHOD: EXPERIMENTAL CORRECTNESS REPORT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Magnet Calibration Linearity (R^2): {cal_r2:.5f}\n\n")

    for r in results:
      plot_quincke(paths, r)
      f.write(f"Dataset: {r['file']}\n")
      f.write(f"  Density: {r['density']} g/mL\n")
      f.write(f"  Volume Susceptibility: {r['chi_vol']:.4e}\n")
      f.write(f"  Mass Susceptibility:   {r['chi_mass']:.4e} m^3/kg\n")
      f.write(f"  Fit Correctness (R^2): {r['r2']:.5f}\n")
      f.write(f"  Relative Uncertainty:  {r['rel_err']:.2f}%\n")
      f.write("-" * 50 + "\n")

  print(f"Analysis Complete. Reports generated in {paths['final']}.")


if __name__ == "__main__":
  main()
