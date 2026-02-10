import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pint import UnitRegistry
from scipy.optimize import curve_fit
from uncertainties import ufloat

# Initialize Unit Registry and satisfy type checkers
ureg = UnitRegistry()
Quantity: Any = ureg.Quantity

# Plotting Configuration
plt.rcParams.update(
  {
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
  }
)


def get_paths() -> dict[str, Path]:
  """Initializes the directory structure relative to the script location."""
  base = Path(__file__).parent
  paths = {
    "data": base / "data",
    "plots": base / "plots",
    "final": base / "final",
    "sample": base / "sample_data",
    "config": base / "config.json",
  }
  for name, path in paths.items():
    if name != "config":
      path.mkdir(parents=True, exist_ok=True)
  return paths


def setup_experiment(paths: dict[str, Path]):
  """Sets up default configuration with per-material currents and sample data."""
  if not paths["config"].exists():
    config = {
      "system_resistance_ohm": 10.0,
      "calibration": {
        "material": "Silver",
        "current_A": 2.5,
        "mass_kg": 0.200,
        "cv_j_kgk": 235.0,
      },
      "samples": [
        {"material": "Copper", "current_A": 2.0, "mass_kg": 0.150},
        {"material": "Aluminum", "current_A": 3.0, "mass_kg": 0.120},
        {"material": "Iron", "current_A": 2.2, "mass_kg": 0.180},
      ],
    }
    with open(paths["config"], "w", encoding="utf-8") as f:
      json.dump(config, f, indent=2)

  cal_path = paths["sample"] / "calibration_silver.csv"
  if not cal_path.exists():
    pd.DataFrame({"dT_K": [2.0, 4.0, 6.0, 8.0], "dt_s": [45, 91, 136, 182]}).to_csv(
      cal_path, index=False
    )

  for name in ["copper", "aluminum", "iron"]:
    p = paths["sample"] / f"{name}_data.csv"
    if not p.exists():
      pd.DataFrame({"dT_K": [], "dt_s": []}).to_csv(p, index=False)


def linear_model(x: np.ndarray, m: float, c: float) -> np.ndarray:
  """Standard linear response model."""
  return m * x + c


LIT_VALUES = {
  "Silver": 235.0,
  "Copper": 385.0,
  "Aluminum": 900.0,
  "Brass": 380.0,
}


def perform_interval_analysis(
  df: pd.DataFrame, temp_min: float = 25.5, temp_max: float = 27.5
) -> dict[str, Any]:
  """Calculates statistics for intervals within a specific temperature range."""
  # Filter data for calculation
  mask = (df["Temp_C"] >= temp_min) & (df["Temp_C"] <= temp_max)
  calc_df = df[mask].copy()

  if calc_df.empty:
    # If range is completely missing, use first 5 points as fallback
    calc_df = df.head(5).copy()
    temp_min, temp_max = calc_df["Temp_C"].min(), calc_df["Temp_C"].max()

  intervals = calc_df["dt_s"].to_numpy()
  mean_dt = np.mean(intervals)
  std_dt = np.std(intervals, ddof=1)

  return {
    "mean_dt": ufloat(mean_dt, std_dt),
    "all_x": df["Temp_C"].to_numpy(),
    "all_y": df["dt_s"].to_numpy(),
    "n": len(intervals),
    "range_str": f"[{temp_min}, {temp_max}]",
  }


def plot_simple_scatter(
  res: dict[str, Any],
  title: str,
  output_path: Path,
  result_text: str,
):
  """Simple scatter plot of all data with a result text box."""
  fig, ax = plt.subplots(figsize=(8, 5))

  ax.scatter(res["all_x"], res["all_y"], color="black", marker="o", s=30, label="Measured Data")

  ax.set_xlabel(r"Temperature $T$ [$^\circ$C]")
  ax.set_ylabel(r"Time to raise Temperature by 0.5 K [s]")
  ax.set_title(title)

  ax.text(
    0.05,
    0.95,
    result_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    fontsize=11,
  )

  ax.grid(True, alpha=0.3, linestyle="--")
  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()


def plot_cumulative_silver(df: pd.DataFrame, output_path: Path):
  """Plots cumulative time vs temperature for Silver with multiple curve fits."""
  times = np.cumsum(df["dt_s"].to_numpy())
  temps = df["Temp_C"].to_numpy()

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.scatter(temps, times, color="black", s=30, label="Data Points", zorder=10)

  x_smooth = np.linspace(temps.min(), temps.max(), 200)

  # Fits: 2, 3, 4, 5, 10
  degrees = [2, 3, 5, 10]
  colors = ["red", "green", "blue", "orange"]
  styles = ["--", "-", "-.", ":"]

  for deg, color, style in zip(degrees, colors, styles):
    p = np.poly1d(np.polyfit(temps, times, deg))
    ax.plot(
      x_smooth, p(x_smooth), color=color, linestyle=style, alpha=0.8, label=f"Degree {deg} Fit"
    )

  ax.set_xlabel(r"Temperature $T$ [$^\circ$C]")
  ax.set_ylabel(r"Total Time Elapsed $t$ [s]")
  ax.set_title("Silver Calibration: Cumulative Time vs Temperature (High Order Fits)")
  ax.grid(True, alpha=0.3, linestyle="--")
  ax.legend(loc="upper left")

  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()


def plot_silver_residuals(df: pd.DataFrame, output_path: Path):
  """Plots residuals for various polynomial degrees on Silver data."""
  times = np.cumsum(df["dt_s"].to_numpy())
  temps = df["Temp_C"].to_numpy()

  fig, ax = plt.subplots(figsize=(8, 6))

  degrees = [2, 3, 5, 10]
  colors = ["red", "green", "blue", "orange"]
  markers = ["o", "s", "D", "x"]

  for deg, color, marker in zip(degrees, colors, markers):
    p = np.poly1d(np.polyfit(temps, times, deg))
    res = times - p(temps)
    ax.plot(
      temps,
      res,
      color=color,
      marker=marker,
      linestyle="-",
      alpha=0.7,
      label=f"Degree {deg} Residuals",
    )

  ax.axhline(0, color="black", linewidth=1, linestyle="-")
  ax.set_xlabel(r"Temperature $T$ [$^\circ$C]")
  ax.set_ylabel(r"Residual $(t_{obs} - t_{pred})$ [s]")
  ax.set_title("Residual Analysis: High-Order Polynomial Fits")
  ax.grid(True, alpha=0.3, linestyle="--")
  ax.legend()

  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()


def main():
  paths = get_paths()
  setup_experiment(paths)

  with open(paths["config"], encoding="utf-8") as f:
    cfg = json.load(f)

  resistance = cfg["system_resistance_ohm"]

  # Phase 1: Calibration
  cal_cfg = cfg["calibration"]
  cal_file = paths["data"] / f"calibration_{cal_cfg['material'].lower()}.csv"

  if not cal_file.exists():
    print(f"Phase 1 missing. Provide calibration data at {cal_file}")
    return

  df_cal = pd.read_csv(cal_file)
  res_cal = perform_interval_analysis(df_cal, temp_min=25.5, temp_max=27.5)

  # Physics: m_slope = mean_dt / 0.5
  effective_slope_cal = res_cal["mean_dt"] / 0.5
  cal_power = (cal_cfg["current_A"] ** 2) * resistance
  water_equivalent = effective_slope_cal * cal_power - (cal_cfg["mass_kg"] * cal_cfg["cv_j_kgk"])

  plot_simple_scatter(
    res_cal,
    f"Calibration Analysis (Silver)",
    paths["plots"] / "calibration-Ag.png",
    rf"Water Equivalent (W):" "\n" rf"${water_equivalent.n:.2f} \pm {water_equivalent.s:.2f}$ J/K",
  )

  # New Cumulative Plot for Silver
  plot_cumulative_silver(df_cal, paths["plots"] / "calibration-Ag-cumulative.png")
  plot_silver_residuals(df_cal, paths["plots"] / "calibration-Ag-residuals.png")

  print(f"Calibration Complete. W = {water_equivalent:.2f} J/K")

  # Phase 2: Iterating through Materials
  summary_results = []
  found_phase2 = False

  for sample in cfg["samples"]:
    mat_name = sample["material"]
    mat_file = paths["data"] / f"{mat_name.lower()}_data.csv"

    if not mat_file.exists() or pd.read_csv(mat_file).empty:
      continue

    found_phase2 = True
    df_mat = pd.read_csv(mat_file)
    res_mat = perform_interval_analysis(df_mat, temp_min=25.5, temp_max=27.5)

    # Physics: Cv = (m_slope * I^2 * R - W) / m_sample
    effective_slope_mat = res_mat["mean_dt"] / 0.5
    mat_power = (sample["current_A"] ** 2) * resistance
    cv_val = (effective_slope_mat * mat_power - water_equivalent) / sample["mass_kg"]

    # Mapping material names for files
    file_map = {"Aluminum": "Al", "Copper": "Cu", "Brass": "Brass"}
    file_name = f"{file_map.get(mat_name, mat_name)}.png"

    plot_simple_scatter(
      res_mat,
      f"Specific Heat Analysis: {mat_name}",
      paths["plots"] / file_name,
      rf"$C_v = {cv_val.n:.2f} \pm {cv_val.s:.2f}$ J/(kg·K)",
    )

    summary_results.append(
      {
        "Material": mat_name,
        "Current": sample["current_A"],
        "Cv": cv_val,
        "R2": 0.0,
      }  # R2 not relevant for mean
    )

  if not found_phase2:
    print("Phase 1 finished. Populate sample data files for Phase 2.")
    return

  # Generate Final Report
  report_path = paths["final"] / "experiment_summary.txt"
  with open(report_path, "w", encoding="utf-8") as f:
    f.write("Calorimetry Analysis Report\n" + "=" * 40 + "\n")
    f.write(f"Water Equivalent (W): {water_equivalent.n:.4e} +/- {water_equivalent.s:.4e} J/K\n")
    f.write("-" * 40 + "\n")
    for item in summary_results:
      f.write(f"Material: {item['Material']}\n")
      f.write(f"  Applied Current: {item['Current']:.2f} A\n")
      f.write(f"  Measured Cv:     {item['Cv'].n:.4e} +/- {item['Cv'].s:.4e} J/(kg·K)\n")
      f.write(f"  Fit Quality R2:  {item['R2']:.5f}\n\n")

  print(f"Analysis complete for {len(summary_results)} materials. Results in {report_path}")


if __name__ == "__main__":
  main()
