import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# ============================================================
# 1. Load Cas9 abundance data
# ============================================================

file_name = "data/CRISPR_Modeling_Data.xlsx"

data = pd.read_excel(
    file_name,
    sheet_name="Cas9 Abundance"
)

print("Columns in Cas9 Abundance:")
print(data.columns.tolist())

print("\nTotal rows:")
print(len(data))


# ============================================================
# 2. Look at the data
# ============================================================

print("\nCas9 abundance data:")
print(data)


# ============================================================
# 3. Select neuron measurements
# ============================================================

neuron_data = data[
    data["cell_type"].str.lower() == "neuron"
].copy()

print("\nNeuron Cas9 measurements:")
print(neuron_data)


# ============================================================
# 4. Calculate mean and SD at each time point
# ============================================================

summary = (
    neuron_data
    .groupby("time_days")["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

print("\nNeuron Cas9 summary:")
print(summary)


time_data = summary["time_days"].to_numpy(dtype=float)
cas9_mean = summary["mean"].to_numpy(dtype=float)
cas9_sd = summary["std"].to_numpy(dtype=float)


# ============================================================
# 5. Exponential Cas9 decay model
# ============================================================

def cas9_decay(t, A, lam):
    """
    Exponential model for Cas9 abundance.

    A:
        Scaling factor for Cas9 abundance.

    lam:
        Cas9 decay rate (per day).

    The immunoblot data are in arbitrary units, so A absorbs
    the unknown measurement scale.
    """

    return A * np.exp(-lam * t)


# ============================================================
# 6. Fit exponential model
# ============================================================

initial_guess = [
    120.0,   # A
    0.05     # lambda
]

parameters, covariance = curve_fit(
    cas9_decay,
    time_data,
    cas9_mean,
    p0=initial_guess,
    bounds=(
        [0.0, 0.0],
        [1000.0, 10.0]
    )
)

A_fit, lambda_fit = parameters


print("\n====================================")
print("CAS9 DECAY FIT")
print("====================================")

print(f"A      = {A_fit:.6f}")
print(f"lambda = {lambda_fit:.6f} per day")


# ============================================================
# 7. Cas9 decay timescale and half-life
# ============================================================

decay_timescale = 1.0 / lambda_fit

half_life = np.log(2) / lambda_fit

print(
    f"Cas9 decay timescale (1/lambda) = "
    f"{decay_timescale:.4f} days"
)

print(
    f"Cas9 half-life = "
    f"{half_life:.4f} days"
)


# ============================================================
# 8. Predictions at measured time points
# ============================================================

cas9_prediction = cas9_decay(
    time_data,
    A_fit,
    lambda_fit
)


# ============================================================
# 9. RMSE
# ============================================================

rmse = np.sqrt(
    np.mean(
        (cas9_mean - cas9_prediction) ** 2
    )
)

print(f"RMSE = {rmse:.6f}")


# ============================================================
# 10. Residuals
# ============================================================

residuals = cas9_mean - cas9_prediction

print("\nResiduals:")

for t, observed, predicted, residual in zip(
    time_data,
    cas9_mean,
    cas9_prediction,
    residuals
):

    print(
        f"Day {t:5.1f}: "
        f"Observed = {observed:8.3f}, "
        f"Predicted = {predicted:8.3f}, "
        f"Residual = {residual:8.3f}"
    )


# ============================================================
# 11. Smooth fitted curve
# ============================================================

t_smooth = np.linspace(
    0,
    max(time_data),
    500
)

cas9_smooth = cas9_decay(
    t_smooth,
    A_fit,
    lambda_fit
)


# ============================================================
# 12. Plot Cas9 abundance and exponential fit
# ============================================================

plt.figure(figsize=(9, 6))

plt.errorbar(
    time_data,
    cas9_mean,
    yerr=cas9_sd,
    fmt="o",
    capsize=5,
    label="Neuron Cas9 mean ± SD"
)

plt.plot(
    t_smooth,
    cas9_smooth,
    linewidth=2,
    label="Exponential decay fit"
)

plt.xlabel("Time (days)")
plt.ylabel("Cas9 abundance (arbitrary units)")

plt.title(
    "Cas9 Abundance in Neurons: Exponential Decay Fit"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plots/cas9/cas9_decay_fit.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. Residual plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.scatter(
    time_data,
    residuals
)

plt.xlabel("Time (days)")
plt.ylabel("Residual (Observed - Predicted)")

plt.title(
    "Cas9 Exponential Decay Residuals"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plots/cas9/cas9_decay_residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nPlots saved:")
print("cas9_decay_fit.png")
print("cas9_decay_residuals.png")