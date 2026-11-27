import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ============================================================
# 1. Slow-repair ODE model
# ============================================================

def slow_repair_ode(t, y, c0, lam, kp, ke):
    """
    Slow DNA repair model.

    U = uncut DNA
    B = broken DNA
    P = precisely repaired DNA
    E = permanently edited DNA
    """

    U, B, P, E = y

    # Time-dependent Cas9 cutting rate
    c_t = c0 * np.exp(-lam * t)

    dUdt = -c_t * U
    dBdt = c_t * U - (kp + ke) * B
    dPdt = kp * B
    dEdt = ke * B

    return [dUdt, dBdt, dPdt, dEdt]


# ============================================================
# 2. Solve model at requested time points
# ============================================================

def simulate_model(params, time_points):

    c0, lam, kp, ke = params

    # Initial conditions from the mathematical model
    y0 = [1.0, 0.0, 0.0, 0.0]

    time_points = np.asarray(time_points, dtype=float)

    # We need to integrate from t = 0 even though
    # the first experimental measurement is at day 0.5.
    solution = solve_ivp(
        slow_repair_ode,
        [0, np.max(time_points)],
        y0,
        args=(c0, lam, kp, ke),
        t_eval=time_points,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution


# ============================================================
# 3. Load experimental data
# ============================================================

file_name = "data/CRISPR_Modeling_Data.xlsx"

data = pd.read_excel(
    file_name,
    sheet_name="Fit Ready"
)

print("Columns in Fit Ready:")
print(data.columns.tolist())

print("\nTotal rows in Fit Ready:")
print(len(data))


# ============================================================
# 4. Select Figure 2a neuronal experiment
# ============================================================

fig2a = data[
    (data["experiment_id"] == "Figure_2a") &
    (data["cell_type"] == "neuron") &
    (data["guide"] == "B2Mg1")
].copy()


print("\nFigure 2a rows:")
print(len(fig2a))

print("\nExperimental time points:")
print(sorted(fig2a["time_days"].unique()))


# ============================================================
# 5. Prepare experimental data
# ============================================================

# Summary statistics are used for plotting
summary = (
    fig2a
    .groupby("time_days")["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

print("\nFigure 2a experimental summary:")
print(summary)


# ------------------------------------------------------------
# Mean data - used only for plotting
# ------------------------------------------------------------

time_mean = summary["time_days"].to_numpy(dtype=float)
E_mean = summary["mean"].to_numpy(dtype=float)
E_sd = summary["std"].to_numpy(dtype=float)


# ------------------------------------------------------------
# Replicate-level data - used for parameter fitting
# ------------------------------------------------------------

time_data = fig2a["time_days"].to_numpy(dtype=float)
E_data = fig2a["value"].to_numpy(dtype=float)

print("\nNumber of replicate-level observations:")
print(len(E_data))

print("\nReplicates per time point:")
print(fig2a.groupby("time_days").size())


LAMBDA_FIXED = 0.078267


def residuals_fixed_lambda(params, time_data, E_data):
    """
    Calculate residuals for every experimental replicate.

    lambda is fixed using the independent Cas9
    abundance experiment.
    """

    c0, kp, ke = params

    full_params = [
        c0,
        LAMBDA_FIXED,
        kp,
        ke
    ]

    # solve_ivp requires unique time points
    unique_times = np.sort(
        np.unique(time_data)
    )

    solution = simulate_model(
        full_params,
        unique_times
    )

    E_unique = solution.y[3]

    # Map each time point to its model prediction
    prediction_lookup = dict(
        zip(unique_times, E_unique)
    )

    # Give every replicate the prediction
    # corresponding to its experimental time
    E_model = np.array([
        prediction_lookup[t]
        for t in time_data
    ])

    return E_model - E_data

# ============================================================
# Fit c0, kp, ke with lambda fixed
# ============================================================

initial_guess = [
    0.5,    # c0
    0.05,   # kp
    0.10    # ke
]

lower_bounds = [
    1e-6,
    1e-6,
    1e-6
]

upper_bounds = [
    10.0,
    10.0,
    10.0
]


fit = least_squares(
    residuals_fixed_lambda,
    initial_guess,
    bounds=(lower_bounds, upper_bounds),
    args=(time_data, E_data)
)


c0_fit, kp_fit, ke_fit = fit.x

lam_fit = LAMBDA_FIXED


print("\n====================================")
print("FITTED PARAMETERS - FIXED LAMBDA")
print("====================================")

print(f"c0     = {c0_fit:.6f} per day")
print(f"lambda = {lam_fit:.6f} per day (fixed from Cas9 data)")
print(f"kp     = {kp_fit:.6f} per day")
print(f"ke     = {ke_fit:.6f} per day")

# ============================================================
# 8. Repair timescale
# ============================================================

repair_timescale = 1.0 / (kp_fit + ke_fit)

print(f"\nRepair timescale = {repair_timescale:.4f} days")


# ============================================================
# 9. Model prediction at experimental times
# ============================================================
fitted_params = [
    c0_fit,
    lam_fit,
    kp_fit,
    ke_fit
]

# ============================================================
# Model predictions for all 36 observations
# ============================================================

unique_times = np.sort(
    np.unique(time_data)
)

fitted_solution = simulate_model(
    fitted_params,
    unique_times
)

E_unique_prediction = fitted_solution.y[3]

prediction_lookup = dict(
    zip(unique_times, E_unique_prediction)
)

E_prediction = np.array([
    prediction_lookup[t]
    for t in time_data
])



# ============================================================
# 10. RMSE
# ============================================================

rmse = np.sqrt(
    np.mean((E_data - E_prediction) ** 2)
)

print(f"RMSE = {rmse:.6f}")


# ============================================================
# 11. AIC
# ============================================================

n = len(E_data)

# c0, kp, ke are estimated from Figure 2a
k = 3

rss = np.sum(
    (E_data - E_prediction) ** 2
)

aic = n * np.log(rss / n) + 2 * k

print(f"AIC = {aic:.4f}")

# ============================================================
# 12. Create smooth fitted curve
# ============================================================

t_smooth = np.linspace(
    0,
    max(time_data),
    500
)

smooth_solution = simulate_model(
    fitted_params,
    t_smooth
)

U = smooth_solution.y[0]
B = smooth_solution.y[1]
P = smooth_solution.y[2]
E = smooth_solution.y[3]


# ============================================================
# 13. Plot experimental data vs fitted model
# ============================================================

# ============================================================
# Plot experimental data vs fitted model
# ============================================================

plt.figure(figsize=(9, 6))


# Plot all 36 individual observations
plt.scatter(
    time_data,
    E_data,
    alpha=0.35,
    label="Individual replicates"
)


# Plot mean and standard deviation
plt.errorbar(
    time_mean,
    E_mean,
    yerr=E_sd,
    fmt="o",
    capsize=4,
    label="Experimental mean ± SD"
)


# Plot fitted model
plt.plot(
    t_smooth,
    E,
    linewidth=2,
    label="Slow-repair model"
)


plt.xlabel("Time (days)")
plt.ylabel("Edited fraction E(t)")

plt.title(
    "Figure 2a: Experimental Data vs Slow-Repair Model"
)

plt.ylim(0, 1)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plots/figure2a/figure2a_slow_repair_fit.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. Plot all predicted DNA states
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(t_smooth, U, label="Uncut U(t)")
plt.plot(t_smooth, B, label="Broken B(t)")
plt.plot(t_smooth, P, label="Precisely repaired P(t)")
plt.plot(t_smooth, E, label="Edited E(t)")

plt.xlabel("Time (days)")
plt.ylabel("Fraction of target DNA")

plt.title(
    "Fitted Slow-Repair Model - Figure 2a"
)

plt.ylim(0, 1)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plots/figure2a/figure2a_fitted_states.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. Residual plot
# ============================================================


model_residuals = E_data - E_prediction


plt.figure(figsize=(8, 5))

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.scatter(
    time_data,
    model_residuals,
    alpha=0.65
)

plt.xlabel("Time (days)")
plt.ylabel("Residual (Experimental - Model)")

plt.title(
    "Slow-Repair Model Residuals - Figure 2a"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plots/figure2a/figure2a_residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. Conservation check
# ============================================================

total_DNA = U + B + P + E

conservation_error = np.max(
    np.abs(total_DNA - 1)
)

print(
    f"\nMaximum DNA conservation error = "
    f"{conservation_error:.3e}"
)


print("\nPlots saved:")
print("figure2a_slow_repair_fit.png")
print("figure2a_fitted_states.png")
print("figure2a_residuals.png")



print("\n====================================")
print("FINAL FIGURE 2a RESULTS")
print("====================================")

print(f"Number of observations = {len(E_data)}")
print(f"c0 = {c0_fit:.6f} per day")
print(f"lambda = {lam_fit:.6f} per day (fixed)")
print(f"kp = {kp_fit:.6f} per day")
print(f"ke = {ke_fit:.6f} per day")

print(
    f"Repair timescale = "
    f"{repair_timescale:.4f} days"
)

print(f"RMSE = {rmse:.6f}")
print(f"AIC = {aic:.4f}")

if np.isclose(c0_fit, upper_bounds[0], rtol=1e-4):
    print(
        "WARNING: c0 reached its upper fitting bound; "
        "do not interpret c0 as a well-identified estimate."
    )

    
# # ============================================================
# # 17. Multi-start fitting
# # ============================================================

# starting_points = [
#     [0.1, 0.01, 0.01, 0.01],
#     [0.5, 0.1, 0.05, 0.10],
#     [1.0, 0.1, 0.1, 0.2],
#     [2.0, 0.5, 0.05, 0.2],
#     [5.0, 1.0, 0.5, 0.5],
#     [9.0, 0.01, 0.1, 0.2]
# ]

# print("\n====================================")
# print("MULTI-START PARAMETER FITTING")
# print("====================================")

# for guess in starting_points:

#     result = least_squares(
#         residuals,
#         guess,
#         bounds=(lower_bounds, upper_bounds),
#         args=(time_data, E_data)
#     )

#     prediction = simulate_model(
#         result.x,
#         time_data
#     ).y[3]

#     rmse_test = np.sqrt(
#         np.mean((E_data - prediction) ** 2)
#     )

#     print("\nInitial guess:", guess)

#     print(
#         "Fitted:",
#         np.round(result.x, 6)
#     )

#     print(
#         f"RMSE: {rmse_test:.6f}"
#     )