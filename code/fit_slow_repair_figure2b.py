import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ============================================================
# 1. Slow-repair ODE model
# ============================================================

def slow_repair_ode(t, y, c0, lam, kp, ke):

    U, B, P, E = y

    # Time-dependent Cas9 cutting rate
    c_t = c0 * np.exp(-lam * t)

    dUdt = -c_t * U
    dBdt = c_t * U - (kp + ke) * B
    dPdt = kp * B
    dEdt = ke * B

    return [dUdt, dBdt, dPdt, dEdt]


# ============================================================
# 2. Solve model
# ============================================================

def simulate_model(params, time_points):

    c0, lam, kp, ke = params

    y0 = [1.0, 0.0, 0.0, 0.0]

    time_points = np.asarray(time_points, dtype=float)

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
# 3. Load ONLY Fit Ready
# ============================================================

file_name = "data/CRISPR_Modeling_Data.xlsx"

data = pd.read_excel(
    file_name,
    sheet_name="Fit Ready"
)


# ============================================================
# 4. Select Figure 2b
# ============================================================

fig2b = data[
    (data["experiment_id"] == "Figure_2b") &
    (data["cell_type"] == "neuron")
].copy()


print("====================================")
print("FIGURE 2b DATA")
print("====================================")

print(f"Total Figure 2b observations: {len(fig2b)}")

print("\nGuides:")
print(fig2b["guide"].value_counts())

print("\nTime points by guide:")

for guide in sorted(fig2b["guide"].unique()):

    guide_data = fig2b[
        fig2b["guide"] == guide
    ]

    print(
        guide,
        sorted(guide_data["time_days"].unique())
    )


# ============================================================
# 5. Lambda fixed from Cas9 abundance analysis
# ============================================================

LAMBDA_FIXED = 0.078267


# ============================================================
# 6. Residual function
# ============================================================

def residuals_fixed_lambda(params, time_data, E_data):

    c0, kp, ke = params

    full_params = [
        c0,
        LAMBDA_FIXED,
        kp,
        ke
    ]

    # solve_ivp requires unique evaluation times
    unique_times = np.sort(
        np.unique(time_data)
    )

    solution = simulate_model(
        full_params,
        unique_times
    )

    E_unique = solution.y[3]

    prediction_lookup = dict(
        zip(unique_times, E_unique)
    )

    # Each replicate receives the model prediction
    # corresponding to its experimental time point
    E_model = np.array([
        prediction_lookup[t]
        for t in time_data
    ])

    return E_model - E_data


# ============================================================
# 7. Function to fit one guide
# ============================================================

def fit_guide(guide_name):

    print("\n")
    print("=" * 60)
    print(f"FITTING GUIDE: {guide_name}")
    print("=" * 60)

    guide_data = fig2b[
        fig2b["guide"] == guide_name
    ].copy()

    guide_data = guide_data.sort_values(
        ["time_days", "replicate"]
    )


    # --------------------------------------------------------
    # Summary for plotting
    # --------------------------------------------------------

    summary = (
        guide_data
        .groupby("time_days")["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    print("\nExperimental summary:")
    print(summary)


    # Replicate-level observations used for fitting
    time_data = guide_data[
        "time_days"
    ].to_numpy(dtype=float)

    E_data = guide_data[
        "value"
    ].to_numpy(dtype=float)


    print(
        f"\nNumber of observations = {len(E_data)}"
    )


    # --------------------------------------------------------
    # Fit parameters
    # --------------------------------------------------------

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

    fitted_params = [
        c0_fit,
        LAMBDA_FIXED,
        kp_fit,
        ke_fit
    ]


    # --------------------------------------------------------
    # Predictions for replicate observations
    # --------------------------------------------------------

    unique_times = np.sort(
        np.unique(time_data)
    )

    solution = simulate_model(
        fitted_params,
        unique_times
    )

    E_unique = solution.y[3]

    prediction_lookup = dict(
        zip(unique_times, E_unique)
    )

    E_prediction = np.array([
        prediction_lookup[t]
        for t in time_data
    ])


    # --------------------------------------------------------
    # Repair timescale
    # --------------------------------------------------------

    repair_timescale = 1.0 / (
        kp_fit + ke_fit
    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            (E_data - E_prediction) ** 2
        )
    )


    # --------------------------------------------------------
    # AIC
    # --------------------------------------------------------

    n = len(E_data)

    # c0, kp and ke are estimated.
    # lambda is fixed independently.
    k = 3

    rss = np.sum(
        (E_data - E_prediction) ** 2
    )

    aic = (
        n * np.log(rss / n)
        + 2 * k
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nFitted parameters:")

    print(
        f"c0     = {c0_fit:.6f} per day"
    )

    print(
        f"lambda = {LAMBDA_FIXED:.6f} per day (fixed)"
    )

    print(
        f"kp     = {kp_fit:.6f} per day"
    )

    print(
        f"ke     = {ke_fit:.6f} per day"
    )

    print(
        f"Repair timescale = "
        f"{repair_timescale:.4f} days"
    )

    print(
        f"RMSE = {rmse:.6f}"
    )

    print(
        f"AIC = {aic:.4f}"
    )


    if np.isclose(
        c0_fit,
        upper_bounds[0],
        rtol=1e-4
    ):

        print(
            "WARNING: c0 reached its upper fitting bound."
        )


    # --------------------------------------------------------
    # Smooth model trajectory
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DNA conservation check
    # --------------------------------------------------------

    total_DNA = U + B + P + E

    conservation_error = np.max(
        np.abs(total_DNA - 1)
    )

    print(
        f"Maximum conservation error = "
        f"{conservation_error:.3e}"
    )


    # --------------------------------------------------------
    # Plot data vs model
    # --------------------------------------------------------

    time_mean = summary[
        "time_days"
    ].to_numpy(dtype=float)

    E_mean = summary[
        "mean"
    ].to_numpy(dtype=float)

    E_sd = summary[
        "std"
    ].to_numpy(dtype=float)


    plt.figure(figsize=(9, 6))


    # Individual replicates
    plt.scatter(
        time_data,
        E_data,
        alpha=0.35,
        label="Individual replicates"
    )


    # Mean +/- SD
    plt.errorbar(
        time_mean,
        E_mean,
        yerr=E_sd,
        fmt="o",
        capsize=4,
        label="Experimental mean ± SD"
    )


    # Model
    plt.plot(
        t_smooth,
        E,
        linewidth=2,
        label="Slow-repair model"
    )


    plt.xlabel("Time (days)")
    plt.ylabel("Edited fraction E(t)")

    plt.title(
        f"Figure 2b - {guide_name}: "
        f"Slow-Repair Model"
    )

    plt.ylim(0, 1)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()


    safe_name = guide_name.replace(
        " ",
        "_"
    )

    plt.savefig(
        f"plots/figure2b/figure2b_{safe_name}_fit.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # Residual plot
    # --------------------------------------------------------

    residual_values = (
        E_data - E_prediction
    )


    plt.figure(figsize=(8, 5))

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.scatter(
        time_data,
        residual_values,
        alpha=0.65
    )

    plt.xlabel("Time (days)")

    plt.ylabel(
        "Residual (Experimental - Model)"
    )

    plt.title(
        f"Figure 2b - {guide_name}: Residuals"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"plots/figure2b/figure2b_{safe_name}_residuals.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # Return results for final comparison
    # --------------------------------------------------------

    return {
        "guide": guide_name,
        "n": n,
        "c0": c0_fit,
        "lambda": LAMBDA_FIXED,
        "kp": kp_fit,
        "ke": ke_fit,
        "repair_timescale": repair_timescale,
        "rmse": rmse,
        "aic": aic,
        "c0_at_bound": np.isclose(
            c0_fit,
            upper_bounds[0],
            rtol=1e-4
        )
    }


# ============================================================
# 8. Fit every Figure 2b guide separately
# ============================================================

guides = sorted(
    fig2b["guide"]
    .dropna()
    .unique()
)


all_results = []

for guide in guides:

    result = fit_guide(guide)

    all_results.append(result)


# ============================================================
# 9. Final comparison table
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n")
print("=" * 70)
print("FIGURE 2b FINAL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# Save results for later use
results_df.to_csv(
    "plots/figure2b/figure2b_slow_repair_results.csv",
    index=False
)


print(
    "\nSaved parameter comparison to:"
)

print(
    "figure2b_slow_repair_results.csv"
)