import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp


# ============================================================
# 1. Slow-repair ODE model
# ============================================================

def slow_repair_ode(t, y, c0, lam, kp, ke):

    U, B, P, E = y

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

    solution = solve_ivp(
        slow_repair_ode,
        [0, np.max(time_points)],
        y0,
        args=(c0, lam, kp, ke),
        t_eval=time_points,
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution


# ============================================================
# 3. Load fitted Figure 2b parameters
# ============================================================

results = pd.read_csv(
    "results/figure2b_slow_repair_results.csv"
)

print("====================================")
print("FIGURE 2b FITTED PARAMETERS")
print("====================================")

print(results)


# ============================================================
# 4. Local normalized sensitivity function
# ============================================================

def calculate_sensitivity(params, parameter_name, time=30.0):

    """
    Calculate normalized local sensitivity:

        S_p = (p / E) * (dE / dp)

    using a central finite difference.
    """

    epsilon = 0.01

    parameter_names = [
        "c0",
        "lambda",
        "kp",
        "ke"
    ]

    parameter_index = parameter_names.index(
        parameter_name
    )

    baseline = np.array(
        params,
        dtype=float
    )

    # Baseline prediction
    baseline_solution = simulate_model(
        baseline,
        np.array([time])
    )

    E_baseline = baseline_solution.y[3][-1]


    # --------------------------------------------------------
    # Increase parameter by 1%
    # --------------------------------------------------------

    plus = baseline.copy()

    plus[parameter_index] *= (
        1 + epsilon
    )


    # --------------------------------------------------------
    # Decrease parameter by 1%
    # --------------------------------------------------------

    minus = baseline.copy()

    minus[parameter_index] *= (
        1 - epsilon
    )


    E_plus = simulate_model(
        plus,
        np.array([time])
    ).y[3][-1]


    E_minus = simulate_model(
        minus,
        np.array([time])
    ).y[3][-1]


    # --------------------------------------------------------
    # Central finite-difference derivative
    # --------------------------------------------------------

    p = baseline[
        parameter_index
    ]

    derivative = (
        E_plus - E_minus
    ) / (
        2 * epsilon * p
    )


    # --------------------------------------------------------
    # Normalized sensitivity
    # --------------------------------------------------------

    sensitivity = (
        p / E_baseline
    ) * derivative


    return sensitivity, E_baseline


# ============================================================
# 5. Calculate sensitivity for every guide
# ============================================================

all_sensitivity_results = []


for _, row in results.iterrows():

    guide = row["guide"]

    params = [
        row["c0"],
        row["lambda"],
        row["kp"],
        row["ke"]
    ]


    print("\n")
    print("=" * 60)
    print(f"GUIDE: {guide}")
    print("=" * 60)


    guide_result = {
        "guide": guide
    }


    for parameter in [
        "c0",
        "lambda",
        "kp",
        "ke"
    ]:

        sensitivity, E30 = calculate_sensitivity(
            params,
            parameter,
            time=30.0
        )

        guide_result[
            f"S_{parameter}"
        ] = sensitivity


        print(
            f"{parameter:8s}: "
            f"{sensitivity:.6f}"
        )


    guide_result["E30"] = E30

    all_sensitivity_results.append(
        guide_result
    )


# ============================================================
# 6. Create final sensitivity table
# ============================================================

sensitivity_df = pd.DataFrame(
    all_sensitivity_results
)


print("\n")
print("=" * 80)
print("NORMALIZED SENSITIVITY AT DAY 30")
print("=" * 80)

print(
    sensitivity_df.to_string(
        index=False
    )
)


# ============================================================
# 7. Determine most influential parameter
# ============================================================

parameter_columns = [
    "S_c0",
    "S_lambda",
    "S_kp",
    "S_ke"
]


print("\n")
print("=" * 80)
print("MOST INFLUENTIAL PARAMETER BY GUIDE")
print("=" * 80)


for _, row in sensitivity_df.iterrows():

    absolute_values = {
        parameter: abs(row[parameter])
        for parameter in parameter_columns
    }

    most_sensitive = max(
        absolute_values,
        key=absolute_values.get
    )

    print(
        f"{row['guide']:10s}: "
        f"{most_sensitive} "
        f"(|S| = "
        f"{absolute_values[most_sensitive]:.4f})"
    )


# ============================================================
# 8. Average absolute sensitivity across guides
# ============================================================

print("\n")
print("=" * 80)
print("MEAN ABSOLUTE SENSITIVITY ACROSS GUIDES")
print("=" * 80)


mean_sensitivity = {}


for parameter in parameter_columns:

    mean_value = np.mean(
        np.abs(
            sensitivity_df[parameter]
        )
    )

    mean_sensitivity[
        parameter
    ] = mean_value

    print(
        f"{parameter:10s}: "
        f"{mean_value:.6f}"
    )


# ============================================================
# 9. Overall ranking
# ============================================================

ranking = sorted(
    mean_sensitivity.items(),
    key=lambda x: x[1],
    reverse=True
)


print("\nOverall sensitivity ranking:")

for rank, (parameter, value) in enumerate(
    ranking,
    start=1
):

    print(
        f"{rank}. {parameter} "
        f"(|S| mean = {value:.6f})"
    )


# ============================================================
# 10. Save results
# ============================================================

sensitivity_df.to_csv(
    "results/figure2b_sensitivity_results.csv",
    index=False
)


print(
    "\nSaved results to "
    "figure2b_sensitivity_results.csv"
)