import numpy as np
import matplotlib.pyplot as plt

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
# 3. Baseline fitted parameters
#    Figure 2b - B2Mg1
# ============================================================

c0_base = 1.222829
lambda_base = 0.078267
kp_base = 0.024016
ke_base = 0.045620


baseline_params = {
    "c0": c0_base,
    "lambda": lambda_base,
    "kp": kp_base,
    "ke": ke_base
}


print("====================================")
print("BASELINE PARAMETERS")
print("====================================")

for name, value in baseline_params.items():
    print(f"{name} = {value:.6f}")


# ============================================================
# 4. Time range
# ============================================================

t = np.linspace(
    0,
    30,
    500
)


# ============================================================
# 5. Sensitivity levels
# ============================================================

multipliers = [
    0.5,
    1.0,
    1.5
]


# ============================================================
# 6. Function for one-parameter sensitivity analysis
# ============================================================

def sensitivity_plot(parameter_name):

    plt.figure(figsize=(9, 6))

    for multiplier in multipliers:

        # Start with baseline values
        params = baseline_params.copy()

        # Change only one parameter
        params[parameter_name] *= multiplier

        parameter_vector = [
            params["c0"],
            params["lambda"],
            params["kp"],
            params["ke"]
        ]

        solution = simulate_model(
            parameter_vector,
            t
        )

        E = solution.y[3]

        plt.plot(
            t,
            E,
            linewidth=2,
            label=(
                f"{multiplier:.1f}x "
                f"({params[parameter_name]:.4f})"
            )
        )

    plt.xlabel("Time (days)")
    plt.ylabel("Edited fraction E(t)")

    plt.title(
        f"Sensitivity of E(t) to {parameter_name}"
    )

    plt.ylim(0, 1)

    plt.legend(
        title=f"{parameter_name} multiplier"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    file_name = (
        f"plots/sensitivity/sensitivity_{parameter_name}.png"
    )

    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved {file_name}")


# ============================================================
# 7. Run sensitivity analysis
# ============================================================

for parameter in [
    "c0",
    "lambda",
    "kp",
    "ke"
]:

    sensitivity_plot(parameter)


# ============================================================
# 8. Numerical sensitivity at important time points
# ============================================================

important_times = np.array([
    2.0,
    4.0,
    9.0,
    16.0,
    30.0
])


print("\n")
print("====================================")
print("NUMERICAL SENSITIVITY")
print("====================================")


for parameter_name in [
    "c0",
    "lambda",
    "kp",
    "ke"
]:

    print("\n")
    print(
        f"Parameter: {parameter_name}"
    )

    print(
        "-" * 60
    )

    for multiplier in multipliers:

        params = baseline_params.copy()

        params[parameter_name] *= multiplier

        parameter_vector = [
            params["c0"],
            params["lambda"],
            params["kp"],
            params["ke"]
        ]

        solution = simulate_model(
            parameter_vector,
            important_times
        )

        E = solution.y[3]

        print(
            f"{multiplier:.1f}x "
            f"{parameter_name}: "
            f"{np.round(E, 4)}"
        )


# ============================================================
# 9. Local normalized sensitivity
# ============================================================

print("\n")
print("====================================")
print("LOCAL NORMALIZED SENSITIVITY")
print("at day 30")
print("====================================")


# Small perturbation: 1%
epsilon = 0.01


baseline_vector = [
    c0_base,
    lambda_base,
    kp_base,
    ke_base
]


baseline_solution = simulate_model(
    baseline_vector,
    np.array([30.0])
)

E_baseline = baseline_solution.y[3][-1]


for parameter_name in [
    "c0",
    "lambda",
    "kp",
    "ke"
]:

    plus_params = baseline_params.copy()
    minus_params = baseline_params.copy()

    plus_params[parameter_name] *= (
        1 + epsilon
    )

    minus_params[parameter_name] *= (
        1 - epsilon
    )


    plus_vector = [
        plus_params["c0"],
        plus_params["lambda"],
        plus_params["kp"],
        plus_params["ke"]
    ]

    minus_vector = [
        minus_params["c0"],
        minus_params["lambda"],
        minus_params["kp"],
        minus_params["ke"]
    ]


    E_plus = simulate_model(
        plus_vector,
        np.array([30.0])
    ).y[3][-1]

    E_minus = simulate_model(
        minus_vector,
        np.array([30.0])
    ).y[3][-1]


    # Central finite difference
    derivative = (
        E_plus - E_minus
    ) / (
        2
        * epsilon
        * baseline_params[parameter_name]
    )


    # Normalized sensitivity:
    #
    # (p / E) * dE/dp
    #
    normalized_sensitivity = (
        baseline_params[parameter_name]
        / E_baseline
    ) * derivative


    print(
        f"{parameter_name:8s}: "
        f"{normalized_sensitivity:.6f}"
    )


print("\nSensitivity analysis complete.")