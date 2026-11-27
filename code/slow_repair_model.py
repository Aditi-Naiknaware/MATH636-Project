import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------
# Slow Repair CRISPR-Cas9 Model
# ---------------------------------------------------------

def slow_repair_ode(t, y, c0, lam, kp, ke):
    """
    ODE system for the slow DNA repair model.

    States:
        U = uncut DNA
        B = broken DNA
        P = precisely repaired DNA
        E = permanently edited DNA

    Parameters:
        c0  = initial Cas9 cutting rate
        lam = Cas9 activity decay rate
        kp  = precise repair rate
        ke  = error-prone repair rate
    """

    U, B, P, E = y

    # Cas9 cutting activity decreases exponentially with time
    c_t = c0 * np.exp(-lam * t)

    # ODE equations
    dUdt = -c_t * U
    dBdt = c_t * U - (kp + ke) * B
    dPdt = kp * B
    dEdt = ke * B

    return [dUdt, dBdt, dPdt, dEdt]


def simulate_slow_repair(params, t_eval):
    """
    Solve the slow-repair model for a given set of parameters.
    """

    c0, lam, kp, ke = params

    # Initially all target DNA is uncut
    y0 = [1.0, 0.0, 0.0, 0.0]

    solution = solve_ivp(
        slow_repair_ode,
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        args=(c0, lam, kp, ke),
        method="RK45"
    )

    if not solution.success:
        raise RuntimeError("ODE solver failed: " + solution.message)

    return solution


# ---------------------------------------------------------
# Initial test
# ---------------------------------------------------------

# These are ONLY initial test values.
# Later we will estimate them from the experimental data.
c0 = 0.5
lam = 0.1
kp = 0.05
ke = 0.10

params = [c0, lam, kp, ke]

# Simulate for 16 days
t = np.linspace(0, 16, 200)

solution = simulate_slow_repair(params, t)

U = solution.y[0]
B = solution.y[1]
P = solution.y[2]
E = solution.y[3]


# ---------------------------------------------------------
# Check DNA conservation
# ---------------------------------------------------------

total_DNA = U + B + P + E

print("Maximum conservation error:")
print(np.max(np.abs(total_DNA - 1.0)))

print("\nFinal state:")
print(f"U(16) = {U[-1]:.4f}")
print(f"B(16) = {B[-1]:.4f}")
print(f"P(16) = {P[-1]:.4f}")
print(f"E(16) = {E[-1]:.4f}")

# Repair timescale from teammate's model
repair_timescale = 1 / (kp + ke)

print(f"\nRepair timescale = {repair_timescale:.4f} days")


# ---------------------------------------------------------
# Plot model states
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(t, U, label="Uncut DNA U(t)")
plt.plot(t, B, label="Broken DNA B(t)")
plt.plot(t, P, label="Precisely Repaired P(t)")
plt.plot(t, E, label="Permanently Edited E(t)")

plt.xlabel("Time (days)")
plt.ylabel("Fraction of target DNA")
plt.title("Slow-Repair CRISPR-Cas9 Model")

plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/figure2a/slow_repair_states.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved slow_repair_states.png")

# ---------------------------------------------------------
# Plot Cas9 cutting activity
# ---------------------------------------------------------

cas9_activity = c0 * np.exp(-lam * t)

plt.figure(figsize=(8, 5))

plt.plot(t, cas9_activity)

plt.xlabel("Time (days)")
plt.ylabel("Effective cutting rate c(t)")
plt.title("Decay of Cas9 Cutting Activity")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/cas9/cas9_activity.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved cas9_activity.png")