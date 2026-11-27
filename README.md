# Mathematical Modeling of Prolonged CRISPR-Cas9 Editing

This repository contains the computational implementation for our MATH 636 project investigating prolonged CRISPR-Cas9 editing in nondividing human cells.

The project compares two possible mechanisms that may explain prolonged editing:

1. **Slow DNA repair** — Cas9-induced DNA breaks remain unresolved for an extended period before being repaired.
2. **Repeated precise repair and recutting** — DNA is precisely repaired back to a cuttable state, allowing persistent Cas9 to cut the same target again.

The current repository contains the completed implementation and analysis pipeline for the **slow-repair model**. The recutting model will be added separately.

---

## Slow-Repair Model

The slow-repair model tracks four fractions of the target DNA population:

- \(U(t)\): uncut DNA
- \(B(t)\): DNA containing a Cas9-induced double-strand break
- \(P(t)\): precisely repaired DNA
- \(E(t)\): permanently edited DNA

The total DNA population is normalized so that

\[
U(t)+B(t)+P(t)+E(t)=1.
\]

Cas9 cutting activity is modeled as exponentially decreasing:

\[
c(t)=c_0e^{-\lambda t}.
\]

The ODE system is

\[
\frac{dU}{dt}=-c(t)U,
\]

\[
\frac{dB}{dt}=c(t)U-(k_p+k_e)B,
\]

\[
\frac{dP}{dt}=k_pB,
\]

\[
\frac{dE}{dt}=k_eB.
\]

The model assumes that precisely repaired DNA is a terminal state and therefore does **not** allow recutting from \(P(t)\).

The experimental observable used for model fitting is the permanently edited fraction \(E(t)\).

---

## Parameters

| Parameter | Meaning |
|---|---|
| \(c_0\) | Initial Cas9 cutting rate |
| \(\lambda\) | Cas9 activity-decay rate |
| \(k_p\) | Precise DNA-repair rate |
| \(k_e\) | Error-prone DNA-repair rate |

The characteristic repair timescale is

\[
\tau_{\text{repair}}=\frac{1}{k_p+k_e}.
\]

---

## Data

The model uses experimental CRISPR-Cas9 editing measurements organized in:

```text
CRISPR_Modeling_Data.xlsx
```

The workbook itself is **not included in this repository**.

To run the analysis, place a local copy at:

```text
data/CRISPR_Modeling_Data.xlsx
```

The main editing trajectories are read from the **`Fit Ready`** sheet.

The analysis currently uses:

- Figure 2a neuronal B2Mg1 editing measurements
- Figure 2b neuronal measurements for:
  - B2Mg1
  - B2Mg2
  - HSPB1g2
  - NEFLg1
- neuronal Cas9-abundance measurements to independently estimate the Cas9 decay rate \(\lambda\)

Figure 2a and Figure 2b are treated as separate experiments and are not pooled into a single trajectory.

---

## Project Structure

```text
MATH636-Project/
│
├── code/
│   ├── slow_repair_model.py
│   ├── fit_cas9_decay.py
│   ├── fit_slow_repair.py
│   ├── fit_slow_repair_figure2b.py
│   ├── sensitivity_analysis.py
│   └── sensitivity_all_guides.py
│
├── data/
│   └── CRISPR_Modeling_Data.xlsx
│       # Local only; excluded from Git
│
├── results/
│   ├── figure2b_slow_repair_results.csv
│   └── figure2b_sensitivity_results.csv
│
├── plots/
│   ├── cas9/
│   ├── figure2a/
│   ├── figure2b/
│   └── sensitivity/
│
├── .gitignore
└── README.md
```

---

## Code Overview

### `slow_repair_model.py`

Implements the four-state ODE system and performs an initial numerical simulation.

It also verifies the conservation condition

\[
U+B+P+E=1.
\]

---

### `fit_cas9_decay.py`

Fits an exponential decay model to neuronal Cas9-abundance measurements:

\[
C(t)=Ae^{-\lambda t}.
\]

The fitted decay rate is used to independently constrain \(\lambda\) in the slow-repair model.

Current estimate:

\[
\lambda \approx 0.0783\ \text{day}^{-1}.
\]

This corresponds to a Cas9 half-life of approximately

\[
8.86\text{ days}.
\]

---

### `fit_slow_repair.py`

Fits the slow-repair model to the Figure 2a neuronal B2Mg1 data.

The fitting uses all 36 replicate-level observations while keeping \(\lambda\) fixed using the independent Cas9-abundance analysis.

The script calculates:

- fitted model parameters
- repair timescale
- RMSE
- AIC
- residuals
- DNA-conservation error
- experimental-data vs. model plots
- predicted \(U(t), B(t), P(t), E(t)\) trajectories

The fitted \(c_0\) reaches its optimization upper bound in this experiment, so it should not be interpreted as a well-identified biological estimate.

---

### `fit_slow_repair_figure2b.py`

Fits the slow-repair model independently to the four Figure 2b guides:

- B2Mg1
- B2Mg2
- HSPB1g2
- NEFLg1

Each guide is treated as a separate trajectory.

The script produces:

- fitted \(c_0\), \(k_p\), and \(k_e\)
- repair timescale
- RMSE
- AIC
- residual plots
- experimental-data vs. model plots
- a combined results CSV

---

### `sensitivity_analysis.py`

Performs one-at-a-time sensitivity analysis for the Figure 2b B2Mg1 fit.

Each parameter is evaluated at:

- 50% of its fitted value
- 100% of its fitted value
- 150% of its fitted value

The analysis examines the effects of

\[
c_0,\quad\lambda,\quad k_p,\quad k_e
\]

on the predicted edited fraction \(E(t)\).

It also calculates normalized local sensitivity at day 30.

---

### `sensitivity_all_guides.py`

Calculates normalized day-30 parameter sensitivities for all four Figure 2b guides.

The normalized sensitivity is

\[
S_p=
\frac{p}{E(30)}
\frac{\partial E(30)}{\partial p}.
\]

Across the four Figure 2b guides, the current mean absolute sensitivity ranking is

\[
k_e > c_0 > k_p > \lambda.
\]

The most influential parameter is not identical for every guide:

- B2Mg1: \(k_e\)
- B2Mg2: \(k_e\)
- HSPB1g2: \(c_0\)
- NEFLg1: \(c_0\)

---

## Running the Analysis

The analysis requires Python with:

- NumPy
- pandas
- SciPy
- Matplotlib
- openpyxl

Make sure the dataset is located at:

```text
data/CRISPR_Modeling_Data.xlsx
```

Run all commands from the repository root.

Individual scripts can be run with:

```bash
python code/slow_repair_model.py
python code/fit_cas9_decay.py
python code/fit_slow_repair.py
python code/fit_slow_repair_figure2b.py
python code/sensitivity_analysis.py
python code/sensitivity_all_guides.py
```

To run the complete slow-repair pipeline sequentially:

```bash
python code/slow_repair_model.py && \
python code/fit_cas9_decay.py && \
python code/fit_slow_repair.py && \
python code/fit_slow_repair_figure2b.py && \
python code/sensitivity_analysis.py && \
python code/sensitivity_all_guides.py
```

Using `&&` ensures that the pipeline stops if one of the preceding analyses fails.

---

## Current Results

### Figure 2a

Using the independently constrained

\[
\lambda=0.078267\ \text{day}^{-1},
\]

the Figure 2a B2Mg1 fit gives approximately:

| Quantity | Value |
|---|---:|
| \(k_p\) | 0.0871 day\(^{-1}\) |
| \(k_e\) | 0.1962 day\(^{-1}\) |
| Repair timescale | 3.53 days |
| RMSE | 0.0262 |
| Number of observations | 36 |

The fitted \(c_0\) reaches the imposed upper bound and is therefore not treated as a well-identified parameter.

### Figure 2b

| Guide | \(c_0\) | \(k_p\) | \(k_e\) | Repair timescale | RMSE |
|---|---:|---:|---:|---:|---:|
| B2Mg1 | 1.2228 | 0.0240 | 0.0456 | 14.36 days | 0.0543 |
| B2Mg2 | 0.1689 | 0.0450 | 0.0815 | 7.91 days | 0.0655 |
| HSPB1g2 | 0.1532 | 0.0488 | 0.1143 | 6.13 days | 0.0644 |
| NEFLg1 | 0.1123 | 0.0461 | 0.1057 | 6.59 days | 0.0711 |

These parameter values should be interpreted as fitted effective parameters under the assumptions of the slow-repair model.

---

## Model Comparison

The overall project is designed to compare the slow-repair mechanism against a second model involving **precise repair followed by repeated Cas9 recutting**.

Once the recutting model is implemented, both mechanisms will be evaluated on the same experimental observations using metrics such as:

- RMSE
- residual analysis
- Akaike Information Criterion (AIC)

This will allow us to determine whether the available editing measurements favor one mechanism or whether additional biological measurements are required to distinguish them.

---

## Reference

Ramadoss, G. N., Namaganda, S. J., Kumar, M. M., et al. (2025). *Characterizing and controlling CRISPR repair outcomes in nondividing human cells*. Nature Communications, 16, 9883.