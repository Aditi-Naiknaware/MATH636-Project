# Mathematical Modeling of Prolonged CRISPR-Cas9 Editing

Python implementation of a slow DNA-repair model for prolonged CRISPR-Cas9
editing in nondividing human neurons.

## Project Structure

- `code/` - ODE simulation, parameter fitting, and sensitivity analysis
- `data/` - experimental data (not included in the repository)
- `results/` - fitted parameter and sensitivity results
- `plots/` - generated figures

## Slow-Repair Model

The model tracks four DNA states:

- U(t): uncut DNA
- B(t): broken DNA
- P(t): precisely repaired DNA
- E(t): permanently edited DNA

The model is fitted to experimental CRISPR editing data using SciPy.

## Running the Analysis

Run from the project root:

```bash
python code/slow_repair_model.py
python code/fit_cas9_decay.py
python code/fit_slow_repair.py
python code/fit_slow_repair_figure2b.py
python code/sensitivity_analysis.py
python code/sensitivity_all_guides.py
