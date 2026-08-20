# Explainable GNN experiments for health-insurance risk assessment

This repository package contains the exact Python workflow and machine-readable
outputs used for the manuscript experiments. The task is node-level regression
of a synthetic policyholder health-risk score bounded to `[0,1]`. Claims,
providers, and health-insurance objects supply relational context to the graph
models.

## Experimental design

- Data and split seeds: 2026 to 2035, giving 10 independently generated graphs
  and policyholder splits.
- Model seeds: 0 to 4 for every evaluated model, giving 50 fits per model
  specification.
- Population explanations: the seed-0 GraphSAGE model explains all 120 test
  policyholders in each graph, giving 1,200 cases.
- Matched controls: 20 same-size random samples per explained case.
- Stability: graph seeds 2026 to 2030, model seeds 0 to 2, 20 risk-stratified
  test policyholders per graph, 10 noise repetitions, 60 GNNExplainer epochs for
  original and perturbed explanations, and top-k sizes 2, 4, 6, and 8.
- Synthetic causal validation: nine controlled direct interventions for each of
  the 1,200 test cases, with exogenous noise and nonintervened values fixed.

The independently generated graph is the inferential unit for confidence
intervals and paired tests. Model seeds quantify optimization variability within
each graph.

## Repository contents

- `analysis/health_risk_experiment.py`: synthetic generator, graph models,
  training utilities, GNNExplainer, Integrated Gradients, and shared functions.
- `analysis/extended_validation_experiment.py`: multigraph predictive,
  explanation, matched-control, stability, and causal-intervention experiments.
- `analysis/extract_multigraph_attributions.py`: raw six-feature Integrated
  Gradients, completeness, and attribution timing summaries.
- `analysis/postprocess_extended_results.py`: graph-level statistical summaries
  used in the manuscript.
- `analysis/extended_experiment_output/`: CSV and JSON results produced by the
  workflow and reported in the manuscript.
- `requirements.txt`: recorded package versions.

All four scripts are required because the main experiment imports the generator
and model utilities, while the two final scripts derive attribution and
manuscript-level summaries from the primary outputs.

## Environment

The recorded experiment environment was Python 3.12.13 with:

- PyTorch 2.5.1 CPU build
- PyTorch Geometric 2.7.0
- NumPy 1.26.4
- SciPy 1.17.1
- scikit-learn 1.6.1

Create and activate a Python 3.12 virtual environment, then install the pinned
dependencies:

```bash
python -m pip install -r requirements.txt
```

If a platform-specific PyTorch wheel is needed, install the PyTorch 2.5.1 CPU
wheel for that platform first, then rerun the requirements command.

## Reproduce the experiments

Run the following commands from the repository root and in this order:

```bash
PYTHONPATH=analysis python analysis/extended_validation_experiment.py
PYTHONPATH=analysis python analysis/extract_multigraph_attributions.py
PYTHONPATH=analysis python analysis/postprocess_extended_results.py
```

The scripts write to `analysis/extended_experiment_output/`. Existing result
files are included so that manuscript values can be checked without rerunning
the full explanation workload. Rerunning with the pinned environment and
recorded seeds replaces those outputs deterministically, apart from small
platform-dependent floating-point and timing differences.

