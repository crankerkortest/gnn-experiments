# Supervisor corrections

This update applies the supervisor's numerical-zero, capped top-k, magnitude,
sign-agreement, and experiment-count clarifications.

## Reproduce the corrected summaries

Run from the repository root after the original experiment outputs are present:

```bash
python analysis/postprocess_causal_corrections.py
```

Model retraining is not required. The script reads the deposited intervention,
explanation-size, and robustness CSV files and writes corrected machine-readable
summaries.

## Conventions

- A variable intervention is a no-op when the represented input does not change
  or its generator contribution is zero.
- Residual effects with absolute magnitude below `1e-12` are normalized to exact
  zero.
- Sign agreement is calculated only for nonzero generator effects and includes
  its denominator.
- A nominal top-k structural explanation contains
  `min(k, available_local_edges)` relationships. Precision uses the actual
  selected-set size.
- The 1,050 robustness cases are paired policyholder-condition cases, each
  evaluated by both GNNExplainer and Integrated Gradients.
- The 12,000 stability rows are four size-indexed measurements from 3,000
  perturbed explanation evaluations.

## Corrected outputs

- `synthetic_causal_interventions_corrected.csv`
- `causal_intervention_graph_summary_corrected.csv`
- `causal_intervention_variable_summary_corrected.csv`
- `causal_intervention_summary_corrected.json`
- `explanation_size_sensitivity_corrected.csv`
- `top_k_selection_size_summary.csv`
- `supervisor_correction_summary.json`
- `robustness_experiment_output/provider_path_capped_top_k_summary.json`

