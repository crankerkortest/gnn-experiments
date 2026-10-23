# Generator and network robustness extension

`robustness_experiment.py` evaluates whether the synthetic-ground-truth
findings persist when one aspect of the generator or graph is changed at a
time. It runs the baseline and six paired conditions over graph seeds
2026–2030, trains the seed-0 GraphSAGE model under the original optimization
protocol, and explains 30 risk-stratified test policyholders per graph and
condition.

The six conditions remove the provider causal path, halve all non-intercept
effects, double the recovered outcome-noise term, add three prespecified
nonlinear interactions, add an unobserved Beta(2,2) cause, or add 600
semantically plausible but noncausal relationships (25% of the baseline unique
relationship count). Modified generators are intercept-calibrated to the
baseline target mean. Exact counterfactual effects are recomputed under each
modified generator while the other inputs and exogenous residual are held
fixed.

Run from the repository root:

```bash
python analysis/robustness_experiment.py
```

For a short validation run:

```bash
python analysis/robustness_experiment.py --smoke
```

Outputs are written to `analysis/robustness_experiment_output/`. The summary
files contain graph-level means and standard deviations for predictive error,
intervention-effect recovery, Integrated Gradients feature recovery, and
GNNExplainer causal-edge precision, recall, and Jaccard overlap. The graph- and
node-level CSV files retain the underlying measurements.

Every one of the 1,050 explained policyholder-condition cases is evaluated by
both GNNExplainer and Integrated Gradients. The number therefore denotes paired
cases, not the sum of separate calls to the two methods. For a nominal top-k
budget, the selected structural set contains `min(k, available_local_edges)`
relationships; precision uses this actual size. Intervention residuals with
absolute magnitude below `1e-12` are normalized to zero, and direction
agreement is evaluated only where the generator-defined effect is nonzero.
