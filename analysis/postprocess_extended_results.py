"""Create manuscript-ready, graph-level summaries from extended outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from extended_validation_experiment import (
    DATA_SEEDS,
    graph_level_matched_test,
    scalar_summary,
)
from health_risk_experiment import POLICYHOLDER_FEATURES


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def graph_level_feature_summary(rows):
    output = {}
    for index, name in enumerate(POLICYHOLDER_FEATURES):
        graph_means = []
        for seed in DATA_SEEDS:
            values = [float(row[f"ig_{index}"]) for row in rows if int(row["data_seed"]) == seed]
            graph_means.append(float(np.mean(values)))
        output[name] = scalar_summary(graph_means)
    for metric in ("explanation_seconds", "ig_completeness_residual"):
        graph_means = []
        for seed in DATA_SEEDS:
            values = [float(row[metric]) for row in rows if int(row["data_seed"]) == seed]
            graph_means.append(float(np.mean(values)))
        output[metric] = scalar_summary(graph_means)
    return output


def graph_level_population_metric(rows, metric):
    values = []
    for seed in DATA_SEEDS:
        current = [float(row[metric]) for row in rows if int(row["data_seed"]) == seed]
        values.append(float(np.mean(current)))
    return scalar_summary(values)


def graph_level_feature_causal(rows):
    output = {}
    for metric in ("feature_rank_spearman", "feature_top2_causal_overlap"):
        graph_means = []
        for seed in DATA_SEEDS:
            current = [float(row[metric]) for row in rows if int(row["data_seed"]) == seed]
            graph_means.append(float(np.mean(current)))
        null = 0.0 if metric == "feature_rank_spearman" else 1.0 / 3.0
        test = wilcoxon(np.asarray(graph_means) - null, alternative="two-sided", zero_method="wilcox")
        output[metric] = {
            **scalar_summary(graph_means),
            "null_value": null,
            "two_sided_wilcoxon_statistic": float(test.statistic),
            "two_sided_p_value": float(test.pvalue),
        }
    return output


def main():
    folder = Path("analysis/extended_experiment_output")
    population = read_csv(folder / "population_explanations.csv")
    sizes = read_csv(folder / "explanation_size_sensitivity.csv")
    attributions = read_csv(folder / "multigraph_feature_attributions.csv")
    size_four = [row for row in sizes if int(row["top_k"]) == 4]

    causal_edge_test = graph_level_matched_test(
        size_four, "causal_edge_precision", "random_causal_edge_precision"
    )
    output = {
        "feature_attributions": graph_level_feature_summary(attributions),
        "top_four_metrics": {
            metric: graph_level_population_metric(population, metric)
            for metric in (
                "fidelity_plus",
                "fidelity_minus",
                "characterization",
                "sparsity",
                "causal_edge_precision",
                "causal_edge_recall",
            )
        },
        "causal_edge_precision_test": causal_edge_test,
        "feature_causal_validation": graph_level_feature_causal(population),
    }
    (folder / "manuscript_ready_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
