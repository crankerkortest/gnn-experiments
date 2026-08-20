"""Extract graph-level Integrated Gradients summaries and explanation runtime.

The main extended run retained rank-concordance values but not the six raw
target-row attributions. This compact reproducibility pass retrains only the
seed-0 GraphSAGE model for each already specified graph and records those
quantities without changing the experimental design.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from extended_validation_experiment import DATA_SEEDS
from health_risk_experiment import (
    ExperimentConfig,
    POLICYHOLDER_FEATURES,
    explain_local_subgraph,
    generate_health_graph,
    integrated_gradients_subgraph,
    train_gnn,
)


def main() -> None:
    output = Path("analysis/extended_experiment_output")
    rows = []
    cfg0 = ExperimentConfig(model_seeds=(0,), explainer_epochs=60)
    for data_seed in DATA_SEEDS:
        cfg = replace(cfg0, data_seed=data_seed)
        data, _ = generate_health_graph(cfg)
        model, _, _ = train_gnn(data, "GraphSAGE", 0, cfg)
        for position, node in enumerate(torch.where(data.test_mask)[0].tolist(), start=1):
            explanation_seed = 100_000 + 1000 * (data_seed - DATA_SEEDS[0]) + node
            base = explain_local_subgraph(
                model, data, node, cfg, explanation_seed=explanation_seed, epochs=60
            )
            ig_start = time.perf_counter()
            attribution = integrated_gradients_subgraph(
                model,
                base["sub_x"],
                base["sub_edge_index"],
                int(base["local_target"]),
                cfg.integrated_gradient_steps,
            )
            ig_seconds = time.perf_counter() - ig_start
            values = attribution[int(base["local_target"]), :6].abs().detach().numpy()
            with torch.no_grad():
                baseline_prediction = float(
                    model(torch.zeros_like(base["sub_x"]), base["sub_edge_index"])[
                        int(base["local_target"])
                    ]
                )
            row = {
                "data_seed": data_seed,
                "model_seed": 0,
                "node_index": node,
                "explanation_seconds": float(base["explainer_seconds"] + ig_seconds),
                "ig_completeness_residual": float(
                    abs(
                        (float(base["origin"]) - baseline_prediction)
                        - float(attribution.sum())
                    )
                ),
            }
            for index, name in enumerate(POLICYHOLDER_FEATURES):
                row[f"ig_{index}"] = float(values[index])
                row[f"feature_{index}"] = name
            rows.append(row)
        print(f"attributions complete for data seed {data_seed}", flush=True)

    fields = list(rows[0].keys())
    with (output / "multigraph_feature_attributions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    graph_summaries = []
    for data_seed in DATA_SEEDS:
        current = [row for row in rows if row["data_seed"] == data_seed]
        graph_row = {"data_seed": data_seed}
        for index, name in enumerate(POLICYHOLDER_FEATURES):
            graph_row[name] = float(np.mean([row[f"ig_{index}"] for row in current]))
        graph_row["explanation_seconds"] = float(
            np.mean([row["explanation_seconds"] for row in current])
        )
        graph_row["ig_completeness_residual"] = float(
            np.mean([row["ig_completeness_residual"] for row in current])
        )
        graph_summaries.append(graph_row)
    (output / "multigraph_attribution_graph_means.json").write_text(
        json.dumps(graph_summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
