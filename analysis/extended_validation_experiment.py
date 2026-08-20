"""Extended validation for the health-insurance GraphSAGE study.

This script adds sampling/split variability, explanation-size sensitivity,
matched-control inference, multi-model stability, and validation against the
known intervention structure of the synthetic data-generating mechanism.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr, t as student_t, wilcoxon

from health_risk_experiment import (
    ExperimentConfig,
    GraphRegressor,
    POLICYHOLDER_FEATURES,
    edge_mask_from_pairs,
    explain_local_subgraph,
    generate_health_graph,
    integrated_gradients_subgraph,
    jaccard,
    run_tabular_baselines,
    seed_all,
    shuffled_edge_index,
    train_gnn,
)


DATA_SEEDS = tuple(range(2026, 2036))
MODEL_SEEDS = (0, 1, 2, 3, 4)
EXPLANATION_MODEL_SEEDS = (0, 1, 2)
STABILITY_DATA_SEEDS = DATA_SEEDS[:5]
TOP_K_VALUES = (2, 4, 6, 8)
STABILITY_REPETITIONS = 10
STABILITY_NODES_PER_GRAPH = 20
RANDOM_CONTROL_TRIALS = 20


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scalar_summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    crit = float(student_t.ppf(0.975, len(arr) - 1)) if len(arr) > 1 else 0.0
    half = crit * sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {
        "n": int(len(arr)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half_width": half,
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }


def nested_predictive_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    summaries: List[Dict[str, object]] = []
    for model, model_rows in grouped.items():
        per_graph: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for row in model_rows:
            per_graph[int(row["data_seed"])].append(row)
        summary: Dict[str, object] = {
            "model": model,
            "n_data_seeds": len(per_graph),
            "n_training_runs": len(model_rows),
        }
        for metric in ("rmse", "mae", "r2"):
            graph_means = [
                float(np.mean([float(item[metric]) for item in items]))
                for _, items in sorted(per_graph.items())
            ]
            stats = scalar_summary(graph_means)
            for key, value in stats.items():
                summary[f"{metric}_{key}"] = value
        times = np.asarray([float(item["training_seconds"]) for item in model_rows])
        summary.update(
            {
                "training_seconds_median": float(np.median(times)),
                "training_seconds_q1": float(np.quantile(times, 0.25)),
                "training_seconds_q3": float(np.quantile(times, 0.75)),
            }
        )
        summaries.append(summary)
    return sorted(summaries, key=lambda row: float(row["rmse_mean"]))


def paired_graph_comparison(
    rows: Sequence[Dict[str, object]], model_a: str, model_b: str, metric: str
) -> Dict[str, object]:
    graph_model: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in rows:
        if row["model"] in {model_a, model_b}:
            graph_model[(int(row["data_seed"]), str(row["model"]))].append(float(row[metric]))
    a = []
    b = []
    for seed in DATA_SEEDS:
        a.append(float(np.mean(graph_model[(seed, model_a)])))
        b.append(float(np.mean(graph_model[(seed, model_b)])))
    diff = np.asarray(a) - np.asarray(b)
    test = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "model_a_graph_mean": float(np.mean(a)),
        "model_b_graph_mean": float(np.mean(b)),
        "paired_difference_a_minus_b": scalar_summary(diff.tolist()),
        "two_sided_wilcoxon_statistic": float(test.statistic),
        "two_sided_p_value": float(test.pvalue),
    }


def selected_pairs_for_k(base: Dict[str, object], k: int) -> List[Tuple[int, int]]:
    ranked = sorted(base["pair_scores"].items(), key=lambda item: item[1], reverse=True)
    return [pair for pair, _ in ranked[: min(k, len(ranked))]]


def globalize_pairs(base: Dict[str, object], pairs: Iterable[Tuple[int, int]]) -> set[Tuple[int, int]]:
    subset = base["subset"].tolist()
    return {
        tuple(sorted((int(subset[int(a)]), int(subset[int(b)]))))
        for a, b in pairs
    }


def causal_edge_set(node: int, cfg: ExperimentConfig, metadata: Dict[str, object]) -> set[Tuple[int, int]]:
    claim = cfg.num_policyholders + node
    policy = cfg.num_policyholders + cfg.num_claims + int(metadata["policy_for_holder"][node])
    provider = (
        cfg.num_policyholders
        + cfg.num_claims
        + cfg.num_health_policies
        + int(metadata["provider_for_claim"][node])
    )
    return {
        tuple(sorted((node, claim))),
        tuple(sorted((node, policy))),
        tuple(sorted((claim, provider))),
        tuple(sorted((claim, policy))),
    }


def prediction_for_pairs(
    model: GraphRegressor,
    base: Dict[str, object],
    pairs: Sequence[Tuple[int, int]],
    keep_selected: bool,
) -> float:
    keep = edge_mask_from_pairs(base["sub_edge_index"], pairs, keep_selected=keep_selected)
    with torch.no_grad():
        return float(model(base["sub_x"], base["sub_edge_index"][:, keep])[int(base["local_target"])])


def direct_feature_causal_effects(data, node: int) -> np.ndarray:
    x = data.x[node, :6].numpy().astype(float)
    contributions = np.array(
        [
            1.15 * x[0],
            1.55 * (1.0 - x[1]),
            0.85 * x[2],
            0.75 * x[3],
            0.65 * (1.0 - x[4]),
            0.65 * x[5],
        ],
        dtype=float,
    )
    y = float(data.y[node])
    logit = math.log(y / (1.0 - y))
    counterfactual = 1.0 / (1.0 + np.exp(-(logit - contributions)))
    return y - counterfactual


def intervention_specifications(data, node: int, cfg: ExperimentConfig, metadata: Dict[str, object]):
    claim = cfg.num_policyholders + node
    policy = cfg.num_policyholders + cfg.num_claims + int(metadata["policy_for_holder"][node])
    provider = (
        cfg.num_policyholders
        + cfg.num_claims
        + cfg.num_health_policies
        + int(metadata["provider_for_claim"][node])
    )
    x = data.x
    return [
        ("age", node, 0, 0.0, 1.15 * float(x[node, 0])),
        ("poor_health", node, 1, 1.0, 1.55 * (1.0 - float(x[node, 1]))),
        ("smoking", node, 2, 0.0, 0.85 * float(x[node, 2])),
        ("bmi_risk", node, 3, 0.0, 0.75 * float(x[node, 3])),
        ("nonadherence", node, 4, 1.0, 0.65 * (1.0 - float(x[node, 4]))),
        ("claim_rate", node, 5, 0.0, 0.65 * float(x[node, 5])),
        ("claim_severity", claim, 0, 0.0, 1.55 * float(x[claim, 0])),
        ("provider_complication", provider, 0, 0.0, 0.70 * float(x[provider, 0])),
        ("coverage_gap", policy, 0, 0.0, 0.75 * float(x[policy, 0])),
    ]


def causal_intervention_rows(
    model: GraphRegressor,
    data,
    metadata: Dict[str, object],
    cfg: ExperimentConfig,
    data_seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        original_predictions = model(data.x, data.edge_index)
    for node in torch.where(data.test_mask)[0].tolist():
        target = float(data.y[node])
        target_logit = math.log(target / (1.0 - target))
        original_prediction = float(original_predictions[node])
        for variable, changed_node, feature, baseline, contribution in intervention_specifications(
            data, node, cfg, metadata
        ):
            counterfactual_x = data.x.clone()
            counterfactual_x[changed_node, feature] = baseline
            with torch.no_grad():
                counterfactual_prediction = float(model(counterfactual_x, data.edge_index)[node])
            counterfactual_target = 1.0 / (1.0 + math.exp(-(target_logit - contribution)))
            rows.append(
                {
                    "data_seed": data_seed,
                    "model_seed": 0,
                    "node_index": node,
                    "variable": variable,
                    "true_effect": target - counterfactual_target,
                    "model_effect": original_prediction - counterfactual_prediction,
                    "absolute_effect_error": abs(
                        (target - counterfactual_target)
                        - (original_prediction - counterfactual_prediction)
                    ),
                    "sign_agreement": int((original_prediction - counterfactual_prediction) >= 0.0),
                }
            )
    return rows


def evaluate_population(
    model: GraphRegressor,
    data,
    metadata: Dict[str, object],
    cfg: ExperimentConfig,
    data_seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    population_rows: List[Dict[str, object]] = []
    size_rows: List[Dict[str, object]] = []
    test_nodes = torch.where(data.test_mask)[0].tolist()
    for pos, node in enumerate(test_nodes, start=1):
        explanation_seed = 100_000 + 1000 * (data_seed - DATA_SEEDS[0]) + node
        base = explain_local_subgraph(
            model, data, node, cfg, explanation_seed=explanation_seed, epochs=cfg.explainer_epochs
        )
        attribution = integrated_gradients_subgraph(
            model,
            base["sub_x"],
            base["sub_edge_index"],
            int(base["local_target"]),
            cfg.integrated_gradient_steps,
        )
        direct_attr = attribution[int(base["local_target"]), :6].abs().detach().numpy()
        causal_feature_effects = direct_feature_causal_effects(data, node)
        rho = float(spearmanr(direct_attr, causal_feature_effects).statistic)
        top_attr = set(np.argsort(direct_attr)[-cfg.top_k_features :].tolist())
        top_causal = set(np.argsort(causal_feature_effects)[-cfg.top_k_features :].tolist())
        feature_top2_overlap = len(top_attr & top_causal) / cfg.top_k_features

        origin = float(base["origin"])
        selected_features = sorted(top_attr)
        x_selected = base["sub_x"].clone()
        x_selected[int(base["local_target"]), selected_features] = 0.0
        with torch.no_grad():
            selected_feature_delta = abs(
                origin
                - float(model(x_selected, base["sub_edge_index"])[int(base["local_target"])])
            )

        rng = np.random.default_rng(200_000 + 1000 * (data_seed - DATA_SEEDS[0]) + node)
        random_feature_deltas = []
        for _ in range(RANDOM_CONTROL_TRIALS):
            random_features = rng.choice(6, size=cfg.top_k_features, replace=False).tolist()
            x_random = base["sub_x"].clone()
            x_random[int(base["local_target"]), random_features] = 0.0
            with torch.no_grad():
                random_feature_deltas.append(
                    abs(
                        origin
                        - float(model(x_random, base["sub_edge_index"])[int(base["local_target"])])
                    )
                )

        row = {
            "data_seed": data_seed,
            "model_seed": 0,
            "node_index": node,
            "target": float(data.y[node]),
            "prediction": origin,
            "important_feature_delta": selected_feature_delta,
            "random_feature_delta_mean": float(np.mean(random_feature_deltas)),
            "feature_rank_spearman": rho,
            "feature_top2_causal_overlap": feature_top2_overlap,
        }

        all_pairs = list(base["pair_scores"].keys())
        causal_edges = causal_edge_set(node, cfg, metadata)
        for k in TOP_K_VALUES:
            selected = selected_pairs_for_k(base, k)
            selected_global = globalize_pairs(base, selected)
            pred_removed = prediction_for_pairs(model, base, selected, keep_selected=False)
            pred_retained = prediction_for_pairs(model, base, selected, keep_selected=True)
            fid_plus = abs(origin - pred_removed)
            fid_minus = abs(origin - pred_retained)
            sufficiency = max(0.0, 1.0 - fid_minus)
            characterization = 2.0 / (
                (1.0 / max(fid_plus, 1e-8)) + (1.0 / max(sufficiency, 1e-8))
            )
            random_deltas = []
            random_precision = []
            random_recall = []
            actual_k = min(k, len(all_pairs))
            for _ in range(RANDOM_CONTROL_TRIALS):
                indexes = rng.choice(len(all_pairs), size=actual_k, replace=False)
                random_pairs = [all_pairs[int(index)] for index in indexes]
                random_removed = prediction_for_pairs(model, base, random_pairs, keep_selected=False)
                random_deltas.append(abs(origin - random_removed))
                random_global = globalize_pairs(base, random_pairs)
                overlap = len(random_global & causal_edges)
                random_precision.append(overlap / max(1, len(random_global)))
                random_recall.append(overlap / len(causal_edges))
            overlap = len(selected_global & causal_edges)
            size_row = {
                "data_seed": data_seed,
                "model_seed": 0,
                "node_index": node,
                "top_k": k,
                "fidelity_plus": fid_plus,
                "fidelity_minus": fid_minus,
                "characterization": characterization,
                "sparsity": 1.0 - len(selected) / max(1, len(all_pairs)),
                "random_edge_delta_mean": float(np.mean(random_deltas)),
                "causal_edge_precision": overlap / max(1, len(selected_global)),
                "causal_edge_recall": overlap / len(causal_edges),
                "random_causal_edge_precision": float(np.mean(random_precision)),
                "random_causal_edge_recall": float(np.mean(random_recall)),
            }
            size_rows.append(size_row)
            if k == 4:
                row.update(
                    {
                        "important_edge_delta": fid_plus,
                        "random_edge_delta_mean": float(np.mean(random_deltas)),
                        "fidelity_plus": fid_plus,
                        "fidelity_minus": fid_minus,
                        "characterization": characterization,
                        "sparsity": size_row["sparsity"],
                        "causal_edge_precision": size_row["causal_edge_precision"],
                        "causal_edge_recall": size_row["causal_edge_recall"],
                        "random_causal_edge_precision": size_row[
                            "random_causal_edge_precision"
                        ],
                        "random_causal_edge_recall": size_row["random_causal_edge_recall"],
                    }
                )
        population_rows.append(row)
        if pos % 20 == 0:
            print(f"data seed {data_seed}: explained {pos}/{len(test_nodes)}", flush=True)
    return population_rows, size_rows


def stratified_stability_nodes(data) -> List[int]:
    nodes = torch.where(data.test_mask)[0].numpy()
    order = nodes[np.argsort(data.y[nodes].numpy())]
    positions = np.linspace(0, len(order) - 1, STABILITY_NODES_PER_GRAPH).round().astype(int)
    return order[positions].tolist()


def evaluate_stability(
    models: Dict[int, GraphRegressor],
    data,
    cfg: ExperimentConfig,
    data_seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    within_rows: List[Dict[str, object]] = []
    between_rows: List[Dict[str, object]] = []
    nodes = stratified_stability_nodes(data)
    original_sets: Dict[Tuple[int, int, int], set] = {}
    for model_seed, model in models.items():
        for pos, node in enumerate(nodes, start=1):
            exp_seed = 400_000 + 10_000 * (data_seed - DATA_SEEDS[0]) + node
            base = explain_local_subgraph(
                model, data, node, cfg, explanation_seed=exp_seed, epochs=cfg.explainer_epochs
            )
            for k in TOP_K_VALUES:
                original_sets[(model_seed, node, k)] = set(selected_pairs_for_k(base, k))
            for repetition in range(STABILITY_REPETITIONS):
                generator = torch.Generator().manual_seed(
                    500_000
                    + 100_000 * (data_seed - DATA_SEEDS[0])
                    + 1000 * model_seed
                    + 10 * node
                    + repetition
                )
                noise = (
                    torch.randn(base["sub_x"][:, :6].shape, generator=generator)
                    * cfg.stability_noise_sd
                )
                perturbed = explain_local_subgraph(
                    model,
                    data,
                    node,
                    cfg,
                    explanation_seed=exp_seed,
                    epochs=cfg.explainer_epochs,
                    x_noise=noise,
                )
                for k in TOP_K_VALUES:
                    perturbed_set = set(selected_pairs_for_k(perturbed, k))
                    within_rows.append(
                        {
                            "data_seed": data_seed,
                            "model_seed": model_seed,
                            "node_index": node,
                            "repetition": repetition,
                            "top_k": k,
                            "jaccard": jaccard(original_sets[(model_seed, node, k)], perturbed_set),
                        }
                    )
            if pos % 5 == 0:
                print(
                    f"data seed {data_seed}, model seed {model_seed}: stability {pos}/{len(nodes)}",
                    flush=True,
                )
    for node in nodes:
        for k in TOP_K_VALUES:
            for first, second in ((0, 1), (0, 2), (1, 2)):
                between_rows.append(
                    {
                        "data_seed": data_seed,
                        "node_index": node,
                        "top_k": k,
                        "model_seed_a": first,
                        "model_seed_b": second,
                        "jaccard": jaccard(
                            original_sets[(first, node, k)], original_sets[(second, node, k)]
                        ),
                    }
                )
    return within_rows, between_rows


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        value = min(1.0, (m - rank) * p_value)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def graph_level_matched_test(
    rows: Sequence[Dict[str, object]], selected_key: str, control_key: str
) -> Dict[str, object]:
    graph_diffs = []
    graph_selected = []
    graph_controls = []
    for seed in DATA_SEEDS:
        current = [row for row in rows if int(row["data_seed"]) == seed]
        selected = float(np.mean([float(row[selected_key]) for row in current]))
        control = float(np.mean([float(row[control_key]) for row in current]))
        graph_selected.append(selected)
        graph_controls.append(control)
        graph_diffs.append(selected - control)
    test = wilcoxon(graph_selected, graph_controls, alternative="two-sided", zero_method="wilcox")
    all_diffs = np.asarray(
        [float(row[selected_key]) - float(row[control_key]) for row in rows], dtype=float
    )
    return {
        "n_independent_graphs": len(DATA_SEEDS),
        "n_policyholders": len(rows),
        "selected_graph_mean": scalar_summary(graph_selected),
        "control_graph_mean": scalar_summary(graph_controls),
        "paired_graph_mean_difference": scalar_summary(graph_diffs),
        "policyholder_level_median_difference": float(np.median(all_diffs)),
        "two_sided_wilcoxon_statistic": float(test.statistic),
        "two_sided_p_value": float(test.pvalue),
    }


def summarize_by_k(rows: Sequence[Dict[str, object]], metric_names: Sequence[str]):
    output = {}
    for k in TOP_K_VALUES:
        current = [row for row in rows if int(row["top_k"]) == k]
        per_metric = {}
        for metric in metric_names:
            graph_means = []
            seeds = sorted({int(row["data_seed"]) for row in current})
            for seed in seeds:
                values = [float(row[metric]) for row in current if int(row["data_seed"]) == seed]
                graph_means.append(float(np.mean(values)))
            per_metric[metric] = scalar_summary(graph_means)
        output[str(k)] = per_metric
    return output


def causal_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    graph_rows = []
    for seed in DATA_SEEDS:
        current = [row for row in rows if int(row["data_seed"]) == seed]
        true_effect = np.asarray([float(row["true_effect"]) for row in current])
        model_effect = np.asarray([float(row["model_effect"]) for row in current])
        graph_rows.append(
            {
                "data_seed": seed,
                "spearman": float(spearmanr(true_effect, model_effect).statistic),
                "sign_agreement": float(np.mean(model_effect >= 0.0)),
                "effect_mae": float(np.mean(np.abs(true_effect - model_effect))),
            }
        )
    return {
        "per_graph": graph_rows,
        "across_graphs": {
            key: scalar_summary([float(row[key]) for row in graph_rows])
            for key in ("spearman", "sign_agreement", "effect_mae")
        },
    }


def main() -> None:
    outdir = Path("analysis/extended_experiment_output")
    outdir.mkdir(parents=True, exist_ok=True)
    base_cfg = ExperimentConfig(
        model_seeds=MODEL_SEEDS,
        explainer_epochs=60,
        random_control_trials=RANDOM_CONTROL_TRIALS,
        stability_runs=STABILITY_REPETITIONS,
    )
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    seed_all(2026)

    predictive_rows: List[Dict[str, object]] = []
    population_rows: List[Dict[str, object]] = []
    size_rows: List[Dict[str, object]] = []
    stability_rows: List[Dict[str, object]] = []
    between_model_rows: List[Dict[str, object]] = []
    intervention_rows: List[Dict[str, object]] = []
    metadata_by_seed = {}
    started = time.perf_counter()

    for data_seed in DATA_SEEDS:
        cfg = replace(base_cfg, data_seed=data_seed)
        print(f"\n=== data seed {data_seed} ===", flush=True)
        data, metadata = generate_health_graph(cfg)
        metadata_by_seed[str(data_seed)] = {
            key: metadata[key]
            for key in (
                "node_counts",
                "unique_relationships",
                "directed_edge_entries",
                "split_counts",
                "target_min",
                "target_max",
                "target_mean",
                "target_sd",
                "data_seed",
            )
        }

        for row in run_tabular_baselines(data, cfg):
            predictive_rows.append({"data_seed": data_seed, "model_seed": row["seed"], **row})

        graph_models: Dict[int, GraphRegressor] = {}
        for kind in ("GCN", "GAT", "GraphSAGE"):
            for model_seed in MODEL_SEEDS:
                model, metrics, _ = train_gnn(data, kind, model_seed, cfg)
                predictive_rows.append(
                    {
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "model": kind,
                        **metrics,
                    }
                )
                if kind == "GraphSAGE" and model_seed in EXPLANATION_MODEL_SEEDS:
                    graph_models[model_seed] = model

        for model_seed in MODEL_SEEDS:
            empty_edges = torch.empty((2, 0), dtype=torch.long)
            _, metrics, _ = train_gnn(
                data, "GraphSAGE", model_seed, cfg, edge_index_override=empty_edges
            )
            predictive_rows.append(
                {
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "model": "GraphSAGE features only",
                    **metrics,
                }
            )

            topology_x = data.x.clone()
            topology_x[:, :6] = 0.0
            _, metrics, _ = train_gnn(
                data, "GraphSAGE", model_seed, cfg, x_override=topology_x
            )
            predictive_rows.append(
                {
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "model": "GraphSAGE topology only",
                    **metrics,
                }
            )

            shuffled = shuffled_edge_index(data, seed=50_000 + 100 * data_seed + model_seed)
            _, metrics, _ = train_gnn(
                data, "GraphSAGE", model_seed, cfg, edge_index_override=shuffled
            )
            predictive_rows.append(
                {
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "model": "GraphSAGE shuffled edges",
                    **metrics,
                }
            )

        current_population, current_sizes = evaluate_population(
            graph_models[0], data, metadata, cfg, data_seed
        )
        population_rows.extend(current_population)
        size_rows.extend(current_sizes)
        intervention_rows.extend(
            causal_intervention_rows(graph_models[0], data, metadata, cfg, data_seed)
        )

        if data_seed in STABILITY_DATA_SEEDS:
            within, between = evaluate_stability(graph_models, data, cfg, data_seed)
            stability_rows.extend(within)
            between_model_rows.extend(between)

        print(
            f"completed data seed {data_seed}; elapsed {(time.perf_counter()-started)/60:.1f} min",
            flush=True,
        )

    predictive_summary = nested_predictive_summary(predictive_rows)
    comparisons = [
        paired_graph_comparison(predictive_rows, "GraphSAGE", other, "rmse")
        for other in ("GraphSAGE features only", "GCN", "GAT")
    ]

    relationship_test = graph_level_matched_test(
        population_rows, "important_edge_delta", "random_edge_delta_mean"
    )
    feature_test = graph_level_matched_test(
        population_rows, "important_feature_delta", "random_feature_delta_mean"
    )
    adjusted = holm_adjust(
        {
            "relationships": float(relationship_test["two_sided_p_value"]),
            "policyholder_features": float(feature_test["two_sided_p_value"]),
        }
    )
    relationship_test["holm_adjusted_p_value"] = adjusted["relationships"]
    feature_test["holm_adjusted_p_value"] = adjusted["policyholder_features"]

    explanation_summary = {
        "design": {
            "data_seeds": list(DATA_SEEDS),
            "population_explanation_model_seed": 0,
            "population_test_nodes_per_graph": 120,
            "random_control_trials_per_case": RANDOM_CONTROL_TRIALS,
            "stability_data_seeds": list(STABILITY_DATA_SEEDS),
            "stability_model_seeds": list(EXPLANATION_MODEL_SEEDS),
            "stability_nodes_per_graph": STABILITY_NODES_PER_GRAPH,
            "stability_repetitions": STABILITY_REPETITIONS,
            "explainer_epochs_original_and_perturbed": base_cfg.explainer_epochs,
            "top_k_values": list(TOP_K_VALUES),
            "feature_noise_sd": base_cfg.stability_noise_sd,
        },
        "matched_control_tests": {
            "relationships": relationship_test,
            "policyholder_features": feature_test,
        },
        "size_sensitivity": summarize_by_k(
            size_rows,
            (
                "fidelity_plus",
                "fidelity_minus",
                "characterization",
                "sparsity",
                "causal_edge_precision",
                "causal_edge_recall",
                "random_causal_edge_precision",
                "random_causal_edge_recall",
            ),
        ),
        "within_model_input_stability": summarize_by_k(stability_rows, ("jaccard",)),
        "between_model_seed_agreement": summarize_by_k(between_model_rows, ("jaccard",)),
        "feature_causal_ranking": {
            "spearman": scalar_summary(
                [float(row["feature_rank_spearman"]) for row in population_rows]
            ),
            "top2_overlap": scalar_summary(
                [float(row["feature_top2_causal_overlap"]) for row in population_rows]
            ),
        },
        "synthetic_causal_interventions": causal_summary(intervention_rows),
    }

    (outdir / "config.json").write_text(
        json.dumps(
            {
                **asdict(base_cfg),
                "data_seeds": DATA_SEEDS,
                "explanation_model_seeds": EXPLANATION_MODEL_SEEDS,
                "stability_data_seeds": STABILITY_DATA_SEEDS,
                "top_k_values": TOP_K_VALUES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "metadata_by_seed.json").write_text(
        json.dumps(metadata_by_seed, indent=2), encoding="utf-8"
    )
    (outdir / "predictive_summary.json").write_text(
        json.dumps(predictive_summary, indent=2), encoding="utf-8"
    )
    (outdir / "predictive_comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    (outdir / "explanation_summary.json").write_text(
        json.dumps(explanation_summary, indent=2), encoding="utf-8"
    )

    write_csv(
        outdir / "predictive_runs.csv",
        predictive_rows,
        (
            "data_seed",
            "model_seed",
            "model",
            "rmse",
            "mae",
            "r2",
            "training_seconds",
            "full_graph_inference_ms",
            "best_epoch",
            "best_val_rmse",
        ),
    )
    write_csv(
        outdir / "population_explanations.csv",
        population_rows,
        tuple(population_rows[0].keys()),
    )
    write_csv(outdir / "explanation_size_sensitivity.csv", size_rows, tuple(size_rows[0].keys()))
    write_csv(outdir / "stability_repetitions.csv", stability_rows, tuple(stability_rows[0].keys()))
    write_csv(
        outdir / "between_model_seed_agreement.csv",
        between_model_rows,
        tuple(between_model_rows[0].keys()),
    )
    write_csv(
        outdir / "synthetic_causal_interventions.csv",
        intervention_rows,
        tuple(intervention_rows[0].keys()),
    )
    print("\nPredictive summary")
    for row in predictive_summary:
        print(
            row["model"],
            f"RMSE={row['rmse_mean']:.4f}",
            f"95% CI [{row['rmse_ci95_low']:.4f}, {row['rmse_ci95_high']:.4f}]",
            f"R2={row['r2_mean']:.4f}",
        )
    print("\nMatched controls")
    print(json.dumps(explanation_summary["matched_control_tests"], indent=2))
    print("\nCausal validation")
    print(json.dumps(explanation_summary["synthetic_causal_interventions"], indent=2))
    print(f"total elapsed minutes={(time.perf_counter()-started)/60:.2f}")


if __name__ == "__main__":
    main()
