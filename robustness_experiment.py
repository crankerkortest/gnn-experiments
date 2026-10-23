"""Robustness analysis for synthetic causal-ground-truth validation.

The experiment changes one aspect of the synthetic data-generating process at
a time while retaining paired graph seeds.  It evaluates predictive accuracy,
intervention-effect recovery, feature ranking, and structural edge recovery.
The conditions are deliberately diagnostic rather than alternative claims of
real-world causal identification.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from extended_validation_experiment import (
    globalize_pairs,
    selected_pairs_for_k,
    stratified_stability_nodes,
)
from health_risk_experiment import (
    ExperimentConfig,
    NUMERICAL_ZERO_TOLERANCE,
    explain_local_subgraph,
    generate_health_graph,
    integrated_gradients_subgraph,
    regression_metrics,
    seed_all,
    train_gnn,
)


DATA_SEEDS = tuple(range(2026, 2031))
CONDITIONS = (
    "baseline",
    "causal_path_removed",
    "weaker_effects",
    "higher_noise",
    "nonlinear_effects",
    "omitted_latent_cause",
    "spurious_relationships",
)
CONDITION_LABELS = {
    "baseline": "Baseline generator",
    "causal_path_removed": "Provider path removed",
    "weaker_effects": "Effects multiplied by 0.5",
    "higher_noise": "Outcome noise multiplied by 2",
    "nonlinear_effects": "Nonlinear interactions added",
    "omitted_latent_cause": "Unobserved latent cause added",
    "spurious_relationships": "25% noncausal relationships added",
}
VARIABLES = (
    "age",
    "poor_health",
    "smoking",
    "bmi_risk",
    "nonadherence",
    "claim_rate",
    "claim_severity",
    "provider_complication",
    "coverage_gap",
)


@dataclass(frozen=True)
class ConditionState:
    name: str
    residual: np.ndarray
    latent: np.ndarray
    intercept_shift: float


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def context_arrays(data, metadata: Dict[str, object], cfg: ExperimentConfig):
    n = cfg.num_policyholders
    claim_nodes = np.arange(cfg.num_policyholders, cfg.num_policyholders + n)
    policy_nodes = (
        cfg.num_policyholders
        + cfg.num_claims
        + np.asarray(metadata["policy_for_holder"], dtype=int)
    )
    provider_nodes = (
        cfg.num_policyholders
        + cfg.num_claims
        + cfg.num_health_policies
        + np.asarray(metadata["provider_for_claim"], dtype=int)
    )
    x = data.x.detach().cpu().numpy()
    return x[:n, :6], x[claim_nodes, :6], x[policy_nodes, :6], x[provider_nodes, :6]


def deterministic_logit(
    data,
    metadata: Dict[str, object],
    cfg: ExperimentConfig,
    condition: str,
    latent: np.ndarray,
) -> np.ndarray:
    ph, claim, policy, provider = context_arrays(data, metadata, cfg)
    age = ph[:, 0]
    poor_health = 1.0 - ph[:, 1]
    smoking = ph[:, 2]
    bmi_risk = ph[:, 3]
    nonadherence = 1.0 - ph[:, 4]
    claim_rate = ph[:, 5]
    claim_severity = claim[:, 0]
    coverage_gap = policy[:, 0]
    provider_complication = provider[:, 0]

    coefficients = np.asarray([1.15, 1.55, 0.85, 0.75, 0.65, 0.65, 1.55, 0.70, 0.75])
    if condition == "weaker_effects":
        coefficients *= 0.5
    if condition == "causal_path_removed":
        coefficients[7] = 0.0
    terms = np.column_stack(
        [
            age,
            poor_health,
            smoking,
            bmi_risk,
            nonadherence,
            claim_rate,
            claim_severity,
            provider_complication,
            coverage_gap,
        ]
    )
    result = -4.10 + terms @ coefficients
    if condition == "nonlinear_effects":
        result = (
            result
            + 0.80 * poor_health * claim_severity
            + 0.60 * smoking * bmi_risk
            + 0.50 * coverage_gap * provider_complication
        )
    if condition == "omitted_latent_cause":
        result = result + 1.00 * (latent - 0.5)
    return result


def intercept_for_mean(logits: np.ndarray, target_mean: float) -> float:
    low, high = -8.0, 8.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if float(sigmoid(logits + middle).mean()) < target_mean:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def build_condition(
    base_data,
    metadata: Dict[str, object],
    cfg: ExperimentConfig,
    condition: str,
) -> Tuple[object, ConditionState]:
    data = copy.deepcopy(base_data)
    n = cfg.num_policyholders
    empty_latent = np.full(n, 0.5, dtype=float)
    base_deterministic = deterministic_logit(data, metadata, cfg, "baseline", empty_latent)
    base_target = data.y[:n].detach().cpu().numpy().astype(float)
    residual = logit(base_target) - base_deterministic
    latent = np.random.default_rng(700_000 + cfg.data_seed).beta(2.0, 2.0, n)

    if condition == "causal_path_removed":
        n_ph = cfg.num_policyholders
        n_claim = cfg.num_claims
        off_provider = n_ph + n_claim + cfg.num_health_policies
        pairs_to_remove = {
            tuple(sorted((n_ph + ph, off_provider + int(metadata["provider_for_claim"][ph]))))
            for ph in range(n_ph)
        }
        keep = [
            tuple(sorted((int(u), int(v)))) not in pairs_to_remove
            for u, v in data.edge_index.t().tolist()
        ]
        keep_tensor = torch.tensor(keep, dtype=torch.bool)
        data.edge_index = data.edge_index[:, keep_tensor]
        data.edge_type = data.edge_type[keep_tensor]

    if condition == "spurious_relationships":
        rng = np.random.default_rng(800_000 + cfg.data_seed)
        n_ph = cfg.num_policyholders
        off_policy = cfg.num_policyholders + cfg.num_claims
        off_provider = off_policy + cfg.num_health_policies
        existing = {
            tuple(sorted((int(u), int(v)))) for u, v in data.edge_index.t().tolist()
        }
        added: List[Tuple[int, int, int]] = []
        for ph in range(n_ph):
            if ph % 2 == 0:
                while True:
                    v = off_policy + int(rng.integers(0, cfg.num_health_policies))
                    pair = tuple(sorted((ph, v)))
                    if pair not in existing:
                        existing.add(pair)
                        added.append((ph, v, 1))
                        break
            else:
                claim = n_ph + ph
                while True:
                    v = off_provider + int(rng.integers(0, cfg.num_providers))
                    pair = tuple(sorted((claim, v)))
                    if pair not in existing:
                        existing.add(pair)
                        added.append((claim, v, 2))
                        break
        directed = []
        types = []
        for u, v, edge_type in added:
            directed.extend([(u, v), (v, u)])
            types.extend([edge_type, edge_type])
        extra_edges = torch.tensor(directed, dtype=torch.long).t().contiguous()
        data.edge_index = torch.cat([data.edge_index, extra_edges], dim=1)
        data.edge_type = torch.cat([data.edge_type, torch.tensor(types, dtype=torch.long)])

    condition_residual = residual * (2.0 if condition == "higher_noise" else 1.0)
    deterministic = deterministic_logit(data, metadata, cfg, condition, latent)
    raw_logits = deterministic + condition_residual
    target_mean = float(base_target.mean())
    shift = 0.0 if condition == "baseline" else intercept_for_mean(raw_logits, target_mean)
    new_target = sigmoid(raw_logits + shift).astype(np.float32)
    data.y[:n] = torch.tensor(new_target)
    return data, ConditionState(condition, condition_residual, latent, shift)


def generator_target(
    data,
    metadata: Dict[str, object],
    cfg: ExperimentConfig,
    state: ConditionState,
) -> np.ndarray:
    logits = deterministic_logit(data, metadata, cfg, state.name, state.latent)
    return sigmoid(logits + state.residual + state.intercept_shift)


def causal_edges(
    node: int,
    cfg: ExperimentConfig,
    metadata: Dict[str, object],
    condition: str,
) -> set[Tuple[int, int]]:
    claim = cfg.num_policyholders + node
    policy = cfg.num_policyholders + cfg.num_claims + int(metadata["policy_for_holder"][node])
    provider = (
        cfg.num_policyholders
        + cfg.num_claims
        + cfg.num_health_policies
        + int(metadata["provider_for_claim"][node])
    )
    edges = {
        tuple(sorted((node, claim))),
        tuple(sorted((node, policy))),
        tuple(sorted((claim, policy))),
    }
    if condition != "causal_path_removed":
        edges.add(tuple(sorted((claim, provider))))
    return edges


def intervention_specs(node: int, cfg: ExperimentConfig, metadata: Dict[str, object]):
    claim = cfg.num_policyholders + node
    policy = cfg.num_policyholders + cfg.num_claims + int(metadata["policy_for_holder"][node])
    provider = (
        cfg.num_policyholders
        + cfg.num_claims
        + cfg.num_health_policies
        + int(metadata["provider_for_claim"][node])
    )
    return (
        ("age", node, 0, 0.0),
        ("poor_health", node, 1, 1.0),
        ("smoking", node, 2, 0.0),
        ("bmi_risk", node, 3, 0.0),
        ("nonadherence", node, 4, 1.0),
        ("claim_rate", node, 5, 0.0),
        ("claim_severity", claim, 0, 0.0),
        ("provider_complication", provider, 0, 0.0),
        ("coverage_gap", policy, 0, 0.0),
    )


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    value = float(spearmanr(first, second).statistic)
    return value if math.isfinite(value) else 0.0


def normalize_numerical_zero(value: float) -> float:
    return 0.0 if abs(value) < NUMERICAL_ZERO_TOLERANCE else float(value)


def stratified_nodes(data, count: int) -> List[int]:
    original = stratified_stability_nodes(data)
    if count == len(original):
        return original
    nodes = torch.where(data.test_mask)[0].detach().cpu().numpy()
    ordered = nodes[np.argsort(data.y[nodes].detach().cpu().numpy())]
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered[positions].tolist()


def evaluate_condition(
    data,
    metadata: Dict[str, object],
    cfg: ExperimentConfig,
    state: ConditionState,
    nodes_per_graph: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model, predictive, _ = train_gnn(data, "GraphSAGE", 0, cfg)
    nodes = stratified_nodes(data, nodes_per_graph)
    node_rows: List[Dict[str, object]] = []
    intervention_true: List[float] = []
    intervention_model: List[float] = []
    true_baseline = generator_target(data, metadata, cfg, state)
    with torch.no_grad():
        model_baseline = model(data.x, data.edge_index).detach().cpu().numpy()

    for position, node in enumerate(nodes, start=1):
        explanation = explain_local_subgraph(
            model,
            data,
            node,
            cfg,
            explanation_seed=900_000 + 10_000 * (cfg.data_seed - DATA_SEEDS[0]) + node,
            epochs=cfg.explainer_epochs,
        )
        selected = selected_pairs_for_k(explanation, 4)
        selected_global = globalize_pairs(explanation, selected)
        truth_edges = causal_edges(node, cfg, metadata, state.name)
        overlap = len(selected_global & truth_edges)

        attribution = integrated_gradients_subgraph(
            model,
            explanation["sub_x"],
            explanation["sub_edge_index"],
            int(explanation["local_target"]),
            cfg.integrated_gradient_steps,
        )
        direct_attr = attribution[int(explanation["local_target"]), :6].abs().detach().cpu().numpy()
        direct_true: List[float] = []
        local_true: List[float] = []
        local_model: List[float] = []
        for variable, changed_node, feature, baseline in intervention_specs(node, cfg, metadata):
            counterfactual = copy.deepcopy(data)
            counterfactual.x[changed_node, feature] = baseline
            cf_target = generator_target(counterfactual, metadata, cfg, state)
            input_is_unchanged = (
                abs(float(data.x[changed_node, feature]) - baseline)
                < NUMERICAL_ZERO_TOLERANCE
            )
            true_effect = 0.0 if input_is_unchanged else normalize_numerical_zero(
                float(true_baseline[node] - cf_target[node])
            )
            with torch.no_grad():
                cf_prediction = float(model(counterfactual.x, counterfactual.edge_index)[node])
            model_effect = normalize_numerical_zero(
                float(model_baseline[node] - cf_prediction)
            )
            intervention_true.append(true_effect)
            intervention_model.append(model_effect)
            local_true.append(true_effect)
            local_model.append(model_effect)
            if variable in VARIABLES[:6]:
                direct_true.append(abs(true_effect))

        top_attr = set(np.argsort(direct_attr)[-cfg.top_k_features :].tolist())
        top_true = set(np.argsort(np.asarray(direct_true))[-cfg.top_k_features :].tolist())
        node_rows.append(
            {
                "condition": state.name,
                "data_seed": cfg.data_seed,
                "node_index": node,
                "feature_rank_spearman": safe_spearman(direct_attr, direct_true),
                "feature_top2_overlap": len(top_attr & top_true) / cfg.top_k_features,
                "causal_edge_precision": overlap / max(1, len(selected_global)),
                "causal_edge_recall": overlap / max(1, len(truth_edges)),
                "causal_edge_jaccard": overlap / max(1, len(selected_global | truth_edges)),
                "selected_edge_count": len(selected_global),
                "available_local_edge_count": len(explanation["pair_scores"]),
                "local_effect_spearman": safe_spearman(local_true, local_model),
                "local_effect_mae": float(np.mean(np.abs(np.asarray(local_true) - np.asarray(local_model)))),
            }
        )
        if position % 5 == 0 or position == len(nodes):
            print(
                f"{state.name} seed {cfg.data_seed}: explained {position}/{len(nodes)}",
                flush=True,
            )

    true_array = np.asarray(intervention_true)
    model_array = np.asarray(intervention_model)
    true_array[np.abs(true_array) < NUMERICAL_ZERO_TOLERANCE] = 0.0
    model_array[np.abs(model_array) < NUMERICAL_ZERO_TOLERANCE] = 0.0
    effect_mae = float(np.mean(np.abs(true_array - model_array)))
    denominator = float(np.mean(np.abs(true_array)))
    sign_mask = np.abs(true_array) >= NUMERICAL_ZERO_TOLERANCE
    sign_agreement = float(
        np.mean(np.sign(true_array[sign_mask]) == np.sign(model_array[sign_mask]))
    ) if np.any(sign_mask) else 1.0
    summary: Dict[str, object] = {
        "condition": state.name,
        "condition_label": CONDITION_LABELS[state.name],
        "data_seed": cfg.data_seed,
        "nodes_evaluated": len(nodes),
        "rmse": predictive["rmse"],
        "mae": predictive["mae"],
        "r2": predictive["r2"],
        "effect_spearman": safe_spearman(true_array, model_array),
        "effect_sign_agreement": sign_agreement,
        "effect_sign_agreement_denominator": int(sign_mask.sum()),
        "effect_noop_count": int((~sign_mask).sum()),
        "effect_mae": effect_mae,
        "effect_normalized_mae": effect_mae / max(denominator, 1e-12),
        "feature_rank_spearman": float(np.mean([row["feature_rank_spearman"] for row in node_rows])),
        "feature_top2_overlap": float(np.mean([row["feature_top2_overlap"] for row in node_rows])),
        "causal_edge_precision": float(np.mean([row["causal_edge_precision"] for row in node_rows])),
        "causal_edge_recall": float(np.mean([row["causal_edge_recall"] for row in node_rows])),
        "causal_edge_jaccard": float(np.mean([row["causal_edge_jaccard"] for row in node_rows])),
        "intercept_shift": state.intercept_shift,
        "directed_edge_entries": int(data.edge_index.size(1)),
    }
    return summary, node_rows


def aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = (
        "rmse",
        "mae",
        "r2",
        "effect_spearman",
        "effect_sign_agreement",
        "effect_mae",
        "effect_normalized_mae",
        "feature_rank_spearman",
        "feature_top2_overlap",
        "causal_edge_precision",
        "causal_edge_recall",
        "causal_edge_jaccard",
    )
    output: List[Dict[str, object]] = []
    for condition in CONDITIONS:
        current = [row for row in rows if row["condition"] == condition]
        item: Dict[str, object] = {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "n_graphs": len(current),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in current])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run one seed, two conditions, and four nodes")
    parser.add_argument("--nodes-per-graph", type=int, default=30)
    parser.add_argument("--outdir", type=Path, default=Path("analysis/robustness_experiment_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    seed_all(2026)
    seeds = DATA_SEEDS[:1] if args.smoke else DATA_SEEDS
    conditions = CONDITIONS[:2] if args.smoke else CONDITIONS
    nodes_per_graph = 4 if args.smoke else args.nodes_per_graph
    cfg_base = ExperimentConfig(
        model_seeds=(0,),
        explainer_epochs=60,
        integrated_gradient_steps=50,
        max_epochs=400,
        patience=40,
    )
    graph_rows: List[Dict[str, object]] = []
    node_rows: List[Dict[str, object]] = []
    for data_seed in seeds:
        cfg = replace(cfg_base, data_seed=data_seed)
        base_data, metadata = generate_health_graph(cfg)
        for condition in conditions:
            print(f"\n=== {condition}; data seed {data_seed} ===", flush=True)
            data, state = build_condition(base_data, metadata, cfg, condition)
            summary, nodes = evaluate_condition(data, metadata, cfg, state, nodes_per_graph)
            graph_rows.append(summary)
            node_rows.extend(nodes)

    aggregate_rows = aggregate(graph_rows) if not args.smoke else []
    suffix = "_smoke" if args.smoke else ""
    write_csv(args.outdir / f"robustness_graph_metrics{suffix}.csv", graph_rows)
    write_csv(args.outdir / f"robustness_node_metrics{suffix}.csv", node_rows)
    if aggregate_rows:
        write_csv(args.outdir / "robustness_summary.csv", aggregate_rows)
        (args.outdir / "robustness_summary.json").write_text(
            json.dumps(aggregate_rows, indent=2), encoding="utf-8"
        )
    (args.outdir / f"robustness_config{suffix}.json").write_text(
        json.dumps(
            {
                "base_config": asdict(cfg_base),
                "data_seeds": list(seeds),
                "conditions": list(conditions),
                "condition_labels": CONDITION_LABELS,
                "nodes_per_graph": nodes_per_graph,
                "structural_explanation_size": 4,
                "structural_explanation_size_rule": "min(k, available_local_edges)",
                "numerical_zero_tolerance": NUMERICAL_ZERO_TOLERANCE,
                "mean_target_matched_across_conditions": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(aggregate_rows or graph_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
