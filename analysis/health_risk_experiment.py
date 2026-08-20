"""Reproducible policyholder health-risk regression experiment.

The experiment replaces the earlier claim-payout task with node-level prediction
of a health-insurance policyholder risk score bounded to [0, 1]. Claims,
health-insurance objects, and healthcare providers are relational context.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from torch import nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import k_hop_subgraph


POLICYHOLDER_FEATURES = [
    "Age (normalized)",
    "Health score",
    "Smoking status",
    "BMI risk index",
    "Preventive check-up adherence",
    "Claims in previous 12 months (normalized)",
]

NODE_TYPES = ["policyholder", "claim", "health_policy", "healthcare_provider"]

RELATION_NAMES = {
    0: "filed",
    1: "holds",
    2: "serviced_by",
    3: "covered_by",
}


@dataclass
class ExperimentConfig:
    data_seed: int = 2026
    model_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    num_policyholders: int = 600
    num_claims: int = 600
    num_health_policies: int = 450
    num_providers: int = 40
    hidden_channels: int = 64
    dropout: float = 0.20
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    max_epochs: int = 400
    patience: int = 40
    smooth_l1_beta: float = 0.10
    explainer_epochs: int = 60
    explainer_lr: float = 0.01
    integrated_gradient_steps: int = 50
    top_k_edges: int = 4
    top_k_features: int = 2
    random_control_trials: int = 20
    stability_runs: int = 2
    stability_noise_sd: float = 0.01


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_health_graph(cfg: ExperimentConfig) -> Tuple[Data, Dict[str, object]]:
    """Generate a deterministic heterogeneous property graph, then homogenize it.

    Six policyholder features and six type-specific contextual slots are followed
    by four one-hot node-type indicators, yielding ten model input dimensions.
    """
    rng = np.random.default_rng(cfg.data_seed)

    n_ph = cfg.num_policyholders
    n_claim = cfg.num_claims
    n_policy = cfg.num_health_policies
    n_provider = cfg.num_providers
    if n_ph != n_claim:
        raise ValueError("This controlled generator expects one current claim per policyholder")

    off_ph = 0
    off_claim = n_ph
    off_policy = off_claim + n_claim
    off_provider = off_policy + n_policy
    n_total = off_provider + n_provider

    # Policyholder attributes, all scaled to [0, 1].
    age_years = np.clip(rng.normal(46.0, 15.0, n_ph), 18.0, 80.0)
    age_norm = (age_years - 18.0) / 62.0
    health_score = rng.beta(4.5, 2.2, n_ph)
    smoking = rng.binomial(1, 0.22, n_ph).astype(float)
    bmi = np.clip(rng.normal(27.0 + 2.0 * smoking, 4.5, n_ph), 17.0, 45.0)
    bmi_risk = np.clip(np.abs(bmi - 22.0) / 18.0, 0.0, 1.0)
    checkup_adherence = rng.beta(4.0, 2.2, n_ph)
    claim_rate = np.clip(
        rng.poisson(0.35 + 1.35 * (1.0 - health_score) + 0.65 * smoking, n_ph),
        0,
        5,
    ) / 5.0

    ph_features = np.column_stack(
        [age_norm, health_score, smoking, bmi_risk, checkup_adherence, claim_rate]
    ).astype(np.float32)

    # Provider context.
    provider_complication = rng.beta(2.0, 7.0, n_provider)
    provider_delay = rng.beta(2.2, 5.5, n_provider)
    provider_quality = rng.beta(6.0, 2.0, n_provider)
    provider_cost = rng.beta(2.4, 4.0, n_provider)
    provider_volume = rng.beta(3.0, 3.0, n_provider)
    provider_specialization = rng.uniform(0.0, 1.0, n_provider)
    provider_features = np.column_stack(
        [
            provider_complication,
            provider_delay,
            provider_quality,
            provider_cost,
            provider_volume,
            provider_specialization,
        ]
    ).astype(np.float32)

    # Health-insurance object (policy/coverage) context.
    coverage_gap = rng.beta(2.2, 5.0, n_policy)
    deductible_ratio = rng.beta(2.0, 5.0, n_policy)
    waiting_period = rng.beta(1.8, 6.0, n_policy)
    copay_ratio = rng.beta(2.2, 4.5, n_policy)
    coverage_scope = rng.beta(5.5, 2.0, n_policy)
    policy_duration = rng.beta(3.5, 2.5, n_policy)
    policy_features = np.column_stack(
        [coverage_gap, deductible_ratio, waiting_period, copay_ratio, coverage_scope, policy_duration]
    ).astype(np.float32)

    # Assign one current claim, one provider, and one health-insurance object to each policyholder.
    claim_owner = np.arange(n_ph, dtype=np.int64)
    provider_for_claim = rng.integers(0, n_provider, n_claim, dtype=np.int64)
    policy_for_holder = rng.integers(0, n_policy, n_ph, dtype=np.int64)
    policy_for_claim = policy_for_holder.copy()

    poor_health = 1.0 - health_score
    claim_severity = np.clip(
        0.48 * poor_health
        + 0.18 * smoking
        + 0.16 * claim_rate
        + rng.normal(0.0, 0.08, n_claim),
        0.0,
        1.0,
    )
    chronic_related = rng.binomial(1, np.clip(0.10 + 0.58 * poor_health, 0.0, 0.9)).astype(float)
    emergency = rng.binomial(1, np.clip(0.08 + 0.55 * claim_severity, 0.0, 0.9)).astype(float)
    normalized_cost = np.clip(0.72 * claim_severity + 0.18 * emergency + rng.normal(0, 0.07, n_claim), 0, 1)
    recency = rng.uniform(0.0, 1.0, n_claim)
    unresolved = rng.binomial(1, np.clip(0.12 + 0.50 * claim_severity, 0.0, 0.85)).astype(float)
    claim_features = np.column_stack(
        [claim_severity, chronic_related, emergency, normalized_cost, recency, unresolved]
    ).astype(np.float32)

    provider_context = provider_complication[provider_for_claim]
    policy_context = coverage_gap[policy_for_holder]
    risk_logit = (
        -4.10
        + 1.15 * age_norm
        + 1.55 * poor_health
        + 0.85 * smoking
        + 0.75 * bmi_risk
        + 0.65 * (1.0 - checkup_adherence)
        + 0.65 * claim_rate
        + 1.55 * claim_severity
        + 0.70 * provider_context
        + 0.75 * policy_context
        + rng.normal(0.0, 0.12, n_ph)
    )
    y_policyholder = sigmoid_np(risk_logit).astype(np.float32)

    # Ten input dimensions: six type-specific numeric slots + four type indicators.
    x = np.zeros((n_total, 10), dtype=np.float32)
    node_type = np.zeros(n_total, dtype=np.int64)
    x[off_ph:off_claim, :6] = ph_features
    x[off_claim:off_policy, :6] = claim_features
    x[off_policy:off_provider, :6] = policy_features
    x[off_provider:, :6] = provider_features
    node_type[off_ph:off_claim] = 0
    node_type[off_claim:off_policy] = 1
    node_type[off_policy:off_provider] = 2
    node_type[off_provider:] = 3
    x[np.arange(n_total), 6 + node_type] = 1.0

    y = np.zeros(n_total, dtype=np.float32)
    y[:n_ph] = y_policyholder

    # Unique semantic relationships and their types.
    unique_edges: List[Tuple[int, int, int]] = []
    for ph in range(n_ph):
        claim = off_claim + ph
        policy = off_policy + int(policy_for_holder[ph])
        provider = off_provider + int(provider_for_claim[ph])
        unique_edges.extend(
            [
                (ph, claim, 0),
                (ph, policy, 1),
                (claim, provider, 2),
                (claim, policy, 3),
            ]
        )

    directed_edges: List[Tuple[int, int]] = []
    directed_relation_type: List[int] = []
    for u, v, rel in unique_edges:
        directed_edges.extend([(u, v), (v, u)])
        directed_relation_type.extend([rel, rel])
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()

    perm = rng.permutation(n_ph)
    train_ids = np.sort(perm[:384])
    val_ids = np.sort(perm[384:480])
    test_ids = np.sort(perm[480:])
    train_mask = np.zeros(n_total, dtype=bool)
    val_mask = np.zeros(n_total, dtype=bool)
    test_mask = np.zeros(n_total, dtype=bool)
    train_mask[train_ids] = True
    val_mask[val_ids] = True
    test_mask[test_ids] = True

    data = Data(
        x=torch.tensor(x),
        edge_index=edge_index,
        y=torch.tensor(y),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
        node_type=torch.tensor(node_type),
        edge_type=torch.tensor(directed_relation_type, dtype=torch.long),
    )

    metadata: Dict[str, object] = {
        "node_counts": {
            "policyholder": n_ph,
            "claim": n_claim,
            "health_policy": n_policy,
            "healthcare_provider": n_provider,
            "total": n_total,
        },
        "unique_relationships": len(unique_edges),
        "directed_edge_entries": edge_index.size(1),
        "split_counts": {"train": 384, "validation": 96, "test": 120},
        "policyholder_features": POLICYHOLDER_FEATURES,
        "input_dimensions": 10,
        "target_min": float(y_policyholder.min()),
        "target_max": float(y_policyholder.max()),
        "target_mean": float(y_policyholder.mean()),
        "target_sd": float(y_policyholder.std(ddof=1)),
        "data_seed": cfg.data_seed,
        "node_names": (
            [f"PH-{i:03d}" for i in range(n_ph)]
            + [f"CL-{i:03d}" for i in range(n_claim)]
            + [f"HP-{i:03d}" for i in range(n_policy)]
            + [f"PR-{i:03d}" for i in range(n_provider)]
        ),
        "unique_edges": unique_edges,
        "provider_for_claim": provider_for_claim.tolist(),
        "policy_for_holder": policy_for_holder.tolist(),
    }
    return data, metadata


class GraphRegressor(nn.Module):
    def __init__(self, kind: str, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.kind = kind
        self.dropout = float(dropout)
        if kind == "GraphSAGE":
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
            self.conv2 = SAGEConv(hidden_channels, 1, aggr="mean")
        elif kind == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, 1)
        elif kind == "GAT":
            heads = 4
            head_width = hidden_channels // heads
            self.conv1 = GATConv(in_channels, head_width, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden_channels, 1, heads=1, concat=False, dropout=dropout)
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).squeeze(-1)
        return torch.sigmoid(x)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_gnn(
    data: Data,
    kind: str,
    seed: int,
    cfg: ExperimentConfig,
    x_override: torch.Tensor | None = None,
    edge_index_override: torch.Tensor | None = None,
) -> Tuple[GraphRegressor, Dict[str, float], Dict[str, object]]:
    seed_all(seed)
    x = data.x if x_override is None else x_override
    edge_index = data.edge_index if edge_index_override is None else edge_index_override
    model = GraphRegressor(kind, x.size(1), cfg.hidden_channels, cfg.dropout)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.SmoothL1Loss(beta=cfg.smooth_l1_beta)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improvement = 0
    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x, edge_index)
        loss = criterion(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_eval = model(x, edge_index)
            val_rmse = float(
                torch.sqrt(torch.mean((pred_eval[data.val_mask] - data.y[data.val_mask]) ** 2))
            )
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, "train_loss": float(loss), "val_rmse": val_rmse})
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= cfg.patience:
            break

    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("No model state captured")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        final_pred = model(x, edge_index)
        inference_ms = (time.perf_counter() - t0) * 1000.0
    y_true = data.y[data.test_mask].numpy()
    y_pred = final_pred[data.test_mask].numpy()
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(
        {
            "training_seconds": float(training_seconds),
            "full_graph_inference_ms": float(inference_ms),
            "best_epoch": int(best_epoch),
            "best_val_rmse": float(best_val),
        }
    )
    details: Dict[str, object] = {
        "history": history,
        "predictions": final_pred.detach().numpy().tolist(),
        "seed": seed,
        "kind": kind,
    }
    return model, metrics, details


def run_tabular_baselines(data: Data, cfg: ExperimentConfig) -> List[Dict[str, object]]:
    x = data.x[: cfg.num_policyholders, :6].numpy()
    y = data.y[: cfg.num_policyholders].numpy()
    train = np.flatnonzero(data.train_mask[: cfg.num_policyholders].numpy())
    val = np.flatnonzero(data.val_mask[: cfg.num_policyholders].numpy())
    test = np.flatnonzero(data.test_mask[: cfg.num_policyholders].numpy())
    fit_idx = np.concatenate([train, val])
    results: List[Dict[str, object]] = []

    for seed in cfg.model_seeds:
        estimators = {
            "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
            "Random forest": RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=-1
            ),
            "Gradient boosting": HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=0.01, random_state=seed
            ),
            "MLP": make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                    learning_rate_init=0.002, max_iter=1000, early_stopping=True,
                    validation_fraction=0.20, n_iter_no_change=40, random_state=seed,
                ),
            ),
        }
        for name, estimator in estimators.items():
            start = time.perf_counter()
            estimator.fit(x[fit_idx], y[fit_idx])
            training_seconds = time.perf_counter() - start
            t0 = time.perf_counter()
            pred = np.clip(estimator.predict(x[test]), 0.0, 1.0)
            inference_ms = (time.perf_counter() - t0) * 1000.0
            metrics = regression_metrics(y[test], pred)
            results.append(
                {
                    "model": name,
                    "seed": seed,
                    **metrics,
                    "training_seconds": training_seconds,
                    "full_graph_inference_ms": inference_ms,
                }
            )
    return results


def shuffled_edge_index(data: Data, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    edges = data.edge_index.numpy().T
    # Work from one orientation of each undirected pair, then rebuild both directions.
    pairs = sorted({tuple(sorted((int(u), int(v)))) for u, v in edges if u != v})
    u = np.array([a for a, _ in pairs], dtype=np.int64)
    v = np.array([b for _, b in pairs], dtype=np.int64)
    shuffled_v = rng.permutation(v)
    rebuilt = []
    for a, b in zip(u, shuffled_v):
        if a == b:
            b = int((b + 1) % data.num_nodes)
        rebuilt.extend([(int(a), int(b)), (int(b), int(a))])
    return torch.tensor(rebuilt, dtype=torch.long).t().contiguous()


def summarize_runs(rows: Sequence[Dict[str, object]], group_key: str = "model") -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    out: List[Dict[str, object]] = []
    for name, items in grouped.items():
        summary: Dict[str, object] = {group_key: name, "n_runs": len(items)}
        for metric in ["rmse", "mae", "r2", "training_seconds", "full_graph_inference_ms"]:
            vals = np.array([float(x[metric]) for x in items], dtype=float)
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_ci95"] = float(ci)
        out.append(summary)
    return sorted(out, key=lambda r: float(r["rmse_mean"]))


def undirected_pair_scores(edge_index: torch.Tensor, edge_mask: torch.Tensor) -> Dict[Tuple[int, int], float]:
    values: Dict[Tuple[int, int], List[float]] = {}
    for i, (u, v) in enumerate(edge_index.t().tolist()):
        if u == v:
            continue
        key = tuple(sorted((int(u), int(v))))
        values.setdefault(key, []).append(float(edge_mask[i]))
    return {key: float(np.mean(scores)) for key, scores in values.items()}


def edge_mask_from_pairs(edge_index: torch.Tensor, selected: Iterable[Tuple[int, int]], keep_selected: bool) -> torch.Tensor:
    selected_set = set(selected)
    keep = []
    for u, v in edge_index.t().tolist():
        is_selected = tuple(sorted((int(u), int(v)))) in selected_set
        keep.append(is_selected if keep_selected else not is_selected)
    return torch.tensor(keep, dtype=torch.bool)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def integrated_gradients_subgraph(
    model: GraphRegressor,
    sub_x: torch.Tensor,
    sub_edge_index: torch.Tensor,
    local_target: int,
    steps: int,
) -> torch.Tensor:
    """Integrated Gradients for one fixed local graph.

    The full node-by-feature matrix is one structured input, not a batch of
    independent node samples. Computing the path integral explicitly avoids
    treating the node dimension as Captum's batch dimension.
    """
    baseline = torch.zeros_like(sub_x)
    delta = sub_x - baseline
    accumulated = torch.zeros_like(sub_x)
    alphas = torch.linspace(0.0, 1.0, steps + 1)
    model.eval()
    for position, alpha in enumerate(alphas):
        interpolated = (baseline + alpha * delta).detach().requires_grad_(True)
        output = model(interpolated, sub_edge_index)[local_target]
        gradient = torch.autograd.grad(output, interpolated, retain_graph=False)[0]
        weight = 0.5 if position in (0, steps) else 1.0
        accumulated += weight * gradient.detach()
    average_gradient = accumulated / steps
    return delta * average_gradient


def explain_local_subgraph(
    model: GraphRegressor,
    data: Data,
    global_node: int,
    cfg: ExperimentConfig,
    explanation_seed: int,
    epochs: int | None = None,
    x_noise: torch.Tensor | None = None,
) -> Dict[str, object]:
    subset, sub_edge_index, mapping, edge_mask_global = k_hop_subgraph(
        global_node,
        2,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    sub_x = data.x[subset].clone()
    if x_noise is not None:
        sub_x[:, :6] = torch.clamp(sub_x[:, :6] + x_noise, 0.0, 1.0)
    local_target = int(mapping.item())
    seed_all(explanation_seed)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs or cfg.explainer_epochs, lr=cfg.explainer_lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="node", return_type="raw"),
    )
    start = time.perf_counter()
    explanation = explainer(sub_x, sub_edge_index, index=local_target)
    explainer_seconds = time.perf_counter() - start
    pair_scores = undirected_pair_scores(sub_edge_index, explanation.edge_mask.detach())
    ranked_pairs = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    k = min(cfg.top_k_edges, len(ranked_pairs))
    selected_pairs = [pair for pair, _ in ranked_pairs[:k]]

    model.eval()
    with torch.no_grad():
        origin = float(model(sub_x, sub_edge_index)[local_target])
        remove_keep = edge_mask_from_pairs(sub_edge_index, selected_pairs, keep_selected=False)
        retain_keep = edge_mask_from_pairs(sub_edge_index, selected_pairs, keep_selected=True)
        pred_removed = float(model(sub_x, sub_edge_index[:, remove_keep])[local_target])
        pred_retained = float(model(sub_x, sub_edge_index[:, retain_keep])[local_target])
    fid_plus = abs(origin - pred_removed)
    fid_minus = abs(origin - pred_retained)
    sufficiency = max(0.0, 1.0 - fid_minus)
    characterization = 2.0 / ((1.0 / max(fid_plus, 1e-8)) + (1.0 / max(sufficiency, 1e-8)))
    sparsity = 1.0 - (k / max(1, len(pair_scores)))

    return {
        "subset": subset,
        "sub_x": sub_x,
        "sub_edge_index": sub_edge_index,
        "local_target": local_target,
        "global_node": global_node,
        "edge_mask": explanation.edge_mask.detach(),
        "pair_scores": pair_scores,
        "selected_pairs": selected_pairs,
        "origin": origin,
        "pred_removed": pred_removed,
        "pred_retained": pred_retained,
        "fidelity_plus": fid_plus,
        "fidelity_minus": fid_minus,
        "characterization": characterization,
        "sparsity": sparsity,
        "explainer_seconds": explainer_seconds,
    }


def evaluate_explanations(
    model: GraphRegressor,
    data: Data,
    metadata: Dict[str, object],
    cfg: ExperimentConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    test_nodes = torch.where(data.test_mask)[0].tolist()
    node_names = metadata["node_names"]
    representative_payload = None

    for position, node in enumerate(test_nodes, start=1):
        base_exp = explain_local_subgraph(model, data, node, cfg, explanation_seed=10_000 + node)
        subset = base_exp["subset"]
        sub_x = base_exp["sub_x"]
        sub_edge_index = base_exp["sub_edge_index"]
        local_target = int(base_exp["local_target"])
        selected_pairs = list(base_exp["selected_pairs"])

        # Integrated Gradients with a zero baseline and 50 trapezoidal steps.
        ig_start = time.perf_counter()
        attribution = integrated_gradients_subgraph(
            model,
            sub_x,
            sub_edge_index,
            local_target,
            steps=cfg.integrated_gradient_steps,
        )
        ig_seconds = time.perf_counter() - ig_start
        direct_feature_attr = attribution[local_target, :6].abs()
        kf = min(cfg.top_k_features, direct_feature_attr.numel())
        important_features = torch.topk(direct_feature_attr, k=kf).indices.tolist()

        with torch.no_grad():
            origin = float(model(sub_x, sub_edge_index)[local_target])
            baseline_prediction = float(
                model(torch.zeros_like(sub_x), sub_edge_index)[local_target]
            )
            x_important = sub_x.clone()
            x_important[local_target, important_features] = 0.0
            important_feature_delta = abs(
                origin - float(model(x_important, sub_edge_index)[local_target])
            )

        rng = np.random.default_rng(20_000 + node)
        all_pairs = list(base_exp["pair_scores"].keys())
        random_edge_deltas: List[float] = []
        random_feature_deltas: List[float] = []
        for _ in range(cfg.random_control_trials):
            random_pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), size=len(selected_pairs), replace=False)]
            keep = edge_mask_from_pairs(sub_edge_index, random_pairs, keep_selected=False)
            with torch.no_grad():
                random_edge_deltas.append(abs(origin - float(model(sub_x, sub_edge_index[:, keep])[local_target])))
            random_features = rng.choice(6, size=kf, replace=False).tolist()
            x_random = sub_x.clone()
            x_random[local_target, random_features] = 0.0
            with torch.no_grad():
                random_feature_deltas.append(abs(origin - float(model(x_random, sub_edge_index)[local_target])))

        stability_values: List[float] = []
        base_selected = set(selected_pairs)
        for run in range(cfg.stability_runs):
            noise_gen = torch.Generator().manual_seed(30_000 + node * 10 + run)
            noise = torch.randn(sub_x[:, :6].shape, generator=noise_gen) * cfg.stability_noise_sd
            perturbed = explain_local_subgraph(
                model,
                data,
                node,
                cfg,
                explanation_seed=10_000 + node,
                epochs=max(30, cfg.explainer_epochs // 2),
                x_noise=noise,
            )
            stability_values.append(jaccard(base_selected, set(perturbed["selected_pairs"])))

        row = {
            "node_index": node,
            "node_id": node_names[node],
            "target": float(data.y[node]),
            "prediction": origin,
            "fidelity_plus": float(base_exp["fidelity_plus"]),
            "fidelity_minus": float(base_exp["fidelity_minus"]),
            "characterization": float(base_exp["characterization"]),
            "sparsity": float(base_exp["sparsity"]),
            "stability": float(np.mean(stability_values)),
            "important_edge_delta": float(base_exp["fidelity_plus"]),
            "random_edge_delta_mean": float(np.mean(random_edge_deltas)),
            "important_feature_delta": float(important_feature_delta),
            "random_feature_delta_mean": float(np.mean(random_feature_deltas)),
            "random_edge_deltas": random_edge_deltas,
            "random_feature_deltas": random_feature_deltas,
            "important_feature_indices": important_features,
            "direct_feature_attributions": direct_feature_attr.detach().numpy().tolist(),
            "ig_completeness_residual": float(
                abs((origin - baseline_prediction) - float(attribution.sum()))
            ),
            "explanation_seconds": float(base_exp["explainer_seconds"] + ig_seconds),
        }
        rows.append(row)

        if representative_payload is None:
            representative_payload = {
                "node_index": node,
                "node_id": node_names[node],
                "subset_global": subset.tolist(),
                "sub_edges": sub_edge_index.t().tolist(),
                "selected_pairs": [list(pair) for pair in selected_pairs],
                "pair_scores": {f"{a}-{b}": score for (a, b), score in base_exp["pair_scores"].items()},
                "node_names": [node_names[int(i)] for i in subset],
                "node_types": [NODE_TYPES[int(data.node_type[int(i)])] for i in subset],
                "local_target": local_target,
                "target": float(data.y[node]),
                "prediction": origin,
            }

        if position % 10 == 0 or position == len(test_nodes):
            print(f"Explained {position}/{len(test_nodes)} test policyholders", flush=True)

    # Replace the provisional representative with the case closest to median absolute error.
    abs_errors = np.array([abs(float(r["prediction"]) - float(r["target"])) for r in rows])
    chosen_pos = int(np.argsort(abs_errors)[len(abs_errors) // 2])
    chosen_node = int(rows[chosen_pos]["node_index"])
    chosen_exp = explain_local_subgraph(model, data, chosen_node, cfg, explanation_seed=10_000 + chosen_node)
    subset = chosen_exp["subset"]
    representative_payload = {
        "node_index": chosen_node,
        "node_id": node_names[chosen_node],
        "subset_global": subset.tolist(),
        "sub_edges": chosen_exp["sub_edge_index"].t().tolist(),
        "selected_pairs": [list(pair) for pair in chosen_exp["selected_pairs"]],
        "pair_scores": {f"{a}-{b}": score for (a, b), score in chosen_exp["pair_scores"].items()},
        "node_names": [node_names[int(i)] for i in subset],
        "node_types": [NODE_TYPES[int(data.node_type[int(i)])] for i in subset],
        "local_target": int(chosen_exp["local_target"]),
        "target": float(data.y[chosen_node]),
        "prediction": float(chosen_exp["origin"]),
    }
    return rows, representative_payload


def numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    ci95 = 1.96 * sd / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95": float(ci95), "n": int(len(a))}


def paired_control_test(selected: Sequence[float], random_control: Sequence[float]) -> Dict[str, float]:
    """One-sided paired Wilcoxon test for larger selected-component effects."""
    selected_arr = np.asarray(selected, dtype=float)
    random_arr = np.asarray(random_control, dtype=float)
    differences = selected_arr - random_arr
    result = wilcoxon(selected_arr, random_arr, alternative="greater", zero_method="wilcox")
    sd = float(differences.std(ddof=1))
    return {
        "n_pairs": int(len(differences)),
        "selected_mean": float(selected_arr.mean()),
        "random_control_mean": float(random_arr.mean()),
        "mean_paired_difference": float(differences.mean()),
        "median_paired_difference": float(np.median(differences)),
        "paired_standardized_difference": float(differences.mean() / sd) if sd > 0 else 0.0,
        "wilcoxon_statistic": float(result.statistic),
        "one_sided_p_value": float(result.pvalue),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="analysis/experiment_output")
    parser.add_argument("--skip-explanations", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    data, metadata = generate_health_graph(cfg)
    torch.save(data, outdir / "health_risk_graph.pt")
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ["node_counts", "unique_relationships", "directed_edge_entries", "split_counts", "target_mean", "target_sd"]}, indent=2))

    all_rows: List[Dict[str, object]] = run_tabular_baselines(data, cfg)
    saved_models: Dict[Tuple[str, int], GraphRegressor] = {}
    training_details: Dict[str, object] = {}
    for kind in ["GCN", "GAT", "GraphSAGE"]:
        for seed in cfg.model_seeds:
            model, metrics, details = train_gnn(data, kind, seed, cfg)
            saved_models[(kind, seed)] = model
            training_details[f"{kind}_seed_{seed}"] = details
            all_rows.append({"model": kind, "seed": seed, **metrics})
            print(kind, seed, {k: round(float(metrics[k]), 5) for k in ["rmse", "mae", "r2", "training_seconds"]})

    # GraphSAGE ablations use the same five training seeds.
    for seed in cfg.model_seeds:
        features_only_edges = torch.empty((2, 0), dtype=torch.long)
        _, metrics, _ = train_gnn(data, "GraphSAGE", seed, cfg, edge_index_override=features_only_edges)
        all_rows.append({"model": "GraphSAGE features only", "seed": seed, **metrics})

        topology_x = data.x.clone()
        topology_x[:, :6] = 0.0
        _, metrics, _ = train_gnn(data, "GraphSAGE", seed, cfg, x_override=topology_x)
        all_rows.append({"model": "GraphSAGE topology only", "seed": seed, **metrics})

        shuffled = shuffled_edge_index(data, seed=50_000 + seed)
        _, metrics, _ = train_gnn(data, "GraphSAGE", seed, cfg, edge_index_override=shuffled)
        all_rows.append({"model": "GraphSAGE shuffled edges", "seed": seed, **metrics})

    summaries = summarize_runs(all_rows)
    write_csv(
        outdir / "predictive_runs.csv",
        all_rows,
        ["model", "seed", "rmse", "mae", "r2", "training_seconds", "full_graph_inference_ms", "best_epoch", "best_val_rmse"],
    )
    write_csv(
        outdir / "predictive_summary.csv",
        summaries,
        [
            "model", "n_runs",
            "rmse_mean", "rmse_sd", "rmse_ci95",
            "mae_mean", "mae_sd", "mae_ci95",
            "r2_mean", "r2_sd", "r2_ci95",
            "training_seconds_mean", "training_seconds_sd", "training_seconds_ci95",
            "full_graph_inference_ms_mean", "full_graph_inference_ms_sd", "full_graph_inference_ms_ci95",
        ],
    )
    (outdir / "training_details.json").write_text(json.dumps(training_details, indent=2), encoding="utf-8")
    print("\nPredictive summary")
    for row in summaries:
        print(row["model"], "RMSE", round(float(row["rmse_mean"]), 4), "R2", round(float(row["r2_mean"]), 4))

    if args.skip_explanations:
        return

    # Use the predefined seed-0 GraphSAGE model for explanation evaluation.
    explanation_model = saved_models[("GraphSAGE", 0)]
    explanation_rows, representative = evaluate_explanations(explanation_model, data, metadata, cfg)
    serializable_rows = []
    for r in explanation_rows:
        serializable_rows.append(r)
    (outdir / "explanation_population.json").write_text(json.dumps(serializable_rows, indent=2), encoding="utf-8")
    (outdir / "representative_explanation.json").write_text(json.dumps(representative, indent=2), encoding="utf-8")

    explanation_fields = [
        "node_index", "node_id", "target", "prediction", "fidelity_plus", "fidelity_minus",
        "characterization", "sparsity", "stability", "important_edge_delta",
        "random_edge_delta_mean", "important_feature_delta", "random_feature_delta_mean",
        "explanation_seconds",
        "ig_completeness_residual",
    ]
    write_csv(outdir / "explanation_population.csv", explanation_rows, explanation_fields)

    metric_summary = {
        metric: numeric_summary([float(r[metric]) for r in explanation_rows])
        for metric in [
            "fidelity_plus", "fidelity_minus", "characterization", "sparsity", "stability",
            "important_edge_delta", "random_edge_delta_mean", "important_feature_delta",
            "random_feature_delta_mean", "explanation_seconds",
            "ig_completeness_residual",
        ]
    }
    feature_attr = np.asarray([r["direct_feature_attributions"] for r in explanation_rows], dtype=float)
    metric_summary["feature_attribution"] = {
        POLICYHOLDER_FEATURES[i]: numeric_summary(feature_attr[:, i].tolist()) for i in range(6)
    }
    metric_summary["matched_control_tests"] = {
        "relationships": paired_control_test(
            [float(r["important_edge_delta"]) for r in explanation_rows],
            [float(r["random_edge_delta_mean"]) for r in explanation_rows],
        ),
        "policyholder_features": paired_control_test(
            [float(r["important_feature_delta"]) for r in explanation_rows],
            [float(r["random_feature_delta_mean"]) for r in explanation_rows],
        ),
    }
    (outdir / "explanation_summary.json").write_text(json.dumps(metric_summary, indent=2), encoding="utf-8")
    print("\nExplanation summary")
    print(json.dumps(metric_summary, indent=2))


if __name__ == "__main__":
    main()
