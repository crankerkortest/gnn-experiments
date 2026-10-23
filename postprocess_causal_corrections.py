"""Correct numerical no-op effects and document capped top-k selections.

This script operates only on deposited CSV outputs. It does not train a model
or rerun an explainer. Values with absolute magnitude below 1e-12 are treated
as numerical zero. For sign agreement and nonzero-only rank correlation, the
denominator contains only interventions with a nonzero generator-defined
effect.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr, t as student_t


ZERO_TOLERANCE = 1e-12
DATA_SEEDS = tuple(range(2026, 2036))
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "extended_experiment_output"
ROBUSTNESS_OUTPUT = ROOT / "robustness_experiment_output"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_zero(value: float) -> float:
    return 0.0 if abs(value) < ZERO_TOLERANCE else float(value)


def scalar_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    critical = float(student_t.ppf(0.975, len(array) - 1)) if len(array) > 1 else 0.0
    half_width = critical * sd / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "n_graphs": int(len(array)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def finite_spearman(first: np.ndarray, second: np.ndarray) -> float:
    value = float(spearmanr(first, second).statistic)
    return value if math.isfinite(value) else 0.0


def correct_interventions() -> tuple[list[dict[str, object]], dict[str, object]]:
    source = read_csv(OUTPUT / "synthetic_causal_interventions.csv")
    corrected: list[dict[str, object]] = []
    for row in source:
        true_raw = float(row["true_effect"])
        model_raw = float(row["model_effect"])
        true_effect = normalize_zero(true_raw)
        model_effect = normalize_zero(model_raw)
        is_noop = abs(true_effect) < ZERO_TOLERANCE
        corrected.append(
            {
                "data_seed": int(row["data_seed"]),
                "model_seed": int(row["model_seed"]),
                "node_index": int(row["node_index"]),
                "variable": row["variable"],
                "true_effect_raw": true_raw,
                "model_effect_raw": model_raw,
                "true_effect": true_effect,
                "model_effect": model_effect,
                "absolute_effect_error": abs(true_effect - model_effect),
                "is_noop": int(is_noop),
                "sign_agreement_nonzero": (
                    ""
                    if is_noop
                    else int(np.sign(true_effect) == np.sign(model_effect))
                ),
            }
        )

    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in corrected:
        by_seed[int(row["data_seed"])].append(row)

    graph_rows: list[dict[str, object]] = []
    for seed in DATA_SEEDS:
        rows = by_seed[seed]
        true_raw = np.asarray([float(row["true_effect_raw"]) for row in rows])
        model_raw = np.asarray([float(row["model_effect_raw"]) for row in rows])
        true_effect = np.asarray([float(row["true_effect"]) for row in rows])
        model_effect = np.asarray([float(row["model_effect"]) for row in rows])
        nonzero = np.abs(true_effect) >= ZERO_TOLERANCE
        graph_rows.append(
            {
                "data_seed": seed,
                "spearman_raw": finite_spearman(true_raw, model_raw),
                "spearman_zero_normalized": finite_spearman(true_effect, model_effect),
                "spearman_nonzero": finite_spearman(
                    true_effect[nonzero], model_effect[nonzero]
                ),
                "sign_agreement_nonzero": float(
                    np.mean(
                        np.sign(true_effect[nonzero])
                        == np.sign(model_effect[nonzero])
                    )
                ),
                "effect_mae": float(np.mean(np.abs(true_effect - model_effect))),
                "n_total": int(len(rows)),
                "n_noop": int((~nonzero).sum()),
                "n_nonzero": int(nonzero.sum()),
            }
        )

    variable_rows: list[dict[str, object]] = []
    for variable in sorted({str(row["variable"]) for row in corrected}):
        variable_graph_rows = []
        all_rows = [row for row in corrected if row["variable"] == variable]
        for seed in DATA_SEEDS:
            rows = [row for row in all_rows if int(row["data_seed"]) == seed]
            true_raw = np.asarray([float(row["true_effect_raw"]) for row in rows])
            model_raw = np.asarray([float(row["model_effect_raw"]) for row in rows])
            true_effect = np.asarray([float(row["true_effect"]) for row in rows])
            model_effect = np.asarray([float(row["model_effect"]) for row in rows])
            nonzero = np.abs(true_effect) >= ZERO_TOLERANCE
            variable_graph_rows.append(
                {
                    "spearman_raw": finite_spearman(true_raw, model_raw),
                    "spearman_zero_normalized": finite_spearman(
                        true_effect, model_effect
                    ),
                    "spearman_nonzero": finite_spearman(
                        true_effect[nonzero], model_effect[nonzero]
                    ),
                    "normalized_mae": float(
                        np.mean(np.abs(true_effect - model_effect))
                        / np.mean(np.abs(true_effect))
                    ),
                }
            )
        true_all = np.asarray([float(row["true_effect"]) for row in all_rows])
        model_all = np.asarray([float(row["model_effect"]) for row in all_rows])
        raw_stats = scalar_summary(row["spearman_raw"] for row in variable_graph_rows)
        normalized_stats = scalar_summary(
            row["spearman_zero_normalized"] for row in variable_graph_rows
        )
        nonzero_stats = scalar_summary(
            row["spearman_nonzero"] for row in variable_graph_rows
        )
        variable_rows.append(
            {
                "variable": variable,
                "n_total": len(all_rows),
                "n_noop": int(np.sum(np.abs(true_all) < ZERO_TOLERANCE)),
                "n_nonzero": int(np.sum(np.abs(true_all) >= ZERO_TOLERANCE)),
                "mean_generator_effect": float(true_all.mean()),
                "mean_model_effect": float(model_all.mean()),
                "effect_mae": float(np.mean(np.abs(true_all - model_all))),
                "normalized_mae_mean": float(
                    np.mean([row["normalized_mae"] for row in variable_graph_rows])
                ),
                "spearman_raw_mean": raw_stats["mean"],
                "spearman_zero_normalized_mean": normalized_stats["mean"],
                "spearman_zero_normalized_sd": normalized_stats["sd"],
                "spearman_zero_normalized_ci95_low": normalized_stats["ci95_low"],
                "spearman_zero_normalized_ci95_high": normalized_stats["ci95_high"],
                "spearman_nonzero_mean": nonzero_stats["mean"],
                "spearman_nonzero_sd": nonzero_stats["sd"],
                "spearman_nonzero_ci95_low": nonzero_stats["ci95_low"],
                "spearman_nonzero_ci95_high": nonzero_stats["ci95_high"],
            }
        )

    summary = {
        "postprocessing_version": "1.0.2",
        "numerical_zero_tolerance": ZERO_TOLERANCE,
        "counts": {
            "n_total": len(corrected),
            "n_noop": sum(int(row["is_noop"]) for row in corrected),
            "n_nonzero": sum(1 - int(row["is_noop"]) for row in corrected),
        },
        "aggregate_across_graphs": {
            metric: scalar_summary(float(row[metric]) for row in graph_rows)
            for metric in (
                "spearman_raw",
                "spearman_zero_normalized",
                "spearman_nonzero",
                "sign_agreement_nonzero",
                "effect_mae",
            )
        },
        "per_graph": graph_rows,
        "variables": variable_rows,
    }
    write_csv(OUTPUT / "synthetic_causal_interventions_corrected.csv", corrected)
    write_csv(OUTPUT / "causal_intervention_variable_summary_corrected.csv", variable_rows)
    write_csv(OUTPUT / "causal_intervention_graph_summary_corrected.csv", graph_rows)
    (OUTPUT / "causal_intervention_summary_corrected.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return corrected, summary


def document_capped_sizes() -> dict[str, object]:
    rows = read_csv(OUTPUT / "explanation_size_sensitivity.csv")
    grouped: dict[tuple[int, int], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["data_seed"]), int(row["node_index"]))][
            int(row["top_k"])
        ] = row

    corrected_rows: list[dict[str, object]] = []
    counts: dict[int, Counter[int]] = {k: Counter() for k in (2, 4, 6, 8)}
    for _, size_rows in grouped.items():
        top_four = size_rows[4]
        sparsity = float(top_four["sparsity"])
        available = int(round(4.0 / max(1.0 - sparsity, ZERO_TOLERANCE)))
        if available < 4:
            raise ValueError("Baseline top-four case unexpectedly has fewer than four edges")
        for k, row in sorted(size_rows.items()):
            selected = min(k, available)
            counts[k][selected] += 1
            corrected_rows.append(
                {
                    **row,
                    "top_k": k,
                    "selected_edge_count": selected,
                    "available_local_edge_count": available,
                }
            )

    summary_rows = [
        {
            "nominal_top_k": k,
            "selected_edge_count": selected,
            "case_count": count,
        }
        for k in sorted(counts)
        for selected, count in sorted(counts[k].items())
    ]
    write_csv(OUTPUT / "explanation_size_sensitivity_corrected.csv", corrected_rows)
    write_csv(OUTPUT / "top_k_selection_size_summary.csv", summary_rows)

    provider_rows = [
        row
        for row in read_csv(ROBUSTNESS_OUTPUT / "robustness_node_metrics.csv")
        if row["condition"] == "causal_path_removed"
    ]
    provider_sizes: Counter[int] = Counter()
    provider_pairs: Counter[tuple[float, float]] = Counter()
    for row in provider_rows:
        precision = float(row["causal_edge_precision"])
        recall = float(row["causal_edge_recall"])
        overlap = int(round(3 * recall))
        selected = int(round(overlap / precision)) if precision > 0.0 else 4
        provider_sizes[selected] += 1
        provider_pairs[(precision, recall)] += 1
    provider_summary = {
        "condition": "causal_path_removed",
        "nominal_top_k": 4,
        "truth_edge_count": 3,
        "selected_size_counts": dict(sorted(provider_sizes.items())),
        "precision_recall_pair_counts": [
            {"precision": pair[0], "recall": pair[1], "case_count": count}
            for pair, count in sorted(provider_pairs.items())
        ],
    }
    (ROBUSTNESS_OUTPUT / "provider_path_capped_top_k_summary.json").write_text(
        json.dumps(provider_summary, indent=2), encoding="utf-8"
    )
    return {
        "baseline_size_counts": {
            str(k): {str(size): count for size, count in sorted(value.items())}
            for k, value in counts.items()
        },
        "provider_path_removed": provider_summary,
    }


def main() -> None:
    _, causal_summary = correct_interventions()
    capped_summary = document_capped_sizes()
    report = {
        "causal_intervention_correction": causal_summary,
        "capped_top_k": capped_summary,
    }
    (OUTPUT / "supervisor_correction_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
