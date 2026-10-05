import argparse
import csv
import glob
import json
import os
from typing import Any

import numpy as np


def parse_quantiles(value: str, expected: int, name: str) -> list[float]:
    quantiles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(quantiles) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values, got {value!r}")
    if any(q <= 0.0 or q >= 1.0 for q in quantiles):
        raise ValueError(f"{name} values must be in (0, 1), got {value!r}")
    return quantiles


def load_npz(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def scalar_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size <= 1:
        return np.zeros_like(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks / float(values.size - 1)


def trapezoid_integral(values: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x, axis=axis)
    return np.trapz(values, x, axis=axis)


def label_effect(abs_delta: float, q_small: float, q_moderate: float, q_large: float) -> str:
    if abs_delta < q_small:
        return "weak"
    if abs_delta < q_moderate:
        return "small"
    if abs_delta < q_large:
        return "moderate"
    return "large"


def label_uncertainty(score: float, q_low: float, q_high: float) -> str:
    if score <= q_low:
        return "low"
    if score <= q_high:
        return "medium"
    return "high"


def direction_phrase(direction: str) -> str:
    if direction == "up":
        return "higher than control"
    if direction == "down":
        return "lower than control"
    return "near the control state"


def endpoint_direction_phrase(direction: str) -> str:
    if direction == "up":
        return "up-regulated"
    if direction == "down":
        return "down-regulated"
    return "not clearly changed"


def infer_cell_line(perturbation_dir: str, first_record: dict[str, Any]) -> str:
    if "cell_line" in first_record:
        return scalar_str(first_record["cell_line"])
    parent = os.path.basename(os.path.dirname(os.path.abspath(perturbation_dir)))
    if "=" in parent:
        return parent.split("=", 1)[1]
    return "pooled"


def cell_line_clause(cell_line: str) -> str:
    if not cell_line or cell_line == "pooled":
        return ""
    return f" in {cell_line}"


def stability_phrase(stability: str, min_stable_time: float) -> str:
    if stability == "stable":
        return f"stable from t={min_stable_time:g} to t=1.0"
    if stability == "partially_stable":
        return f"partially stable from t={min_stable_time:g} to t=1.0"
    if stability == "not_assessed":
        return "not assessed"
    return f"unstable from t={min_stable_time:g} to t=1.0"


def safe_float(value: Any) -> float:
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def time_key(t: float) -> str:
    return f"{safe_float(t):g}"


def signed_deviation_label(delta: float, effect: str, threshold: float) -> str:
    if abs(delta) < threshold:
        return "near-control"
    sign = "positive" if delta > 0 else "negative"
    return f"{effect} {sign}"


def onset_label(times: np.ndarray, series: np.ndarray, direction_sign: float, threshold: float) -> str:
    if direction_sign == 0.0:
        return "none"
    hits = np.where(series * direction_sign >= threshold)[0]
    if hits.size == 0:
        return "none"
    first_idx = int(hits[0])
    frac = first_idx / max(len(times) - 1, 1)
    if frac <= 0.34:
        return "early"
    if frac <= 0.67:
        return "mid"
    return "late"


def classify_trajectory(series: np.ndarray, threshold: float, direction_sign: float) -> str:
    series = np.asarray(series, dtype=float)
    abs_series = np.abs(series)
    if abs_series.max() < threshold:
        return "flat"

    significant = abs_series >= threshold
    significant_signs = np.sign(series[significant])
    if significant_signs.size >= 2 and np.any(significant_signs != significant_signs[0]):
        return "reversal"

    diffs = np.diff(series)
    nonzero_diffs = diffs[np.abs(diffs) >= max(threshold * 0.25, 1e-8)]
    if direction_sign > 0:
        if nonzero_diffs.size == 0 or np.mean(nonzero_diffs >= 0.0) >= 0.75:
            return "monotonic_increase"
        if abs(series[0]) < threshold and series[-1] >= threshold:
            return "delayed_increase"
        if series.max() >= threshold and series[-1] < 0.5 * series.max():
            return "transient_up"
    elif direction_sign < 0:
        if nonzero_diffs.size == 0 or np.mean(nonzero_diffs <= 0.0) >= 0.75:
            return "monotonic_decrease"
        if abs(series[0]) < threshold and series[-1] <= -threshold:
            return "delayed_decrease"
        if series.min() <= -threshold and abs(series[-1]) < 0.5 * abs(series.min()):
            return "transient_down"

    return "mixed_path"


def classify_velocity(
    velocity_series: np.ndarray,
    direction_sign: float,
    velocity_auc: float,
    velocity_auc_threshold: float,
) -> str:
    velocity_series = np.asarray(velocity_series, dtype=float)
    if velocity_auc < velocity_auc_threshold or np.abs(velocity_series).max() < 1e-8:
        return "weak"

    signs = np.sign(velocity_series)
    if direction_sign > 0 and np.mean(signs > 0.0) >= 0.75:
        return "sustained_positive"
    if direction_sign < 0 and np.mean(signs < 0.0) >= 0.75:
        return "sustained_negative"

    midpoint = max(len(velocity_series) // 2, 1)
    early = float(np.abs(velocity_series[:midpoint]).mean())
    late = float(np.abs(velocity_series[midpoint:]).mean()) if midpoint < len(velocity_series) else early
    if early > late * 1.5:
        return "early_push"
    if late > early * 1.5:
        return "late_push"
    if np.abs(velocity_series[-1]) < np.abs(velocity_series[0]) * 0.5:
        return "decelerating"
    return "fluctuating"


def trajectory_pattern_phrase(pattern: str) -> str:
    phrases = {
        "flat": "stays close to the control state",
        "monotonic_increase": "increases monotonically along the ODE path",
        "monotonic_decrease": "decreases monotonically along the ODE path",
        "delayed_increase": "shows a delayed increase along the ODE path",
        "delayed_decrease": "shows a delayed decrease along the ODE path",
        "transient_up": "shows a transient upward excursion along the ODE path",
        "transient_down": "shows a transient downward excursion along the ODE path",
        "reversal": "changes direction along the ODE path",
        "mixed_path": "shows a mixed trajectory along the ODE path",
    }
    return phrases.get(pattern, pattern)


def velocity_pattern_phrase(pattern: str) -> str:
    phrases = {
        "weak": "weak velocity signal",
        "sustained_positive": "sustained positive velocity",
        "sustained_negative": "sustained negative velocity",
        "early_push": "stronger early velocity",
        "late_push": "stronger late velocity",
        "decelerating": "decelerating velocity",
        "fluctuating": "fluctuating velocity",
    }
    return phrases.get(pattern, pattern)


def make_path_state_sentence(
    gene: str,
    perturbation: str,
    cell_line: str,
    times: np.ndarray,
    series: np.ndarray,
    effect_labels: list[str],
    threshold: float,
    trajectory_pattern: str,
    onset: str,
) -> str:
    state_parts = []
    for t, delta, effect in zip(times, series, effect_labels):
        state_parts.append(f"t={time_key(t)}: {signed_deviation_label(delta, effect, threshold)}")
    return (
        f"Along the scDFM ODE trajectory under {perturbation}{cell_line_clause(cell_line)}, "
        f"{gene} path-state deviations from control are {', '.join(state_parts)}; "
        f"overall the gene {trajectory_pattern_phrase(trajectory_pattern)}, with {onset} onset."
    )


def make_velocity_sentence(
    gene: str,
    times: np.ndarray,
    velocity_series: np.ndarray,
    velocity_pattern: str,
) -> str:
    values = ", ".join(f"t={time_key(t)}: {safe_float(v):.4g}" for t, v in zip(times, velocity_series))
    return (
        f"Velocity evidence for {gene}: {velocity_pattern_phrase(velocity_pattern)} "
        f"across the saved ODE states ({values})."
    )


def make_endpoint_sentence(
    gene: str,
    direction: str,
    effect: str,
    stability: str,
    min_stable_time: float,
    uncertainty: str,
) -> str:
    return (
        f"Endpoint forecast: {gene} is {endpoint_direction_phrase(direction)} at the final ODE endpoint; "
        f"endpoint effect is {effect}, path direction stability is {stability_phrase(stability, min_stable_time)}, "
        f"and uncertainty is {uncertainty}."
    )


def build_program_context(
    genes: np.ndarray,
    times: np.ndarray,
    path_delta_by_time: np.ndarray,
    late_delta: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    if top_k <= 0:
        return {
            "top_up_genes": [],
            "top_down_genes": [],
            "top_up_genes_by_time": {},
            "top_down_genes_by_time": {},
            "program_sentence": "",
        }

    top_k = min(top_k, genes.size)
    up_idx = np.argsort(late_delta)[-top_k:][::-1]
    down_idx = np.argsort(late_delta)[:top_k]
    top_up_genes = [str(genes[i]) for i in up_idx]
    top_down_genes = [str(genes[i]) for i in down_idx]

    top_up_by_time = {}
    top_down_by_time = {}
    for t_idx, t in enumerate(times):
        delta_t = path_delta_by_time[t_idx]
        time_up_idx = np.argsort(delta_t)[-top_k:][::-1]
        time_down_idx = np.argsort(delta_t)[:top_k]
        top_up_by_time[time_key(t)] = [str(genes[i]) for i in time_up_idx]
        top_down_by_time[time_key(t)] = [str(genes[i]) for i in time_down_idx]

    program_sentence = (
        f"Late ODE path top higher-than-control genes include {', '.join(top_up_genes[:5])}; "
        f"late ODE path top lower-than-control genes include {', '.join(top_down_genes[:5])}."
    )
    return {
        "top_up_genes": top_up_genes,
        "top_down_genes": top_down_genes,
        "top_up_genes_by_time": top_up_by_time,
        "top_down_genes_by_time": top_down_by_time,
        "program_sentence": program_sentence,
    }


def build_rows_for_perturbation(
    perturbation_dir: str,
    effect_quantiles: list[float],
    uncertainty_quantiles: list[float],
    effect_threshold_quantile: float,
    min_stable_time: float,
    max_genes_per_perturbation: int,
    program_top_k: int,
) -> list[dict[str, Any]]:
    seed_files = sorted(glob.glob(os.path.join(perturbation_dir, "seed_*.npz")))
    if not seed_files:
        return []

    seed_data = [load_npz(path) for path in seed_files]
    first = seed_data[0]
    required = [
        "perturbation",
        "times",
        "gene_names",
        "source_mean",
        "source_var",
        "x_t_mean",
        "x1_hat_mean",
        "velocity_mean",
        "final_mean",
        "final_var",
    ]
    missing = [key for key in required if key not in first]
    if missing:
        raise KeyError(f"{perturbation_dir} is missing required keys: {missing}")

    perturbation = scalar_str(first["perturbation"])
    cell_line = infer_cell_line(perturbation_dir, first)
    times = np.asarray(first["times"], dtype=float)
    genes = np.asarray(first["gene_names"]).astype(str)
    source_mean = np.asarray(first["source_mean"], dtype=float)
    source_var = np.asarray(first["source_var"], dtype=float)

    x_t = np.stack([np.asarray(item["x_t_mean"], dtype=float) for item in seed_data], axis=0)
    x1_hat = np.stack([np.asarray(item["x1_hat_mean"], dtype=float) for item in seed_data], axis=0)
    velocity = np.stack([np.asarray(item["velocity_mean"], dtype=float) for item in seed_data], axis=0)
    final = np.stack([np.asarray(item["final_mean"], dtype=float) for item in seed_data], axis=0)
    final_var = np.stack([np.asarray(item["final_var"], dtype=float) for item in seed_data], axis=0).mean(axis=0)

    delta_xt_seed = x_t - source_mean[None, None, :]
    delta_xhat_seed = x1_hat - source_mean[None, None, :]
    path_delta_by_time = delta_xt_seed.mean(axis=0)
    velocity_by_time = velocity.mean(axis=0)
    xhat_delta_by_time = delta_xhat_seed.mean(axis=0)
    delta_final = final - source_mean[None, :]
    endpoint_delta = delta_final.mean(axis=0)

    late_time_mask = times >= min_stable_time
    if not late_time_mask.any():
        late_time_mask = np.ones_like(times, dtype=bool)

    late_path_delta_seed = delta_xt_seed[:, late_time_mask, :].mean(axis=1)
    late_path_delta = late_path_delta_seed.mean(axis=0)
    path_abs_max = np.abs(path_delta_by_time).max(axis=0)

    q_small, q_moderate, q_large = np.quantile(path_abs_max, effect_quantiles)
    effect_threshold = float(np.quantile(path_abs_max, effect_threshold_quantile))

    seed_std = late_path_delta_seed.std(axis=0)
    endpoint_width = np.sqrt(np.maximum(final_var, 0.0))
    control_width = np.sqrt(np.maximum(source_var, 0.0))
    path_width = delta_xt_seed.std(axis=0).mean(axis=0)
    uncertainty_score = seed_std + 0.25 * endpoint_width + 0.25 * control_width + 0.25 * path_width
    q_unc_low, q_unc_high = np.quantile(uncertainty_score, uncertainty_quantiles)

    velocity_auc = trapezoid_integral(np.abs(velocity).mean(axis=0), times, axis=0)
    velocity_auc_threshold = float(np.quantile(velocity_auc, effect_threshold_quantile))
    path_abs_rank = percentile_ranks(path_abs_max)
    endpoint_abs_rank = percentile_ranks(np.abs(endpoint_delta))
    velocity_auc_rank = percentile_ranks(velocity_auc)
    uncertainty_rank = percentile_ranks(uncertainty_score)

    program_context = build_program_context(genes, times, path_delta_by_time, late_path_delta, program_top_k)

    gene_order = np.argsort(path_abs_max)[::-1]
    if max_genes_per_perturbation > 0:
        gene_order = gene_order[:max_genes_per_perturbation]

    rows = []
    for gene_idx in gene_order:
        late_value = late_path_delta[gene_idx]
        path_abs_value = path_abs_max[gene_idx]
        endpoint_value = endpoint_delta[gene_idx]

        if late_value > effect_threshold:
            direction = "up"
            majority_sign = 1.0
        elif late_value < -effect_threshold:
            direction = "down"
            majority_sign = -1.0
        else:
            direction = "no_clear_change"
            majority_sign = 0.0

        if endpoint_value > effect_threshold:
            endpoint_direction = "up"
        elif endpoint_value < -effect_threshold:
            endpoint_direction = "down"
        else:
            endpoint_direction = "no_clear_change"

        if majority_sign == 0.0:
            sign_stability = 0.0
            stability = "not_assessed"
        else:
            signs = np.sign(delta_xt_seed[:, late_time_mask, gene_idx])
            sign_stability = float(np.mean(signs == majority_sign))
            if sign_stability >= 0.8:
                stability = "stable"
            elif sign_stability >= 0.5:
                stability = "partially_stable"
            else:
                stability = "unstable"

        effect = label_effect(path_abs_value, q_small, q_moderate, q_large)
        endpoint_effect = label_effect(abs(endpoint_value), q_small, q_moderate, q_large)
        if len(seed_files) < 2:
            uncertainty = "not_estimated"
        else:
            uncertainty = label_uncertainty(uncertainty_score[gene_idx], q_unc_low, q_unc_high)

        series = path_delta_by_time[:, gene_idx]
        velocity_series = velocity_by_time[:, gene_idx]
        trajectory_pattern = classify_trajectory(series, effect_threshold, majority_sign)
        onset = onset_label(times, series, majority_sign, effect_threshold)
        velocity_pattern = classify_velocity(
            velocity_series,
            majority_sign,
            velocity_auc[gene_idx],
            velocity_auc_threshold,
        )
        time_effect_labels = [label_effect(abs(value), q_small, q_moderate, q_large) for value in series]

        path_state_sentence = make_path_state_sentence(
            gene=str(genes[gene_idx]),
            perturbation=perturbation,
            cell_line=cell_line,
            times=times,
            series=series,
            effect_labels=time_effect_labels,
            threshold=effect_threshold,
            trajectory_pattern=trajectory_pattern,
            onset=onset,
        )
        velocity_sentence = make_velocity_sentence(
            gene=str(genes[gene_idx]),
            times=times,
            velocity_series=velocity_series,
            velocity_pattern=velocity_pattern,
        )
        endpoint_forecast_sentence = make_endpoint_sentence(
            gene=str(genes[gene_idx]),
            direction=endpoint_direction,
            effect=endpoint_effect,
            stability=stability,
            min_stable_time=min_stable_time,
            uncertainty=uncertainty,
        )

        simcontext = (
            f"{path_state_sentence} {velocity_sentence} "
            f"{endpoint_forecast_sentence} {program_context['program_sentence']}"
        ).strip()

        row = {
            "perturbation": perturbation,
            "cell_line": cell_line,
            "gene": str(genes[gene_idx]),
            "direction": direction,
            "trajectory_direction": direction,
            "endpoint_direction": endpoint_direction,
            "effect_size": effect,
            "path_effect_size": effect,
            "endpoint_effect_size": endpoint_effect,
            "stability": stability,
            "uncertainty": uncertainty,
            "trajectory_pattern": trajectory_pattern,
            "onset": onset,
            "velocity_pattern": velocity_pattern,
            "path_state_sentence": path_state_sentence,
            "velocity_sentence": velocity_sentence,
            "endpoint_forecast_sentence": endpoint_forecast_sentence,
            "simcontext": simcontext,
            "delta_xt_by_time": {
                time_key(t): safe_float(series[t_idx])
                for t_idx, t in enumerate(times)
            },
            "velocity_by_time": {
                time_key(t): safe_float(velocity_series[t_idx])
                for t_idx, t in enumerate(times)
            },
            "x1_hat_delta_by_time": {
                time_key(t): safe_float(xhat_delta_by_time[t_idx, gene_idx])
                for t_idx, t in enumerate(times)
            },
            "time_state_labels": {
                time_key(t): signed_deviation_label(series[t_idx], time_effect_labels[t_idx], effect_threshold)
                for t_idx, t in enumerate(times)
            },
            "path_late_delta": safe_float(late_value),
            "path_abs_max": safe_float(path_abs_value),
            "path_peak_time": time_key(times[int(np.argmax(np.abs(series)))]),
            "delta_final": safe_float(endpoint_value),
            "abs_delta_final": safe_float(abs(endpoint_value)),
            "path_abs_rank_percentile": safe_float(path_abs_rank[gene_idx]),
            "abs_delta_rank_percentile": safe_float(endpoint_abs_rank[gene_idx]),
            "sign_stability": safe_float(sign_stability),
            "velocity_auc": safe_float(velocity_auc[gene_idx]),
            "velocity_auc_rank_percentile": safe_float(velocity_auc_rank[gene_idx]),
            "uncertainty_score": safe_float(uncertainty_score[gene_idx]),
            "uncertainty_rank_percentile": safe_float(uncertainty_rank[gene_idx]),
            "n_seeds": len(seed_files),
            "trace_times": [safe_float(t) for t in times.tolist()],
            "source_mean": safe_float(source_mean[gene_idx]),
            "source_var": safe_float(source_var[gene_idx]),
            "final_mean": safe_float(final[:, gene_idx].mean()),
            "final_var": safe_float(final_var[gene_idx]),
            "top_up_genes": program_context["top_up_genes"],
            "top_down_genes": program_context["top_down_genes"],
            "top_up_genes_by_time": program_context["top_up_genes_by_time"],
            "top_down_genes_by_time": program_context["top_down_genes_by_time"],
        }

        if "target_mean" in first:
            target_mean = np.stack([np.asarray(item["target_mean"], dtype=float) for item in seed_data], axis=0).mean(axis=0)
            row["target_delta"] = safe_float(target_mean[gene_idx] - source_mean[gene_idx])

        rows.append(row)

    return rows


def write_jsonl(rows: list[dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_summary(rows: list[dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fields = [
        "perturbation",
        "cell_line",
        "gene",
        "direction",
        "path_effect_size",
        "trajectory_pattern",
        "onset",
        "velocity_pattern",
        "stability",
        "uncertainty",
        "path_late_delta",
        "path_abs_max",
        "path_abs_rank_percentile",
        "delta_final",
        "abs_delta_rank_percentile",
        "sign_stability",
        "velocity_auc_rank_percentile",
        "uncertainty_rank_percentile",
        "n_seeds",
        "path_state_sentence",
        "velocity_sentence",
        "endpoint_forecast_sentence",
        "simcontext",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def collect_perturbation_dirs(trace_dir: str) -> list[str]:
    """Accept an experiment dir, a trace dir, or a single perturbation dir."""
    trace_dir = os.path.abspath(trace_dir)
    candidate_roots = [trace_dir]

    for suffix in ("trace", os.path.join("test_only", "trace")):
        candidate = os.path.join(trace_dir, suffix)
        if os.path.isdir(candidate):
            candidate_roots.append(candidate)

    perturbation_dirs = []
    seen = set()
    for root in candidate_roots:
        if glob.glob(os.path.join(root, "seed_*.npz")):
            if root not in seen:
                perturbation_dirs.append(root)
                seen.add(root)

        for path in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(path) and glob.glob(os.path.join(path, "seed_*.npz")):
                if path not in seen:
                    perturbation_dirs.append(path)
                    seen.add(path)

    if perturbation_dirs:
        return perturbation_dirs

    seed_files = sorted(glob.glob(os.path.join(trace_dir, "**", "seed_*.npz"), recursive=True))
    for seed_file in seed_files:
        path = os.path.dirname(seed_file)
        if path not in seen:
            perturbation_dirs.append(path)
            seen.add(path)
    return perturbation_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build trajectory-oriented gene-level SimContext JSONL from scDFM ODE trace npz files."
    )
    parser.add_argument("--trace-dir", required=True, help="Directory containing trace seed files.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--csv-out", default="", help="Optional compact CSV summary path.")
    parser.add_argument(
        "--max-genes-per-perturbation",
        type=int,
        default=0,
        help="0 means all genes; otherwise keep top genes by max |X_t - control| along the ODE path.",
    )
    parser.add_argument("--program-top-k", type=int, default=0, help="Include ODE-time top up/down gene lists.")
    parser.add_argument("--min-stable-time", type=float, default=0.5, help="Use times >= this value for direction-stability assessment.")
    parser.add_argument("--effect-quantiles", default="0.60,0.80,0.95", help="Quantiles for weak/small/moderate/large labels.")
    parser.add_argument("--effect-threshold-quantile", type=float, default=0.60, help="Abs-delta quantile used as up/down threshold.")
    parser.add_argument("--uncertainty-quantiles", default="0.33,0.66", help="Quantiles for low/medium/high uncertainty labels.")
    args = parser.parse_args()

    effect_quantiles = parse_quantiles(args.effect_quantiles, 3, "--effect-quantiles")
    uncertainty_quantiles = parse_quantiles(args.uncertainty_quantiles, 2, "--uncertainty-quantiles")
    if args.effect_threshold_quantile <= 0.0 or args.effect_threshold_quantile >= 1.0:
        raise ValueError("--effect-threshold-quantile must be in (0, 1)")

    perturbation_dirs = collect_perturbation_dirs(args.trace_dir)
    if not perturbation_dirs:
        raise FileNotFoundError(f"No perturbation seed files found under {args.trace_dir}")

    rows = []
    for perturbation_dir in perturbation_dirs:
        rows.extend(
            build_rows_for_perturbation(
                perturbation_dir=perturbation_dir,
                effect_quantiles=effect_quantiles,
                uncertainty_quantiles=uncertainty_quantiles,
                effect_threshold_quantile=args.effect_threshold_quantile,
                min_stable_time=args.min_stable_time,
                max_genes_per_perturbation=args.max_genes_per_perturbation,
                program_top_k=args.program_top_k,
            )
        )

    write_jsonl(rows, args.out)
    if args.csv_out:
        write_csv_summary(rows, args.csv_out)

    print(f"processed perturbation/cell-line trace dirs: {len(perturbation_dirs)}")
    print(f"wrote trajectory SimContext rows: {len(rows)}")
    print(f"jsonl: {args.out}")
    if args.csv_out:
        print(f"csv: {args.csv_out}")


if __name__ == "__main__":
    main()
