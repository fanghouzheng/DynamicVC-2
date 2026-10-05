import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any


GENERIC_GO_TERMS = {
    "protein binding",
    "cytoplasm",
    "nucleus",
    "cytosol",
    "plasma membrane",
    "membrane",
    "extracellular exosome",
    "extracellular region",
    "cellular component",
    "biological process",
    "molecular function",
}


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def logcomb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(k: int, population_size: int, success_count: int, draw_count: int) -> float:
    max_k = min(success_count, draw_count)
    if k > max_k:
        return 1.0
    denominator = logcomb(population_size, draw_count)
    log_terms = [
        logcomb(success_count, i) + logcomb(population_size - success_count, draw_count - i) - denominator
        for i in range(k, max_k + 1)
    ]
    max_log = max(log_terms)
    return math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)


def bh_fdr(hit_rows: list[dict[str, Any]], total_tests: int) -> None:
    if not hit_rows:
        return
    ordered = sorted(enumerate(hit_rows), key=lambda item: item[1]["p_value"])
    adjusted = [1.0] * len(ordered)
    running_min = 1.0
    for rank_from_end, (original_idx, row) in enumerate(reversed(ordered), start=1):
        rank = len(ordered) - rank_from_end + 1
        q_value = row["p_value"] * total_tests / max(rank, 1)
        running_min = min(running_min, q_value)
        adjusted[original_idx] = min(running_min, 1.0)
    for row, q_value in zip(hit_rows, adjusted):
        row["fdr"] = q_value


def group_by_query_context(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["perturbation"], row.get("cell_line", "pooled"))].append(row)
    return dict(grouped)


def cell_line_clause(cell_line: str) -> str:
    if not cell_line or cell_line == "pooled":
        return ""
    return f" in {cell_line}"


def time_key_value(key: str) -> float:
    try:
        return float(key)
    except ValueError:
        return 0.0


def available_trace_times(rows: list[dict[str, Any]]) -> list[str]:
    times = set()
    for row in rows:
        for key in row.get("delta_xt_by_time", {}).keys():
            times.add(str(key))
    if times:
        return sorted(times, key=time_key_value)

    fallback = rows[0].get("trace_times", []) if rows else []
    return [f"{float(t):g}" for t in fallback]


def delta_at_time(row: dict[str, Any], time: str) -> float:
    deltas = row.get("delta_xt_by_time", {})
    if time in deltas:
        return float(deltas[time])
    compatible = f"{float(time):g}"
    if compatible in deltas:
        return float(deltas[compatible])
    return float(row.get("delta_final", 0.0))


def row_matches_direction_at_time(row: dict[str, Any], time: str, direction: str) -> bool:
    delta = delta_at_time(row, time)
    if direction == "up":
        return delta > 0.0
    return delta < 0.0


def select_direction_genes_at_time(
    rows: list[dict[str, Any]],
    time: str,
    direction: str,
    min_selected: int,
    fallback_top_n: int,
) -> tuple[set[str], str]:
    strong = [
        row for row in rows
        if row_matches_direction_at_time(row, time, direction)
    ]
    strong = [
        row for row in strong
        if row.get("path_effect_size", row.get("effect_size")) in {"large", "moderate"}
        and row.get("stability") != "unstable"
    ]
    if len(strong) >= min_selected:
        return {row["gene"] for row in strong}, f"strong_path_effect_non_unstable_at_t_{time}"

    if direction == "up":
        candidates = [row for row in rows if delta_at_time(row, time) > 0.0]
        candidates.sort(key=lambda row: delta_at_time(row, time), reverse=True)
    else:
        candidates = [row for row in rows if delta_at_time(row, time) < 0.0]
        candidates.sort(key=lambda row: delta_at_time(row, time))

    return {row["gene"] for row in candidates[:fallback_top_n]}, f"fallback_top_{fallback_top_n}_by_delta_xt_at_t_{time}"


def load_gene_sets(kg_dir: str) -> dict[str, dict[str, Any]]:
    go_sets = load_json(os.path.join(kg_dir, "go_gsea.json"))
    reactome_sets = load_json(os.path.join(kg_dir, "reactome_gsea.json"))
    go_names_path = os.path.join(kg_dir, "go_dict.json")
    go_names = load_json(go_names_path) if os.path.exists(go_names_path) else {}
    return {
        "reactome": {
            "sets": reactome_sets,
            "names": {},
            "min_overlap": 3,
            "min_size": 3,
            "max_size": 200,
        },
        "go": {
            "sets": go_sets,
            "names": go_names,
            "min_overlap": 4,
            "min_size": 5,
            "max_size": 100,
        },
    }


def is_generic_go_term(term_name: str) -> bool:
    return term_name.strip().lower() in GENERIC_GO_TERMS


def enrich_one_source(
    perturbation: str,
    cell_line: str,
    time: str,
    direction: str,
    selected_genes: set[str],
    selection_rule: str,
    background_genes: set[str],
    source: str,
    source_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not selected_genes:
        return []

    gene_sets = source_config["sets"]
    term_names = source_config["names"]
    min_overlap = source_config["min_overlap"]
    min_size = source_config["min_size"]
    max_size = source_config["max_size"]

    population_size = len(background_genes)
    draw_count = len(selected_genes)
    tested_terms = 0
    hit_rows = []

    for term_id, genes in gene_sets.items():
        term_name = term_names.get(term_id, term_id)
        if source == "go" and is_generic_go_term(term_name):
            continue

        term_genes = set(genes) & background_genes
        term_size = len(term_genes)
        if term_size < min_size or term_size > max_size:
            continue

        tested_terms += 1
        overlap_genes = sorted(term_genes & selected_genes)
        overlap = len(overlap_genes)
        if overlap < min_overlap:
            continue

        p_value = hypergeom_sf(overlap, population_size, term_size, draw_count)
        hit_rows.append({
            "perturbation": perturbation,
            "cell_line": cell_line,
            "time": time,
            "direction": direction,
            "source": source,
            "term_id": term_id,
            "term_name": term_name,
            "p_value": p_value,
            "fdr": 1.0,
            "overlap": overlap,
            "term_size_in_background": term_size,
            "selected_gene_count": draw_count,
            "background_gene_count": population_size,
            "tested_terms": tested_terms,
            "selection_rule": selection_rule,
            "overlap_genes": overlap_genes,
        })

    bh_fdr(hit_rows, total_tests=max(tested_terms, 1))
    for row in hit_rows:
        row["tested_terms"] = tested_terms
        row["program_sentence"] = make_program_sentence(row)
    hit_rows.sort(key=lambda row: (row["fdr"], row["p_value"], -row["overlap"]))
    return hit_rows


def make_program_sentence(row: dict[str, Any]) -> str:
    direction_phrase = "higher-than-control" if row["direction"] == "up" else "lower-than-control"
    source_name = "Reactome" if row["source"] == "reactome" else "GO"
    term_name = str(row["term_name"]).strip()
    genes = ", ".join(row["overlap_genes"][:8])
    fdr = float(row.get("fdr", 1.0))
    strength = "are enriched for" if fdr <= 0.25 else "show a non-significant overlap with"
    return (
        f"At t={row.get('time', 'NA')} along the scDFM ODE trajectory for "
        f"{row['perturbation']}{cell_line_clause(row.get('cell_line', 'pooled'))}, "
        f"{direction_phrase} path-state genes {strength} "
        f"{source_name} term {term_name} (BH-FDR={fdr:.3g}), with overlap genes {genes}."
    )


def build_gene_evidence(rows: list[dict[str, Any]], max_genes: int) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row.get("direction") != "no_clear_change"
        and row.get("path_effect_size", row.get("effect_size")) in {"large", "moderate"}
        and row.get("stability") in {"stable", "partially_stable"}
        and row.get("trajectory_pattern", "flat") != "flat"
        and float(row.get("velocity_auc_rank_percentile", 0.0)) >= 0.6
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("path_abs_rank_percentile", row.get("abs_delta_rank_percentile", 0.0))),
            float(row.get("velocity_auc_rank_percentile", 0.0)),
            float(row.get("sign_stability", 0.0)),
        ),
        reverse=True,
    )
    evidence = []
    for row in candidates[:max_genes]:
        evidence.append({
            "perturbation": row["perturbation"],
            "cell_line": row.get("cell_line", "pooled"),
            "gene": row["gene"],
            "direction": row["direction"],
            "effect_size": row.get("path_effect_size", row.get("effect_size")),
            "trajectory_pattern": row.get("trajectory_pattern", ""),
            "onset": row.get("onset", ""),
            "velocity_pattern": row.get("velocity_pattern", ""),
            "stability": row["stability"],
            "uncertainty": row["uncertainty"],
            "path_late_delta": row.get("path_late_delta"),
            "path_abs_max": row.get("path_abs_max"),
            "delta_final": row.get("delta_final"),
            "sign_stability": row.get("sign_stability"),
            "delta_xt_by_time": row.get("delta_xt_by_time", {}),
            "velocity_by_time": row.get("velocity_by_time", {}),
            "time_state_labels": row.get("time_state_labels", {}),
            "path_state_sentence": row.get("path_state_sentence", ""),
            "velocity_sentence": row.get("velocity_sentence", ""),
            "endpoint_forecast_sentence": row.get("endpoint_forecast_sentence", ""),
            "simcontext": row.get("simcontext", ""),
        })
    return evidence


def build_prompt_rows(
    perturbation: str,
    cell_line: str,
    gene_evidence: list[dict[str, Any]],
    enrichment_rows: list[dict[str, Any]],
    max_program_terms: int,
) -> list[dict[str, Any]]:
    selected_terms = [
        row for row in enrichment_rows
        if row["perturbation"] == perturbation and row.get("cell_line", "pooled") == cell_line
    ]
    selected_terms.sort(key=lambda row: (time_key_value(str(row.get("time", 0.0))), row["fdr"], row["p_value"], -row["overlap"]))
    selected_terms = selected_terms[:max_program_terms]

    program_sentences = [row["program_sentence"] for row in selected_terms]
    prompt_rows = []
    for item in gene_evidence:
        supporting_genes = [
            (
                f"{other['gene']}: {other.get('trajectory_pattern', 'trajectory')}, "
                f"path effect {other['effect_size']}, onset {other.get('onset', 'NA')}, "
                f"velocity {other.get('velocity_pattern', 'NA')}, uncertainty {other['uncertainty']}."
            )
            for other in gene_evidence
            if other["gene"] != item["gene"]
        ][:10]

        prompt_lines = [
            f"Query tuple: (drug={perturbation}, gene={item['gene']}, cell_line={cell_line})",
            "Evidence source: scDFM inference-time ODE trace. Treat X_t path-state and velocity as primary evidence; endpoint forecast is secondary.",
            "ODE path-state evidence:",
            f"- {item.get('path_state_sentence', item.get('simcontext', ''))}",
            "Velocity evidence:",
            f"- {item.get('velocity_sentence', '')}",
        ]
        if program_sentences:
            prompt_lines.append("Time-resolved program evidence for the same drug and cell line:")
            prompt_lines.extend([f"- {sentence}" for sentence in program_sentences])
        prompt_lines.append("Endpoint forecast evidence:")
        prompt_lines.append(f"- {item.get('endpoint_forecast_sentence', '')}")
        if supporting_genes:
            prompt_lines.append("Other trajectory gene evidence in the same drug and cell line:")
            prompt_lines.extend([f"- {sentence}" for sentence in supporting_genes])
        prompt_lines.append(
            "Use these statements as model-derived trajectory evidence, not as independently validated biology."
        )

        prompt_rows.append({
            "query": {
                "drug": perturbation,
                "gene": item["gene"],
                "cell_line": cell_line,
            },
            "perturbation": perturbation,
            "gene": item["gene"],
            "cell_line": cell_line,
            "program_evidence": selected_terms,
            "gene_evidence": item,
            "supporting_gene_evidence": [other for other in gene_evidence if other["gene"] != item["gene"]][:10],
            "prompt_context": "\n".join(prompt_lines),
        })
    return prompt_rows


def default_prompt_out(out_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(out_path)), "prompt_ready_simcontext.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run time-resolved Reactome/GO ORA on scDFM trajectory SimContext rows and build prompt-ready context."
    )
    parser.add_argument("--simcontext-jsonl", required=True)
    parser.add_argument("--kg-dir", required=True)
    parser.add_argument("--out", required=True, help="Output program enrichment JSONL.")
    parser.add_argument("--prompt-out", default="", help="Output prompt-ready SimContext JSONL. Defaults next to --out.")
    parser.add_argument("--fallback-top-n", type=int, default=100)
    parser.add_argument("--min-selected", type=int, default=10)
    parser.add_argument("--max-terms-per-group", type=int, default=10)
    parser.add_argument("--max-program-terms-in-prompt", type=int, default=12)
    parser.add_argument("--max-gene-evidence-in-prompt", type=int, default=30)
    parser.add_argument("--fdr-threshold", type=float, default=0.25)
    args = parser.parse_args()

    sim_rows = load_jsonl(args.simcontext_jsonl)
    grouped = group_by_query_context(sim_rows)
    gene_sets = load_gene_sets(args.kg_dir)

    enrichment_rows = []
    prompt_rows = []
    for (perturbation, cell_line), rows in sorted(grouped.items()):
        background_genes = {row["gene"] for row in rows}
        perturbation_enrichment = []
        trace_times = available_trace_times(rows)

        for trace_time in trace_times:
            for direction in ("up", "down"):
                selected_genes, selection_rule = select_direction_genes_at_time(
                    rows,
                    time=trace_time,
                    direction=direction,
                    min_selected=args.min_selected,
                    fallback_top_n=args.fallback_top_n,
                )
                if len(selected_genes) < 3:
                    continue

                for source, source_config in gene_sets.items():
                    hits = enrich_one_source(
                        perturbation=perturbation,
                        cell_line=cell_line,
                        time=trace_time,
                        direction=direction,
                        selected_genes=selected_genes,
                        selection_rule=selection_rule,
                        background_genes=background_genes,
                        source=source,
                        source_config=source_config,
                    )
                    filtered_hits = [
                        row for row in hits
                        if row["fdr"] <= args.fdr_threshold
                    ]
                    if not filtered_hits:
                        filtered_hits = hits[:args.max_terms_per_group]
                    else:
                        filtered_hits = filtered_hits[:args.max_terms_per_group]

                    perturbation_enrichment.extend(filtered_hits)

        enrichment_rows.extend(perturbation_enrichment)
        gene_evidence = build_gene_evidence(rows, args.max_gene_evidence_in_prompt)
        prompt_rows.extend(
            build_prompt_rows(
                perturbation=perturbation,
                cell_line=cell_line,
                gene_evidence=gene_evidence,
                enrichment_rows=perturbation_enrichment,
                max_program_terms=args.max_program_terms_in_prompt,
            )
        )

    prompt_out = args.prompt_out or default_prompt_out(args.out)
    write_jsonl(enrichment_rows, args.out)
    write_jsonl(prompt_rows, prompt_out)

    print(f"drug-cell-line contexts: {len(grouped)}")
    print(f"program enrichment rows: {len(enrichment_rows)}")
    print(f"prompt-ready rows: {len(prompt_rows)}")
    print(f"program enrichment: {args.out}")
    print(f"prompt ready: {prompt_out}")


if __name__ == "__main__":
    main()
