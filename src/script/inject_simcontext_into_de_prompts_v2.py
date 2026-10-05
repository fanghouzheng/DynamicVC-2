import argparse
import json
import re
from pathlib import Path
from typing import Any


SEPARATOR = "=" * 80


def normalize_drug(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_gene(value: str) -> str:
    return value.strip().upper()


def normalize_cell_line(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+cell line$", "", value)
    value = re.sub(r"\s+cells$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def load_simcontext(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    index = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (
                normalize_drug(str(row.get("perturbation", ""))),
                normalize_gene(str(row.get("gene", ""))),
                normalize_cell_line(str(row.get("cell_line", ""))),
            )
            old = index.get(key)
            if old is None or float(row.get("path_abs_rank_percentile", 0.0)) > float(old.get("path_abs_rank_percentile", 0.0)):
                index[key] = row
    return index


def load_program_enrichment(path: Path | None, max_terms: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if path is None:
        return {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (
                normalize_drug(str(row.get("perturbation", ""))),
                normalize_cell_line(str(row.get("cell_line", ""))),
            )
            grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                float(row.get("fdr", 1.0)),
                float(row.get("p_value", 1.0)),
                str(row.get("time", "")),
                -int(row.get("overlap", 0)),
            )
        )
        grouped[key] = rows[:max_terms]
    return grouped


def parse_query(block: str) -> tuple[str, str, str] | None:
    header_match = re.search(r"^=== Prompt\s+\d+\s+\((.*?)\s+\|\s+(.*?)\)\s+===", block, flags=re.MULTILINE)
    if not header_match:
        return None
    drug = header_match.group(1).strip()
    gene = header_match.group(2).strip()
    goal_match = re.search(
        r"Goal:\s*Determine if a perturbation of .*? in the (.*?) cell line results",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cell_line = goal_match.group(1).strip() if goal_match else "C32"
    return drug, gene, cell_line


def format_program_context(program_rows: list[dict[str, Any]], drug: str, cell_line: str) -> list[str]:
    if not program_rows:
        return []
    lines = [f"  Drug-cell-level trajectory program evidence for (drug={drug}, cell_line={cell_line}):"]
    for row in program_rows:
        sentence = str(row.get("program_sentence", "")).strip()
        if not sentence:
            source = row.get("source", "pathway")
            term = row.get("term_name", row.get("term_id", "NA"))
            direction = row.get("direction", "NA")
            time = row.get("time", "NA")
            fdr = float(row.get("fdr", 1.0))
            sentence = f"At t={time}, {direction} path-state genes overlap with {source} term {term} (BH-FDR={fdr:.3g})."
        lines.append(f"  - {sentence}")
    return lines


def format_simcontext(
    row: dict[str, Any] | None,
    drug: str,
    gene: str,
    cell_line: str,
    program_rows: list[dict[str, Any]] | None = None,
) -> str:
    header = "- Model-derived scDFM trajectory evidence:"
    program_rows = program_rows or []
    if row is None:
        parts = [
            header,
            f"  No gene-level scDFM trajectory evidence is available for (drug={drug}, gene={gene}, cell_line={cell_line}).",
        ]
        parts.extend(format_program_context(program_rows, drug=drug, cell_line=cell_line))
        if program_rows:
            parts.append(
                "  Evidence caveat: Program evidence is drug-cell-level model-derived trajectory context, "
                "not gene-specific experimental validation."
            )
        else:
            parts.append("  Evidence caveat: No model-derived scDFM trajectory evidence is available for this tuple.")
        return "\n".join(parts)

    parts = [
        header,
        f"  Query tuple: (drug={row.get('perturbation', drug)}, gene={row.get('gene', gene)}, cell_line={row.get('cell_line', cell_line)})",
    ]
    path_sentence = str(row.get("path_state_sentence", "")).strip()
    velocity_sentence = str(row.get("velocity_sentence", "")).strip()
    endpoint_sentence = str(row.get("endpoint_forecast_sentence", "")).strip()
    if path_sentence:
        parts.append(f"  ODE path-state evidence: {path_sentence}")
    if velocity_sentence:
        parts.append(f"  Velocity evidence: {velocity_sentence}")
    if endpoint_sentence:
        parts.append(f"  Endpoint forecast evidence: {endpoint_sentence}")
    labels = [
        f"trajectory_direction={row.get('trajectory_direction', row.get('direction', 'NA'))}",
        f"path_effect={row.get('path_effect_size', row.get('effect_size', 'NA'))}",
        f"trajectory_pattern={row.get('trajectory_pattern', 'NA')}",
        f"onset={row.get('onset', 'NA')}",
        f"stability={row.get('stability', 'NA')}",
        f"uncertainty={row.get('uncertainty', 'NA')}",
    ]
    parts.append(f"  Trajectory labels: {'; '.join(labels)}.")
    top_up = row.get("top_up_genes", [])[:5]
    top_down = row.get("top_down_genes", [])[:5]
    if top_up or top_down:
        up_text = ", ".join(top_up) if top_up else "none"
        down_text = ", ".join(top_down) if top_down else "none"
        parts.append(
            f"  Same drug-cell ODE module context: late higher-than-control genes include {up_text}; "
            f"late lower-than-control genes include {down_text}."
        )
    parts.extend(format_program_context(program_rows, drug=drug, cell_line=cell_line))
    parts.append("  Evidence caveat: This is model-derived scDFM trajectory evidence, not independently validated experimental biology.")
    return "\n".join(parts)


def enhance_prompt_instructions(block: str) -> str:
    if "Model-derived scDFM trajectory evidence: soft model-generated" not in block:
        block = re.sub(
            r"(- Evidence Set: .+?\n)(\nReasoning Guidelines:)",
            (
                r"\1"
                "- Model-derived scDFM trajectory evidence: soft model-generated trajectory evidence summarizing "
                "gene-level ODE path-state changes when available, plus drug-cell-level program evidence as fallback.\n"
                r"\2"
            ),
            block,
            count=1,
            flags=re.DOTALL,
        )
    caveat = (
        "When available, use the scDFM trajectory evidence as soft model-derived evidence, not as experimentally "
        "validated ground truth. If it conflicts with the drug mechanism, cell-line context, gene biology, or analogue "
        "cases, explicitly discuss the conflict."
    )
    if caveat not in block:
        block = re.sub(
            r"(Avoid superficial pattern matching\..*?\n)",
            r"\1" + caveat + "\n",
            block,
            count=1,
            flags=re.DOTALL,
        )
    return block


def inject_block(block: str, evidence_text: str) -> str:
    block = enhance_prompt_instructions(block)
    marker = "[End of Input]"
    if marker not in block:
        return block.rstrip() + "\n" + evidence_text + "\n"
    return block.replace(marker, evidence_text + "\n" + marker, 1)


def iter_prompt_blocks(text: str) -> list[str]:
    return [block.strip() for block in text.split(SEPARATOR) if block.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject scDFM trajectory evidence into existing VCWorld DE prompt text files.")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--simcontext-jsonl", required=True, type=Path)
    parser.add_argument("--program-enrichment-jsonl", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max-program-terms", type=int, default=5)
    parser.add_argument("--only-matched", action="store_true", help="Write only prompts with direct gene-level SimContext matches.")
    args = parser.parse_args()

    simcontext = load_simcontext(args.simcontext_jsonl)
    program_context = load_program_enrichment(args.program_enrichment_jsonl, args.max_program_terms)
    blocks = iter_prompt_blocks(args.prompts.read_text(encoding="utf-8"))

    written = 0
    matched = 0
    missing = 0
    program_fallback = 0
    no_evidence = 0
    output_blocks = []

    for block in blocks:
        query = parse_query(block)
        if query is None:
            continue
        drug, gene, cell_line = query
        key = (normalize_drug(drug), normalize_gene(gene), normalize_cell_line(cell_line))
        program_key = (normalize_drug(drug), normalize_cell_line(cell_line))
        row = simcontext.get(key)
        program_rows = program_context.get(program_key, [])
        if row is None:
            missing += 1
            if program_rows:
                program_fallback += 1
            else:
                no_evidence += 1
            if args.only_matched:
                continue
        else:
            matched += 1

        evidence_text = format_simcontext(row, drug=drug, gene=gene, cell_line=cell_line, program_rows=program_rows)
        output_blocks.append(inject_block(block, evidence_text))
        written += 1
        if args.max_prompts > 0 and written >= args.max_prompts:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text((f"\n\n{SEPARATOR}\n\n").join(output_blocks) + "\n", encoding="utf-8")
    print(f"input prompts: {len(blocks)}")
    print(f"simcontext rows indexed: {len(simcontext)}")
    print(f"program contexts indexed: {len(program_context)}")
    print(f"direct gene-level matches seen: {matched}")
    print(f"missing gene-level seen: {missing}")
    print(f"program fallback seen: {program_fallback}")
    print(f"no evidence seen: {no_evidence}")
    print(f"written prompts: {written}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
