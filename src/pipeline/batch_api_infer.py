import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from glob import glob as expand_glob
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


LIBRARY_PATH_REEXEC_ENV = "BIOCOLLECTION_LIBRARY_PATH_REEXEC"


def configure_process_library_path() -> None:
    """Start with the active environment's C++ runtime ahead of system libs.

    Updating LD_LIBRARY_PATH inside an already-running process is too late for
    glibc's dynamic loader. If the conda lib directory was not already first,
    restart this script once before importing torch/vLLM or sqlite extensions.
    Spawned vLLM workers inherit the corrected environment and must not restart:
    multiprocessing executes this file as ``__mp_main__``, where an exec would
    turn the worker into a second copy of the top-level inference program.
    """

    env_lib = Path(sys.prefix) / "lib"
    if not (env_lib / "libstdc++.so.6").exists():
        return

    env_lib_str = str(env_lib)
    current_paths = [
        path for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path
    ]
    current_paths = [path for path in current_paths if path != env_lib_str]
    corrected_path = os.pathsep.join([env_lib_str, *current_paths])
    already_preferred = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[
        :1
    ] == [env_lib_str]
    os.environ["LD_LIBRARY_PATH"] = corrected_path

    if already_preferred:
        return
    if os.environ.get(LIBRARY_PATH_REEXEC_ENV) == "1":
        raise RuntimeError(
            "Failed to activate the conda C++ runtime after restarting. "
            f"Expected LD_LIBRARY_PATH to begin with {env_lib_str}."
        )

    restart_env = os.environ.copy()
    restart_env[LIBRARY_PATH_REEXEC_ENV] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], restart_env)


# This file talks to a remote OpenAI-compatible endpoint and does not load a
# local model.  The legacy vLLM helpers are retained below only so the prompt
# and result helpers remain byte-for-byte aligned with the original script.


PROMPT_HEADER_RE = re.compile(
    r"^===\s*Prompt\s*(\d+)\s*\((.*?)\)\s*===\s*$",
    re.MULTILINE,
)

CHOICE_CANDIDATES = {
    "yes": "Yes.",
    "no": "No.",
    "insufficient": "There is insufficient evidence to determine.",
}

GRAVITY_ARCHITECTURE = "GravityMoEForCausalLM"
GRAVITY_MODEL_TYPE = "gravity_moe"
GRAVITY_NATIVE_ARCHITECTURE = "DeepseekV3ForCausalLM"
GRAVITY_NATIVE_MODEL_TYPE = "deepseek_v3"
GRAVITY_VLLM_COMPAT_ENV = "BIOCOLLECTION_GRAVITY_VLLM_COMPAT"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def expand_gravity_fused_expert_weights(
    weights: Iterable[tuple[str, Any]],
    *,
    num_experts: int,
) -> Iterable[tuple[str, Any]]:
    """Present fused Gravity expert tensors as per-expert DeepSeek weights."""

    gate_up_suffix = ".mlp.experts.gate_up_proj"
    down_suffix = ".mlp.experts.down_proj"

    for name, loaded_weight in weights:
        normalized_name = name.removesuffix(".weight")
        if normalized_name.endswith(gate_up_suffix):
            if loaded_weight.ndim != 3:
                raise ValueError(
                    f"Expected a 3D fused expert tensor for {name}, got "
                    f"shape={tuple(loaded_weight.shape)}"
                )
            if loaded_weight.shape[0] != num_experts:
                raise ValueError(
                    f"Expected {num_experts} experts in {name}, got "
                    f"shape={tuple(loaded_weight.shape)}"
                )
            if loaded_weight.shape[1] % 2 != 0:
                raise ValueError(
                    f"The gate/up dimension must be even for {name}, got "
                    f"shape={tuple(loaded_weight.shape)}"
                )

            experts_prefix = normalized_name.removesuffix(".gate_up_proj")
            gate_weights, up_weights = loaded_weight.chunk(2, dim=1)
            for expert_id, (gate_weight, up_weight) in enumerate(
                zip(gate_weights.unbind(0), up_weights.unbind(0), strict=True)
            ):
                yield (
                    f"{experts_prefix}.{expert_id}.gate_proj.weight",
                    gate_weight,
                )
                yield (
                    f"{experts_prefix}.{expert_id}.up_proj.weight",
                    up_weight,
                )
            continue

        if normalized_name.endswith(down_suffix):
            if loaded_weight.ndim != 3:
                raise ValueError(
                    f"Expected a 3D fused expert tensor for {name}, got "
                    f"shape={tuple(loaded_weight.shape)}"
                )
            if loaded_weight.shape[0] != num_experts:
                raise ValueError(
                    f"Expected {num_experts} experts in {name}, got "
                    f"shape={tuple(loaded_weight.shape)}"
                )

            experts_prefix = normalized_name.removesuffix(".down_proj")
            for expert_id, down_weight in enumerate(loaded_weight.unbind(0)):
                yield (
                    f"{experts_prefix}.{expert_id}.down_proj.weight",
                    down_weight,
                )
            continue

        yield name, loaded_weight


def install_gravity_vllm_weight_loader() -> None:
    """Patch only vLLM's DeepSeek V3 class to load Gravity fused experts."""

    from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM

    if getattr(DeepseekV3ForCausalLM, "_biocollection_gravity_compat", False):
        return

    original_load_weights = DeepseekV3ForCausalLM.load_weights

    def load_weights(self: Any, weights: Iterable[tuple[str, Any]]) -> set[str]:
        expanded_weights = expand_gravity_fused_expert_weights(
            weights,
            num_experts=self.config.n_routed_experts,
        )
        return original_load_weights(self, expanded_weights)

    DeepseekV3ForCausalLM.load_weights = load_weights
    DeepseekV3ForCausalLM._biocollection_gravity_compat = True


def split_prompts_from_text(text: str) -> list[dict[str, Any]]:
    matches = list(PROMPT_HEADER_RE.finditer(text))
    if not matches:
        sep = "================================================================================"
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        return [
            {
                "prompt_index": i + 1,
                "prompt_title": f"Prompt {i + 1}",
                "prompt_text": p,
            }
            for i, p in enumerate(parts)
        ]

    prompts: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        prompts.append(
            {
                "prompt_index": int(m.group(1)),
                "prompt_title": m.group(2).strip(),
                "prompt_text": text[start:end].strip(),
            }
        )
    return prompts


def extract_judge_text(response_text: str) -> str:
    text = (response_text or "").strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    for line in reversed(lines):
        clean = line.strip().strip("*_`").strip()
        clean = re.sub(r"^(?:[-*•>]\s*)+", "", clean).strip()
        clean = re.sub(
            r"^(?:answer|final answer|final prediction|prediction)\s*[:：-]\s*",
            "",
            clean,
            flags=re.I,
        )
        clean = clean.strip().strip("*_`").strip()

        option_match = re.match(r"^([ABC])\)\s*(.+\S.*)$", clean, re.I)
        if option_match:
            letter, body = option_match.group(1).upper(), option_match.group(2).strip()
            return f"{letter}) {body}".strip().strip("*_`").strip()

        embedded_option_match = re.search(r"\b([ABC])\)\s*(Yes.*|No.*|There.+)$", clean, re.I)
        if embedded_option_match:
            letter = embedded_option_match.group(1).upper()
            body = embedded_option_match.group(2).strip()
            return f"{letter}) {body}".strip().strip("*_`").strip()

        yes_no_match = re.match(
            r"^(Yes|No|There is insufficient(?: evidence)?(?: to determine)?(?:[^\n.]*)?)\s*(?:[.!?].*)?$",
            clean,
            re.I,
        )
        if yes_no_match:
            return yes_no_match.group(1).strip()

    return ""


def normalize_label(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    lower = s.lower()
    if re.match(r"^\s*(?:[-*•]\s*)?a\)", s, re.I) or lower.startswith("no"):
        return "no"
    if re.match(r"^\s*(?:[-*•]\s*)?b\)", s, re.I) or lower.startswith("yes"):
        return "yes"
    if (
        re.match(r"^\s*(?:[-*•]\s*)?c\)", s, re.I)
        or "insufficient" in lower
        or "cannot determine" in lower
        or "unable to determine" in lower
    ):
        return "insufficient"
    return lower


def softmax_from_scores(scores: dict[str, float | None]) -> dict[str, float | None]:
    finite = {k: v for k, v in scores.items() if v is not None and math.isfinite(v)}
    if not finite:
        return {k: None for k in scores}
    max_score = max(finite.values())
    exps = {k: math.exp(v - max_score) for k, v in finite.items()}
    denom = sum(exps.values())
    return {k: (exps[k] / denom if k in exps else None) for k in scores}


def get_logprob_value(logprob_entry: Any, token_id: int) -> float | None:
    if logprob_entry is None:
        return None
    item = None
    if isinstance(logprob_entry, dict):
        item = logprob_entry.get(token_id)
        if item is None:
            item = logprob_entry.get(str(token_id))
    else:
        item = logprob_entry
    if item is None:
        return None
    value = getattr(item, "logprob", item)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def common_prefix_len(left: list[int], right: list[int]) -> int:
    i = 0
    while i < len(left) and i < len(right) and left[i] == right[i]:
        i += 1
    return i


def score_choice_outputs(
    tokenizer: Any,
    base_prompts: list[str],
    outputs: list[Any],
    *,
    score_mode: str,
) -> list[dict[str, Any]]:
    labels = list(CHOICE_CANDIDATES)
    scored: list[dict[str, Any]] = []
    out_i = 0

    for base_prompt in base_prompts:
        per_label: dict[str, dict[str, Any]] = {}
        base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
        base_len = len(base_ids)

        for label in labels:
            candidate = CHOICE_CANDIDATES[label]
            full_text = base_prompt + candidate
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            candidate_start = common_prefix_len(base_ids, full_ids)
            candidate_ids = full_ids[candidate_start:]
            output = outputs[out_i]
            out_i += 1

            prompt_logprobs = getattr(output, "prompt_logprobs", None) or []
            token_rows = []
            token_logprobs: list[float] = []
            for pos, token_id in enumerate(candidate_ids, start=candidate_start):
                token = tokenizer.decode([token_id])
                entry = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
                token_logprob = get_logprob_value(entry, token_id)
                token_rows.append(
                    {
                        "token_id": token_id,
                        "token": token,
                        "logprob": token_logprob,
                    }
                )
                if token_logprob is not None:
                    token_logprobs.append(token_logprob)

            complete = len(token_logprobs) == len(candidate_ids) and bool(candidate_ids)
            sequence_logprob = sum(token_logprobs) if complete else None
            avg_logprob = (
                sequence_logprob / len(token_logprobs)
                if sequence_logprob is not None and token_logprobs
                else None
            )
            first_token_logprob = token_logprobs[0] if token_logprobs else None
            if score_mode == "avg":
                score_logprob = avg_logprob
            elif score_mode == "sum":
                score_logprob = sequence_logprob
            else:
                score_logprob = first_token_logprob
            per_label[label] = {
                "candidate_text": candidate,
                "token_count": len(candidate_ids),
                "sequence_logprob": sequence_logprob,
                "avg_logprob": avg_logprob,
                "first_token_logprob": first_token_logprob,
                "score_logprob": score_logprob,
                "prefix_token_mismatch": candidate_start != base_len,
                "tokens": token_rows,
            }

        probs = softmax_from_scores({k: v["score_logprob"] for k, v in per_label.items()})
        for label, prob in probs.items():
            per_label[label]["prob"] = prob
        finite_labels = [
            label
            for label, score in per_label.items()
            if score["score_logprob"] is not None and math.isfinite(score["score_logprob"])
        ]
        pred_label = (
            max(finite_labels, key=lambda k: per_label[k]["score_logprob"])
            if finite_labels
            else ""
        )
        scored.append({"pred_label": pred_label, "class_scores": per_label})

    return scored


def parse_prompt_title(prompt_title: str) -> tuple[str, str]:
    parts = [p.strip() for p in prompt_title.split("|", 1)]
    if len(parts) == 2:
        return parts[0], parts[1]
    return prompt_title.strip(), ""


def guess_cell_line(input_path: Path) -> str:
    return input_path.stem.split("_", 1)[0]


def slugify_model_name(name: str) -> str:
    s = name.strip().replace("\\", "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "model"


def read_model_config(model_path: Path) -> dict[str, Any] | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def read_num_attention_heads(model_path: Path) -> int | None:
    payload = read_model_config(model_path)
    if payload is None:
        return None
    heads = payload.get("num_attention_heads")
    return heads if isinstance(heads, int) else None


def is_gravity_model(config: dict[str, Any] | None) -> bool:
    if not config:
        return False
    architectures = config.get("architectures") or []
    return config.get("model_type") == GRAVITY_MODEL_TYPE or (
        isinstance(architectures, list) and GRAVITY_ARCHITECTURE in architectures
    )


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def huggingface_download_hint(model_path: Path) -> str | None:
    try:
        snapshot_dir = model_path.resolve().parent
    except OSError:
        snapshot_dir = model_path.parent
    if snapshot_dir.name != "snapshots":
        return None
    repo_cache = snapshot_dir.parent
    if not repo_cache.name.startswith("models--"):
        return None
    repo_id = repo_cache.name.removeprefix("models--").replace("--", "/")
    cache_dir = repo_cache.parent
    revision = model_path.name
    return (
        f"hf download {repo_id} --revision {revision} "
        f"--cache-dir {cache_dir}"
    )


def validate_local_checkpoint_files(model_path: Path) -> None:
    if not model_path.is_dir():
        return

    index_paths = [
        model_path / "model.safetensors.index.json",
        model_path / "pytorch_model.bin.index.json",
    ]
    index_path = next((path for path in index_paths if path.exists()), None)
    if index_path is None:
        return

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") or {}
        expected_files = sorted(set(weight_map.values()))
    except Exception as exc:
        raise ValueError(f"Invalid checkpoint index {index_path}: {exc}") from exc

    missing_files = [name for name in expected_files if not (model_path / name).is_file()]
    if not missing_files:
        return

    total_size = (payload.get("metadata") or {}).get("total_size")
    lines = [
        f"Incomplete model checkpoint: {model_path}",
        f"The index expects {len(expected_files)} weight shard(s) "
        f"({human_size(total_size)} total), but {len(missing_files)} are missing:",
        *(f"  - {name}" for name in missing_files),
    ]
    if hint := huggingface_download_hint(model_path):
        lines.extend(["Resume the Hugging Face download with:", f"  {hint}"])
    raise FileNotFoundError("\n".join(lines))


def build_gravity_native_model(model_path: Path, config: dict[str, Any]) -> Path:
    """Expose Gravity through vLLM's native DeepSeek V3 implementation.

    GravityMoE is a thin Transformers subclass of DeepSeek V3 with different
    hyperparameters. Rewriting only the config selects vLLM's native DeepSeek
    implementation; a narrow in-process loader patch exposes Gravity's fused
    3D expert tensors as the per-expert views expected by that implementation.
    This avoids vLLM's Transformers MoE backend, which requires transformers>=5
    and cannot provide tensor parallel plans with transformers 4.x.
    """

    patched_config = dict(config)
    patched_config["architectures"] = [GRAVITY_NATIVE_ARCHITECTURE]
    patched_config["model_type"] = GRAVITY_NATIVE_MODEL_TYPE
    patched_config.pop("auto_map", None)
    patched_config_text = json.dumps(
        patched_config,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    tokenizer_config_path = model_path / TOKENIZER_CONFIG_FILE
    if not tokenizer_config_path.is_file():
        raise FileNotFoundError(
            f"Gravity compatibility requires {TOKENIZER_CONFIG_FILE} in {model_path}."
        )
    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid tokenizer config {tokenizer_config_path}: {exc}") from exc
    tokenizer_config["fix_mistral_regex"] = True
    patched_tokenizer_config = json.dumps(
        tokenizer_config,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    source_key = (
        str(model_path.resolve()).encode("utf-8")
        + patched_config_text.encode("utf-8")
        + patched_tokenizer_config.encode("utf-8")
    )
    digest = hashlib.sha256(source_key).hexdigest()[:16]
    compat_dir = (
        Path(tempfile.gettempdir())
        / "biocollection-vllm-model-compat"
        / f"gravity-{digest}"
    )
    compat_dir.mkdir(parents=True, exist_ok=True)

    for source in model_path.iterdir():
        target = compat_dir / source.name
        if source.name == "config.json":
            if (
                not target.exists()
                or target.read_text(encoding="utf-8") != patched_config_text
            ):
                target.write_text(patched_config_text, encoding="utf-8")
            continue
        if source.name == TOKENIZER_CONFIG_FILE:
            if (
                not target.exists()
                or target.read_text(encoding="utf-8") != patched_tokenizer_config
            ):
                target.write_text(patched_tokenizer_config, encoding="utf-8")
            continue

        source_target = source.resolve()
        if target.is_symlink() and target.resolve() == source_target:
            continue
        if target.exists() or target.is_symlink():
            raise RuntimeError(
                f"Refusing to replace unexpected compatibility file: {target}"
            )
        target.symlink_to(source_target)

    return compat_dir


def prepare_model_for_vllm(args: argparse.Namespace, model_path: Path) -> None:
    args.runtime_model = args.model
    config = read_model_config(model_path)
    if not is_gravity_model(config):
        return

    if args.model_impl not in (None, "auto", "vllm"):
        raise ValueError(
            f"{GRAVITY_ARCHITECTURE} should use vLLM's native DeepSeek V3 "
            "implementation. Omit --model-impl or use --model-impl vllm."
        )
    args.model_impl = "vllm"
    os.environ[GRAVITY_VLLM_COMPAT_ENV] = "1"
    install_gravity_vllm_weight_loader()
    compat_dir = build_gravity_native_model(model_path, config)
    args.runtime_model = str(compat_dir)
    print(
        "Gravity compatibility: using vLLM's native DeepSeek V3 backend with "
        f"a patched config at {compat_dir}"
    )


def valid_tensor_parallel_sizes(num_attention_heads: int) -> list[int]:
    return [i for i in range(1, num_attention_heads + 1) if num_attention_heads % i == 0]


def load_done_results(out_json_path: Path) -> dict[int, dict[str, Any]]:
    if not out_json_path.exists():
        return {}
    try:
        payload = json.loads(out_json_path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        done: dict[int, dict[str, Any]] = {}
        for r in results:
            idx = r.get("prompt_index")
            if isinstance(idx, int):
                done[idx] = r
        return done
    except Exception:
        return {}


def result_has_required_fields(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if not row.get("pred_label") and not row.get("judge_text"):
        return False
    if args.score_choices:
        if not isinstance(row.get("class_scores"), dict):
            return False
        # Older API results may contain an empty class_scores object created
        # by the logprob path. Force those rows to be re-scored with JSON
        # probabilities, while preserving completed insufficient answers.
        if "answered" not in row:
            return False
        if row.get("choice_score_error"):
            return False
    return True


def write_json(
    out_path: Path,
    *,
    model_name: str,
    input_file: Path,
    prompt_count: int,
    ordered_results: list[dict[str, Any]],
    save_prompts: bool,
    prompts: list[dict[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "model_name": model_name,
        "input_file": str(input_file),
        "prompt_count": prompt_count,
        "results": ordered_results,
    }
    if save_prompts:
        payload["prompts"] = prompts
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_txt(out_path: Path, input_file: Path, ordered_results: list[dict[str, Any]]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("=== Inference Results ===\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Prompts: {len(ordered_results)}\n\n")
        for r in ordered_results:
            f.write(f"=== 回答 for Prompt {r['prompt_index']} ({r['prompt_title']}) ===\n\n")
            if r.get("error"):
                f.write(f"ERROR: {r['error']}\n\n")
            if r.get("judge_text"):
                f.write(f"judge_text: {r['judge_text']}\n\n")
            if r.get("pred_label"):
                f.write(f"pred_label: {r['pred_label']}\n\n")
            if "de_probability" in r:
                f.write(
                    f"answered: {r.get('answered', False)}\n"
                    f"de_probability: {r.get('de_probability')}\n\n"
                )
            if r.get("class_scores"):
                f.write("class_scores:\n")
                for label, score in r["class_scores"].items():
                    f.write(
                        f"  {label}: score_logprob={score.get('score_logprob')}, "
                        f"prob={score.get('prob')}\n"
                    )
                f.write("\n")
            f.write(r.get("response_text", "") or "")
            f.write("\n\n")
            f.write("================================================================================\n\n")


def write_prediction_csv(
    out_path: Path,
    ordered_results: list[dict[str, Any]],
    *,
    input_file: Path,
    cell_line: str | None,
) -> None:
    fieldnames = [
        "pert",
        "gene",
        "cell line",
        "label",
        "pred_label",
        "choice_pred_label",
        "generated_label",
        "prompt_index",
        "prompt_title",
        "judge_text",
        "score_yes",
        "score_no",
        "score_insufficient",
        "logit_yes",
        "logit_no",
        "logit_insufficient",
        "prob_yes",
        "prob_no",
        "prob_insufficient",
        "answered",
        "de_probability",
    ]
    inferred_cell_line = cell_line or guess_cell_line(input_file)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ordered_results:
            pert, gene = parse_prompt_title(r.get("prompt_title", ""))
            class_scores = r.get("class_scores") or {}
            pred_label = r.get("pred_label") or normalize_label(r.get("judge_text"))
            row = {
                "pert": pert,
                "gene": gene,
                "cell line": inferred_cell_line,
                "label": pred_label,
                "pred_label": pred_label,
                "choice_pred_label": r.get("choice_pred_label"),
                "generated_label": normalize_label(r.get("judge_text")),
                "prompt_index": r.get("prompt_index"),
                "prompt_title": r.get("prompt_title"),
                "judge_text": r.get("judge_text"),
                "answered": r.get("answered"),
                "de_probability": r.get("de_probability"),
            }
            for label in CHOICE_CANDIDATES:
                score = class_scores.get(label) or {}
                row[f"score_{label}"] = score.get("score_logprob")
                row[f"logit_{label}"] = score.get("score_logprob")
                row[f"prob_{label}"] = score.get("prob")
            writer.writerow(row)


def canonical_metric_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def infer_truth_csv(input_path: Path) -> Path | None:
    candidate = input_path.parent / f"{guess_cell_line(input_path)}_DE.csv"
    return candidate if candidate.is_file() else None


def normalize_truth_label(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if text in {"1", "1.0", "yes", "true", "de", "positive"}:
        return 1
    if text in {"0", "0.0", "no", "false", "non-de", "negative"}:
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number in (0.0, 1.0):
        return int(number)
    return None


def load_truth_labels(truth_path: Path) -> dict[tuple[str, str], int]:
    with truth_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"pert", "gene", "label"}
        if not required.issubset(fieldnames):
            raise ValueError(
                f"Truth CSV {truth_path} must contain columns: {sorted(required)}"
            )
        labels: dict[tuple[str, str], int] = {}
        for row in reader:
            label = normalize_truth_label(row.get("label"))
            if label is None:
                continue
            key = (
                canonical_metric_key(row.get("pert")),
                canonical_metric_key(row.get("gene")),
            )
            if not key[0] or not key[1]:
                continue
            previous = labels.get(key)
            if previous is not None and previous != label:
                raise ValueError(f"Conflicting truth labels for key={key} in {truth_path}")
            labels[key] = label
    return labels


def pairwise_auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positive_count = sum(labels)
    if positive_count == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    true_positives = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positive_count


def compute_binary_metrics(
    ordered_results: list[dict[str, Any]],
    truth_labels: dict[tuple[str, str], int],
) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    matched = 0
    answered = 0
    unanswered = 0
    for result in ordered_results:
        pert, gene = parse_prompt_title(result.get("prompt_title", ""))
        truth = truth_labels.get(
            (canonical_metric_key(pert), canonical_metric_key(gene))
        )
        if truth is None:
            continue
        matched += 1
        probability = normalize_probability(result.get("de_probability"))
        if not result.get("answered") or probability is None:
            unanswered += 1
            continue
        answered += 1
        labels.append(truth)
        scores.append(probability)
    total = len(ordered_results)
    return {
        "n_total": total,
        "n_truth_matched": matched,
        "n_answered": answered,
        "n_unanswered": unanswered,
        "answered_rate": answered / matched if matched else None,
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "auroc": pairwise_auroc(labels, scores),
        "auprc": average_precision(labels, scores),
        "score_definition": "de_probability",
        "unanswered_excluded": True,
    }


def write_metrics_json(
    out_path: Path,
    *,
    input_file: Path,
    truth_file: Path,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "input_file": str(input_file),
        "truth_file": str(truth_file),
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_chat_prompt(tokenizer: Any, prompt_text: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt_text

    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt_text


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


CHOICE_LETTERS = {
    "no": "A",
    "yes": "B",
    "insufficient": "C",
}
LETTER_TO_LABEL = {letter: label for label, letter in CHOICE_LETTERS.items()}


def get_chat_endpoint(base_url: str) -> str:
    """Normalize a base URL or a complete chat-completions URL."""

    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def extract_api_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def iter_top_logprob_items(top_logprobs: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(top_logprobs, dict):
        for token, value in top_logprobs.items():
            if isinstance(value, dict):
                yield str(token), value.get("logprob")
            else:
                yield str(token), value
    elif isinstance(top_logprobs, list):
        for item in top_logprobs:
            if isinstance(item, dict) and item.get("token") is not None:
                yield str(item["token"]), item.get("logprob")


def normalize_choice_token(token: Any) -> str:
    return str(token or "").strip().upper().strip("`*_.,:;)]}")


def normalize_answer_label(value: Any) -> str:
    """Normalize JSON answer values to yes/no/insufficient."""

    text = str(value or "").strip()
    if not text:
        return ""
    letter = normalize_choice_token(text[:8])
    if letter in LETTER_TO_LABEL:
        return LETTER_TO_LABEL[letter]
    return normalize_label(text)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object, tolerating markdown fences or extra text."""

    raw = (text or "").strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_probability(value: Any) -> float | None:
    """Normalize a decimal probability or percentage to the closed unit interval."""

    if isinstance(value, bool):
        return None
    explicit_percent = isinstance(value, str) and value.strip().endswith("%")
    if explicit_percent:
        value = value.strip()[:-1].strip()
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability) or probability < 0:
        return None
    if explicit_percent and probability <= 100.0:
        probability /= 100.0
    elif probability > 1.0 and probability <= 100.0:
        probability /= 100.0
    if probability < 0.0 or probability > 1.0:
        return None
    return probability


def parse_probability_response(text: str) -> dict[str, Any]:
    """Parse the model's structured answer and DE probability.

    ``answered`` is deliberately false for an insufficient answer or when no
    valid numeric probability is present. Such rows are excluded from metrics.
    """

    payload = extract_json_object(text)
    if payload is None:
        return {
            "answer": "",
            "de_probability": None,
            "answered": False,
            "parse_error": "no_json_object",
        }

    answer_value = (
        payload.get("answer")
        or payload.get("label")
        or payload.get("prediction")
        or payload.get("decision")
        or payload.get("result")
    )
    answer = normalize_answer_label(answer_value)
    probability_value = None
    for key in (
        "de_probability",
        "de_prob",
        "yes_probability",
        "probability",
        "prob_yes",
        "p_de",
    ):
        if key in payload:
            probability_value = payload[key]
            break
    probability = normalize_probability(probability_value)
    if answer == "insufficient":
        return {
            "answer": answer,
            "de_probability": None,
            "answered": False,
        }
    if answer not in {"yes", "no"}:
        return {
            "answer": answer,
            "de_probability": probability,
            "answered": False,
            "parse_error": "missing_binary_answer",
        }
    if probability is None:
        return {
            "answer": answer,
            "de_probability": None,
            "answered": False,
            "parse_error": "invalid_probability",
        }
    return {
        "answer": answer,
        "de_probability": probability,
        "answered": True,
    }


def empty_class_scores() -> dict[str, dict[str, Any]]:
    return {
        label: {
            "candidate_text": candidate,
            "token_count": 1,
            "sequence_logprob": None,
            "avg_logprob": None,
            "first_token_logprob": None,
            "score_logprob": None,
            "prefix_token_mismatch": False,
            "tokens": [],
            "prob": None,
        }
        for label, candidate in CHOICE_CANDIDATES.items()
    }


def probability_class_scores(probability: float | None) -> dict[str, dict[str, Any]]:
    """Represent the binary DE probability in the existing score schema."""

    scores = empty_class_scores()
    if probability is None:
        return scores
    no_probability = 1.0 - probability
    for label, value in (("yes", probability), ("no", no_probability)):
        # Keep the historical logprob fields populated with the log of the
        # model-reported probability while exposing the original probability
        # through ``prob`` for AUROC/AUPRC consumers.
        log_probability = math.log(max(value, 1e-300))
        scores[label].update(
            {
                "sequence_logprob": log_probability,
                "avg_logprob": log_probability,
                "first_token_logprob": log_probability,
                "score_logprob": log_probability,
                "prob": value,
            }
        )
    return scores


def extract_api_choice_scores(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert OpenAI chat logprobs for A/B/C into the vLLM score schema."""

    scores_by_label: dict[str, float | None] = {
        label: None for label in CHOICE_CANDIDATES
    }
    tokens_by_label: dict[str, list[dict[str, Any]]] = {
        label: [] for label in CHOICE_CANDIDATES
    }
    choices = payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []
    if isinstance(content, list):
        for token_info in content:
            if not isinstance(token_info, dict):
                continue
            candidates: list[tuple[Any, Any]] = []
            if token_info.get("token") is not None:
                candidates.append((token_info["token"], token_info.get("logprob")))
            candidates.extend(iter_top_logprob_items(token_info.get("top_logprobs")))
            for token, value in candidates:
                letter = normalize_choice_token(token)
                label = LETTER_TO_LABEL.get(letter)
                if label is None:
                    continue
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(score):
                    continue
                if scores_by_label[label] is None or score > scores_by_label[label]:
                    scores_by_label[label] = score
                tokens_by_label[label].append({"token": token, "logprob": score})
            if any(score is not None for score in scores_by_label.values()):
                break

    probs = softmax_from_scores(scores_by_label)
    return {
        label: {
            "candidate_text": CHOICE_CANDIDATES[label],
            "token_count": 1,
            "sequence_logprob": scores_by_label[label],
            "avg_logprob": scores_by_label[label],
            "first_token_logprob": scores_by_label[label],
            "score_logprob": scores_by_label[label],
            "prefix_token_mismatch": False,
            "tokens": tokens_by_label[label],
            "prob": probs[label],
        }
        for label in CHOICE_CANDIDATES
    }


def call_api_chat(
    *,
    prompt_text: str,
    model_name: str,
    base_url: str,
    api_key: str,
    request_timeout_s: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    import requests

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = request_timeout_s if request_timeout_s > 0 else None
    last_error: Exception | None = None
    for attempt in range(max(0, max_retries) + 1):
        try:
            response = requests.post(
                get_chat_endpoint(base_url),
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("API response is not a JSON object")
            return data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max(0, max_retries):
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"API request failed after retries: {last_error}") from last_error


def build_api_choice_prompt(prompt_text: str) -> str:
    return (
        f"{prompt_text.rstrip()}\n\n"
        "Now return exactly one JSON object and no markdown or additional text.\n"
        'Use this schema: {"answer":"yes|no|insufficient",'
        '"de_probability":0.00}.\n'
        '"de_probability" is the probability from 0 to 1 that the perturbation '
        "causes differential expression of the gene of interest. A percentage "
        "from 0 to 100 is also accepted.\n"
        'Set "answer" to "yes" or "no" when you can make a binary prediction '
        'and include the numeric "de_probability". If evidence is insufficient, '
        'set "answer" to "insufficient" and omit "de_probability".\n'
        "The JSON object must be parseable by a standard JSON parser."
    )


def infer_one_api_prompt(args: argparse.Namespace, prompt: dict[str, Any]) -> dict[str, Any]:
    try:
        response = call_api_chat(
            prompt_text=prompt["prompt_text"],
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            request_timeout_s=args.request_timeout_s,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_retries=args.max_retries,
        )
        response_text = extract_api_message_text(response)
        judge_text = extract_judge_text(response_text)
        generated_label = normalize_label(judge_text)
        result: dict[str, Any] = {
            "ok": True,
            "prompt_index": prompt["prompt_index"],
            "prompt_title": prompt["prompt_title"],
            "response_text": response_text,
            "judge_text": judge_text,
            "generated_label": generated_label,
            "pred_label": generated_label,
        }
        if args.score_choices:
            result["class_scores"] = empty_class_scores()
            result["answered"] = False
            result["de_probability"] = None
            try:
                choice_response = call_api_chat(
                    prompt_text=build_api_choice_prompt(prompt["prompt_text"]),
                    model_name=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    request_timeout_s=args.request_timeout_s,
                    max_tokens=args.choice_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    max_retries=args.max_retries,
                )
                choice_text = extract_api_message_text(choice_response)
                parsed = parse_probability_response(choice_text)
                probability = parsed.get("de_probability")
                result["class_scores"] = probability_class_scores(probability)
                result["choice_pred_label"] = parsed.get("answer", "")
                result["choice_response_text"] = choice_text
                result["answered"] = bool(parsed.get("answered"))
                result["de_probability"] = probability
                if parsed.get("parse_error"):
                    result["probability_parse_error"] = parsed["parse_error"]
                class_scores = result["class_scores"]
                result["class_logits"] = {
                    label: score["score_logprob"]
                    for label, score in class_scores.items()
                }
                result["class_probs"] = {
                    label: score["prob"] for label, score in class_scores.items()
                }
                result["choice_score_mode"] = "self_reported_probability"
                if args.pred_label_source == "choice":
                    result["pred_label"] = parsed.get("answer") or generated_label
                else:
                    result["pred_label"] = generated_label or parsed.get("answer", "")
            except Exception as exc:  # noqa: BLE001
                result["choice_score_error"] = repr(exc)
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "prompt_index": prompt["prompt_index"],
            "prompt_title": prompt["prompt_title"],
            "response_text": "",
            "judge_text": "",
            "error": repr(exc),
        }


def run_file(
    args: argparse.Namespace,
    input_path: Path,
    out_path: Path,
    *,
    multiple_input_files: bool,
) -> None:
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    prompts = split_prompts_from_text(text)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    total = len(prompts)
    if total == 0:
        print(f"{input_path.name}: no prompts found.")
        return

    truth_path = Path(args.truth_csv) if args.truth_csv else infer_truth_csv(input_path)
    truth_labels: dict[tuple[str, str], int] = {}
    metrics_path: Path | None = None
    if truth_path is not None:
        if not truth_path.is_file():
            raise FileNotFoundError(f"Truth CSV not found: {truth_path}")
        truth_labels = load_truth_labels(truth_path)
        metrics_path = Path(args.metrics_output) if args.metrics_output else out_path.with_name(
            f"{out_path.stem}_metrics.json"
        )
        if multiple_input_files and args.metrics_output:
            metrics_path = metrics_path.with_name(
                f"{metrics_path.stem}_{input_path.stem}{metrics_path.suffix}"
            )

    def write_metrics(ordered: list[dict[str, Any]]) -> None:
        if metrics_path is None or truth_path is None:
            return
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_metrics_json(
            metrics_path,
            input_file=input_path,
            truth_file=truth_path,
            metrics=compute_binary_metrics(ordered, truth_labels),
        )

    done_results = load_done_results(out_path) if args.resume else {}
    prompt_by_idx = {p["prompt_index"]: p for p in prompts}
    done_results = {
        k: v
        for k, v in done_results.items()
        if k in prompt_by_idx and result_has_required_fields(v, args)
    }

    pending = [p for p in prompts if p["prompt_index"] not in done_results]
    completed = len(done_results)

    print(
        f"{input_path.name}: total={total}, resumed_done={len(done_results)}, pending={len(pending)}, batch_size={args.batch_size}"
    )

    if not pending:
        ordered = [done_results[p["prompt_index"]] for p in prompts if p["prompt_index"] in done_results]
        if args.prediction_csv:
            prediction_csv = Path(args.prediction_csv)
            if multiple_input_files:
                prediction_csv = prediction_csv.with_name(
                    f"{prediction_csv.stem}_{input_path.stem}{prediction_csv.suffix}"
                )
            prediction_csv.parent.mkdir(parents=True, exist_ok=True)
            write_prediction_csv(
                prediction_csv,
                ordered,
                input_file=input_path,
                cell_line=args.cell_line,
            )
        write_metrics(ordered)
        print(f"Saved: {out_path}")
        return

    for batch in batched(pending, args.batch_size):
        t0 = time.time()
        # API requests are network-bound; use the existing batch-size setting
        # as the worker count while preserving prompt order in ``map``.
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            results = list(executor.map(lambda p: infer_one_api_prompt(args, p), batch))
        for p, result in zip(batch, results, strict=True):
            done_results[p["prompt_index"]] = result
            completed += 1
            status = "done" if result.get("ok") else "error"
            print(
                f"[{completed}/{total}] Prompt {p['prompt_index']}: {status} "
                f"in {time.time() - t0:.1f}s."
            )

        ordered = [done_results[p["prompt_index"]] for p in prompts if p["prompt_index"] in done_results]
        if args.format == "json":
            write_json(
                out_path,
                model_name=args.model,
                input_file=input_path,
                prompt_count=total,
                ordered_results=ordered,
                save_prompts=args.save_prompts,
                prompts=prompts,
            )
        else:
            write_txt(out_path, input_path, ordered)
        if args.prediction_csv:
            prediction_csv = Path(args.prediction_csv)
            if multiple_input_files:
                prediction_csv = prediction_csv.with_name(
                    f"{prediction_csv.stem}_{input_path.stem}{prediction_csv.suffix}"
                )
            prediction_csv.parent.mkdir(parents=True, exist_ok=True)
            write_prediction_csv(
                prediction_csv,
                ordered,
                input_file=input_path,
                cell_line=args.cell_line,
            )
        write_metrics(ordered)

    print(f"Saved: {out_path}")
    if metrics_path is not None:
        metrics = compute_binary_metrics(ordered, truth_labels)
        print(
            f"Metrics: answered={metrics['n_answered']}/{metrics['n_truth_matched']}, "
            f"AUROC={metrics['auroc']}, AUPRC={metrics['auprc']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch inference for prompt files through an OpenAI-compatible API."
    )
    parser.add_argument("--input-dir", type=str, default=".")
    parser.add_argument("--glob", type=str, default="*.txt")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key; defaults to OPENAI_API_KEY, API_KEY, or PROXY_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL or full /v1/chat/completions URL.",
    )
    parser.add_argument("--format", type=str, choices=["json", "txt"], default="json")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout-s", type=float, default=90.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--score-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--choice-score-mode", choices=["first_token", "avg", "sum"], default="first_token")
    parser.add_argument("--choice-prompt-logprobs", type=int, default=20)
    parser.add_argument(
        "--choice-max-tokens",
        type=int,
        default=128,
        help="Maximum tokens for the structured probability JSON response.",
    )
    parser.add_argument("--pred-label-source", choices=["generated", "choice"], default="generated")
    parser.add_argument("--prediction-csv", type=str, default=None)
    parser.add_argument("--cell-line", type=str, default=None)
    parser.add_argument(
        "--truth-csv",
        type=str,
        default=None,
        help="Truth CSV with pert,gene,label columns; auto-detected beside the input file.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional companion JSON path for answered-only AUROC/AUPRC metrics.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.api_key is None:
        args.api_key = (
            os.getenv("OPENAI_API_KEY", "").strip()
            or os.getenv("API_KEY", "").strip()
            or os.getenv("PROXY_API_KEY", "").strip()
        )
    if args.base_url is None:
        args.base_url = (
            os.getenv("OPENAI_BASE_URL", "").strip()
            or os.getenv("API_BASE_URL", "").strip()
            or os.getenv("PROXY_BASE_URL", "").strip()
        )
    if not args.base_url:
        raise ValueError("Missing API base URL. Provide --base-url or set OPENAI_BASE_URL.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.choice_max_tokens < 1:
        raise ValueError("--choice-max-tokens must be >= 1")

    input_dir = Path(args.input_dir)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    glob_pattern = Path(args.glob)
    if glob_pattern.is_absolute():
        files = sorted(Path(p) for p in expand_glob(args.glob))
    else:
        files = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched: dir={input_dir}, glob={args.glob}")

    model_slug = slugify_model_name(args.model)
    for input_path in files:
        out_path = output_dir / f"{input_path.stem}_{model_slug}.{args.format}"
        run_file(args, input_path, out_path, multiple_input_files=len(files) > 1)


if __name__ == "__main__":
    main()
