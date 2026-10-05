import argparse
import csv
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


if __name__ == "__main__":
    configure_process_library_path()
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


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


# vLLM uses multiprocessing with spawn. The marker is set in the parent after
# CUDA_VISIBLE_DEVICES is configured, then inherited by engine and worker
# processes so they install the same narrow weight-loader patch on startup.
if os.environ.get(GRAVITY_VLLM_COMPAT_ENV) == "1":
    install_gravity_vllm_weight_loader()


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
    if args.score_choices and not isinstance(row.get("class_scores"), dict):
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
            }
            for label in CHOICE_CANDIDATES:
                score = class_scores.get(label) or {}
                row[f"score_{label}"] = score.get("score_logprob")
                row[f"logit_{label}"] = score.get("score_logprob")
                row[f"prob_{label}"] = score.get("prob")
            writer.writerow(row)


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
        print(f"Saved: {out_path}")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    runtime_model = getattr(args, "runtime_model", args.model)
    llm_kwargs: dict[str, Any] = {
        "model": runtime_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "load_format": args.load_format,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.swap_space is not None:
        llm_kwargs["swap_space"] = args.swap_space
    if args.model_impl is not None:
        llm_kwargs["model_impl"] = args.model_impl
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        runtime_model,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    choice_sampling_params = None
    if args.score_choices:
        choice_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=args.choice_prompt_logprobs,
        )

    for batch in batched(pending, args.batch_size):
        prompts_for_llm = [
            build_chat_prompt(tokenizer, p["prompt_text"], args.use_chat_template)
            for p in batch
        ]
        t0 = time.time()
        outputs = llm.generate(prompts_for_llm, sampling_params)
        choice_scores = []
        if choice_sampling_params is not None:
            choice_prompts = [
                prompt + candidate
                for prompt in prompts_for_llm
                for candidate in CHOICE_CANDIDATES.values()
            ]
            choice_outputs = llm.generate(choice_prompts, choice_sampling_params)
            choice_scores = score_choice_outputs(
                tokenizer,
                prompts_for_llm,
                choice_outputs,
                score_mode=args.choice_score_mode,
            )
        for batch_i, (p, out) in enumerate(zip(batch, outputs, strict=True)):
            text_out = ""
            if out.outputs:
                text_out = out.outputs[0].text or ""
            generated_label = normalize_label(extract_judge_text(text_out))
            result = {
                "ok": True,
                "prompt_index": p["prompt_index"],
                "prompt_title": p["prompt_title"],
                "response_text": text_out,
                "judge_text": extract_judge_text(text_out),
                "generated_label": generated_label,
            }
            if choice_scores:
                scored = choice_scores[batch_i]
                choice_pred_label = scored["pred_label"]
                result["choice_pred_label"] = choice_pred_label
                if args.pred_label_source == "choice":
                    result["pred_label"] = choice_pred_label or generated_label
                else:
                    result["pred_label"] = generated_label or choice_pred_label
                result["class_scores"] = scored["class_scores"]
                result["class_logits"] = {
                    label: score.get("score_logprob")
                    for label, score in scored["class_scores"].items()
                }
                result["class_probs"] = {
                    label: score.get("prob")
                    for label, score in scored["class_scores"].items()
                }
                result["choice_score_mode"] = args.choice_score_mode
            else:
                result["pred_label"] = generated_label
            done_results[p["prompt_index"]] = result
            completed += 1
            print(
                f"[{completed}/{total}] Prompt {p['prompt_index']}: done in {time.time() - t0:.1f}s."
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

    print(f"Saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch inference for prompt files using local vLLM/Qwen."
    )
    parser.add_argument("--input-dir", type=str, default=".")
    parser.add_argument("--glob", type=str, default="*.txt")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--format", type=str, choices=["json", "txt"], default="json")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids to expose to vLLM, e.g. 0,1,2,3",
    )
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        help="vLLM weight loading format, e.g. auto or safetensors",
    )
    parser.add_argument(
        "--model-impl",
        type=str,
        default=None,
        help='Force a specific backend, e.g. "vllm" or "transformers".',
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of each visible GPU memory vLLM may use. Lower this if startup fails due to limited free VRAM.",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--swap-space",
        type=int,
        default=None,
        help="Deprecated vLLM option; omitted by default because vLLM 0.19 ignores it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--choice-score-mode", choices=["first_token", "avg", "sum"], default="first_token")
    parser.add_argument("--choice-prompt-logprobs", type=int, default=20)
    parser.add_argument("--pred-label-source", choices=["generated", "choice"], default="generated")
    parser.add_argument("--prediction-csv", type=str, default=None)
    parser.add_argument("--cell-line", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu_ids:
        import os

        gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
        if not gpu_ids:
            raise ValueError("--gpu-ids was provided but no valid ids were parsed.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        if args.tensor_parallel_size > len(gpu_ids):
            raise ValueError(
                f"--tensor-parallel-size ({args.tensor_parallel_size}) cannot exceed "
                f"the number of visible GPUs ({len(gpu_ids)})."
            )

    input_dir = Path(args.input_dir)
    model_path = Path(args.model)

    validate_local_checkpoint_files(model_path)
    prepare_model_for_vllm(args, model_path)

    num_attention_heads = read_num_attention_heads(model_path)
    if num_attention_heads is not None and num_attention_heads % args.tensor_parallel_size != 0:
        valid = valid_tensor_parallel_sizes(num_attention_heads)
        raise ValueError(
            f"tensor-parallel-size={args.tensor_parallel_size} is invalid for {model_path}. "
            f"The model has {num_attention_heads} attention heads, so TP must divide it exactly. "
            f"Valid values: {valid}"
        )

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
