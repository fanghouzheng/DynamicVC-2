import argparse
import csv
import heapq
import json
import math
import multiprocessing as mp
import os
import re
import time
from collections import deque
from glob import glob as expand_glob
from pathlib import Path
from typing import Any


PROMPT_HEADER_RE = re.compile(
    r"^===\s*Prompt\s*(\d+)\s*\((.*?)\)\s*===\s*$",
    re.MULTILINE,
)

CHOICE_CANDIDATES = {
    "yes": "Yes.",
    "no": "No.",
    "insufficient": "There is insufficient evidence to determine.",
}

CHOICE_LETTERS = {
    "yes": "B",
    "no": "A",
    "insufficient": "C",
}

LETTER_TO_LABEL = {letter: label for label, letter in CHOICE_LETTERS.items()}


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


def softmax_from_scores(scores: dict[str, float | None]) -> dict[str, float | None]:
    finite = {k: v for k, v in scores.items() if v is not None and math.isfinite(v)}
    if not finite:
        return {k: None for k in scores}
    max_score = max(finite.values())
    exps = {k: math.exp(v - max_score) for k, v in finite.items()}
    denom = sum(exps.values())
    return {k: (exps[k] / denom if k in exps else None) for k in scores}


def build_choice_prompt(prompt_text: str) -> str:
    return (
        f"{prompt_text.rstrip()}\n\n"
        "Now answer the final prediction again using exactly one letter and no other text.\n"
        "A = No. Perturbation does not impact the gene of interest.\n"
        "B = Yes. Perturbation results in differential expression of the gene of interest.\n"
        "C = There is insufficient evidence to determine the effect.\n"
        "Final answer letter:"
    )


def get_chat_endpoint(base_url: str | None) -> str:
    if not base_url:
        raise ValueError("base_url is required for OpenAI-compatible proxy mode.")

    b = base_url.rstrip("/")
    if b.endswith("/v1/chat/completions"):
        return b
    if b.endswith("/v1"):
        return f"{b}/chat/completions"
    return f"{b}/v1/chat/completions"


def extract_message_text(data: dict[str, Any]) -> str:
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return text or json.dumps(data, ensure_ascii=False)


def iter_top_logprob_items(top_logprobs: Any):
    if isinstance(top_logprobs, dict):
        for token, value in top_logprobs.items():
            if isinstance(value, dict):
                yield token, value.get("logprob")
            else:
                yield token, value
        return

    if isinstance(top_logprobs, list):
        for item in top_logprobs:
            if not isinstance(item, dict):
                continue
            token = item.get("token")
            logprob = item.get("logprob")
            if token is not None:
                yield token, logprob


def normalize_choice_token(token: Any) -> str:
    return str(token or "").strip().upper().strip("`*_.,:;)]}")


def extract_choice_scores_from_logprobs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    choice = (data.get("choices") or [{}])[0]
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []
    scores_by_letter: dict[str, float | None] = {letter: None for letter in LETTER_TO_LABEL}
    token_rows_by_letter: dict[str, list[dict[str, Any]]] = {letter: [] for letter in LETTER_TO_LABEL}

    if isinstance(content, list):
        for token_info in content:
            if not isinstance(token_info, dict):
                continue
            candidates = []
            token = token_info.get("token")
            if token is not None:
                candidates.append((token, token_info.get("logprob")))
            candidates.extend(iter_top_logprob_items(token_info.get("top_logprobs")))

            for token, logprob in candidates:
                letter = normalize_choice_token(token)
                if letter not in LETTER_TO_LABEL:
                    continue
                try:
                    score = float(logprob)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(score):
                    continue
                if scores_by_letter[letter] is None or score > scores_by_letter[letter]:
                    scores_by_letter[letter] = score
                token_rows_by_letter[letter].append(
                    {
                        "token": token,
                        "logprob": score,
                    }
                )

            if any(score is not None for score in scores_by_letter.values()):
                break

    scores_by_label = {
        label: scores_by_letter[letter]
        for letter, label in LETTER_TO_LABEL.items()
    }
    probs = softmax_from_scores(scores_by_label)
    class_scores = {}
    for label, letter in CHOICE_LETTERS.items():
        score = scores_by_label[label]
        class_scores[label] = {
            "candidate_text": letter,
            "token_count": 1,
            "sequence_logprob": score,
            "avg_logprob": score,
            "first_token_logprob": score,
            "score_logprob": score,
            "prefix_token_mismatch": False,
            "tokens": token_rows_by_letter[letter],
            "prob": probs[label],
        }
    return class_scores


def call_openai_compatible_chat(
    *,
    prompt_text: str,
    model_name: str,
    api_key: str,
    base_url: str | None,
    request_timeout_s: float,
    max_tokens: int | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    import requests

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = (request_timeout_s, request_timeout_s) if request_timeout_s > 0 else None
    resp = requests.post(
        get_chat_endpoint(base_url),
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _worker_call_once(
    child_conn,
    *,
    prompt_index: int,
    prompt_title: str,
    prompt_text: str,
    model_name: str,
    api_key: str,
    base_url: str | None,
    vertexai: bool,
    request_timeout_s: float,
    score_choices: bool,
    choice_top_logprobs: int,
    pred_label_source: str,
) -> None:
    try:
        data = call_openai_compatible_chat(
            prompt_text=prompt_text,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            request_timeout_s=request_timeout_s,
        )
        text = extract_message_text(data)
        judge_text = extract_judge_text(text)
        generated_label = normalize_label(judge_text)
        result = {
            "ok": True,
            "prompt_index": prompt_index,
            "prompt_title": prompt_title,
            "response_text": text,
            "judge_text": judge_text,
            "generated_label": generated_label,
            "pred_label": generated_label,
        }

        if score_choices:
            try:
                choice_data = call_openai_compatible_chat(
                    prompt_text=build_choice_prompt(prompt_text),
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    request_timeout_s=request_timeout_s,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=choice_top_logprobs,
                )
                choice_text = extract_message_text(choice_data)
                choice_letter = normalize_choice_token(choice_text[:8])
                choice_pred_label = LETTER_TO_LABEL.get(choice_letter, normalize_label(choice_text))
                class_scores = extract_choice_scores_from_logprobs(choice_data)
                finite_scores = {
                    label: score.get("score_logprob")
                    for label, score in class_scores.items()
                    if score.get("score_logprob") is not None
                }
                if finite_scores:
                    choice_pred_label = max(finite_scores, key=finite_scores.get)
                result["choice_pred_label"] = choice_pred_label
                result["class_scores"] = class_scores
                result["class_logits"] = {
                    label: score.get("score_logprob")
                    for label, score in class_scores.items()
                }
                result["class_probs"] = {
                    label: score.get("prob")
                    for label, score in class_scores.items()
                }
                result["choice_score_mode"] = "first_token"
                result["choice_response_text"] = choice_text
                if pred_label_source == "choice":
                    result["pred_label"] = choice_pred_label or generated_label
                else:
                    result["pred_label"] = generated_label or choice_pred_label
            except Exception as e:  # noqa: BLE001
                result["choice_score_error"] = repr(e)

        child_conn.send(result)
    except Exception as e:  # noqa: BLE001
        child_conn.send(
            {
                "ok": False,
                "prompt_index": prompt_index,
                "prompt_title": prompt_title,
                "error": repr(e),
            }
        )
    finally:
        try:
            child_conn.close()
        except Exception:
            pass


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
    if args.score_choices and not isinstance(row.get("class_scores"), dict) and not row.get("choice_score_error"):
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
                "generated_label": r.get("generated_label") or normalize_label(r.get("judge_text")),
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

    pending = deque()
    for p in prompts:
        idx = p["prompt_index"]
        if idx in done_results:
            continue
        pending.append(
            {
                "prompt_index": p["prompt_index"],
                "prompt_title": p["prompt_title"],
                "prompt_text": p["prompt_text"],
                "attempt": 1,
            }
        )

    print(
        f"{input_path.name}: total={total}, resumed_done={len(done_results)}, pending={len(pending)}, concurrency={args.concurrency}"
    )

    def write_outputs(ordered: list[dict[str, Any]]) -> None:
        if args.format == "json":
            write_json(
                out_path,
                model_name=args.model_name,
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

    retries_heap: list[tuple[float, int, dict[str, Any]]] = []
    retry_seq = 0
    active: dict[int, dict[str, Any]] = {}
    ctx = mp.get_context("spawn")

    completed = len(done_results)

    if not pending:
        ordered = [done_results[p["prompt_index"]] for p in prompts if p["prompt_index"] in done_results]
        write_outputs(ordered)
        print(f"Saved: {out_path}")
        return

    def finalize_result(result: dict[str, Any]) -> None:
        nonlocal completed
        idx = result["prompt_index"]
        if idx in done_results:
            return
        done_results[idx] = result
        completed += 1

        ordered = [done_results[p["prompt_index"]] for p in prompts if p["prompt_index"] in done_results]
        write_outputs(ordered)

        status = "done" if not result.get("error") else "error"
        print(f"[{completed}/{total}] Prompt {idx}: {status}.")

    while completed < total:
        now = time.time()

        while retries_heap and retries_heap[0][0] <= now:
            _, _, task = heapq.heappop(retries_heap)
            pending.append(task)

        while len(active) < args.concurrency and pending:
            task = pending.popleft()
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_worker_call_once,
                kwargs={
                    "child_conn": child_conn,
                    "prompt_index": task["prompt_index"],
                    "prompt_title": task["prompt_title"],
                    "prompt_text": task["prompt_text"],
                    "model_name": args.model_name,
                    "api_key": args.api_key,
                    "base_url": args.base_url,
                    "vertexai": args.vertexai,
                    "request_timeout_s": args.request_timeout_s,
                    "score_choices": args.score_choices,
                    "choice_top_logprobs": args.choice_top_logprobs,
                    "pred_label_source": args.pred_label_source,
                },
                daemon=True,
            )
            proc.start()
            active[proc.pid] = {
                "proc": proc,
                "conn": parent_conn,
                "task": task,
                "start_ts": time.time(),
            }
            child_conn.close()
            if args.launch_interval_s > 0:
                time.sleep(args.launch_interval_s)

        progressed = False
        now = time.time()
        for pid, info in list(active.items()):
            proc = info["proc"]
            conn = info["conn"]
            task = info["task"]
            elapsed = now - info["start_ts"]

            if conn.poll():
                msg = conn.recv()
                try:
                    proc.join(timeout=0.1)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
                active.pop(pid, None)
                progressed = True

                if msg.get("ok"):
                    finalize_result(msg)
                else:
                    if task["attempt"] < args.max_retries:
                        task["attempt"] += 1
                        retry_wait = min(60, 2 ** (task["attempt"] - 1))
                        retry_seq += 1
                        heapq.heappush(retries_heap, (time.time() + retry_wait, retry_seq, task))
                        print(
                            f"Prompt {task['prompt_index']} failed (attempt {task['attempt']-1}/{args.max_retries}), retry in {retry_wait}s: {msg.get('error')}"
                        )
                    else:
                        finalize_result(
                            {
                                "prompt_index": task["prompt_index"],
                                "prompt_title": task["prompt_title"],
                                "response_text": "",
                                "judge_text": "",
                                "error": msg.get("error", "unknown_error"),
                            }
                        )
                continue

            if elapsed > args.hard_timeout_s:
                try:
                    proc.terminate()
                    proc.join(timeout=1.0)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
                active.pop(pid, None)
                progressed = True

                if task["attempt"] < args.max_retries:
                    task["attempt"] += 1
                    retry_wait = min(60, 2 ** (task["attempt"] - 1))
                    retry_seq += 1
                    heapq.heappush(retries_heap, (time.time() + retry_wait, retry_seq, task))
                    print(
                        f"Prompt {task['prompt_index']} hard-timeout>{args.hard_timeout_s}s, retry in {retry_wait}s (attempt {task['attempt']}/{args.max_retries})."
                    )
                else:
                    finalize_result(
                        {
                            "prompt_index": task["prompt_index"],
                            "prompt_title": task["prompt_title"],
                            "response_text": "",
                            "judge_text": "",
                            "error": f"hard_timeout>{args.hard_timeout_s}s",
                        }
                    )

        if not progressed:
            time.sleep(0.2)

    for info in active.values():
        proc = info["proc"]
        conn = info["conn"]
        try:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=0.5)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    print(f"Saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robust batch Gemini inference with hard timeout, retry, and resume."
    )
    parser.add_argument("--input-dir", type=str, default=".")
    parser.add_argument("--glob", type=str, default="*.txt")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--format", type=str, choices=["json", "txt"], default="json")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout-s", type=float, default=90.0)
    parser.add_argument("--hard-timeout-s", type=float, default=150.0)
    parser.add_argument("--launch-interval-s", type=float, default=0.2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--vertexai", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--prediction-csv", type=str, default=None)
    parser.add_argument("--cell-line", type=str, default=None)
    parser.add_argument("--score-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--choice-top-logprobs",
        type=int,
        default=10,
        help="Top token logprobs requested for the A/B/C choice-scoring call.",
    )
    parser.add_argument(
        "--choice-score-mode",
        choices=["first_token", "avg", "sum"],
        default="first_token",
        help="Accepted for compatibility with batch_vllm_infer.py. Gemini scoring uses first-token A/B/C logprobs.",
    )
    parser.add_argument(
        "--pred-label-source",
        choices=["generated", "choice"],
        default="generated",
        help="Use generated final answer labels, or the A/B/C choice-scoring label when available.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.api_key is None:
        args.api_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("PROXY_API_KEY", "").strip()
        )
    if args.base_url is None:
        args.base_url = os.getenv("GEMINI_BASE_URL", "").strip() or None

    if not args.api_key:
        raise ValueError("Missing API key. Provide --api-key or set GEMINI_API_KEY/PROXY_API_KEY.")

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

    model_slug = slugify_model_name(args.model_name)
    for input_path in files:
        out_path = output_dir / f"{input_path.stem}_{model_slug}.{args.format}"
        run_file(args, input_path, out_path, multiple_input_files=len(files) > 1)


if __name__ == "__main__":
    main()
