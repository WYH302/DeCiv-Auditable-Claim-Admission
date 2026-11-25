"""Run frontier closed-weight LLM verifier (gpt-6.1-sol) on DeCiv-Real-1200 via Responses API.

Reads API credentials strictly from environment variables (never hardcoded):
  - DECIV_API_KEY (or OPENAI_API_KEY)
  - DECIV_API_BASE_URL (default: https://api.openai.com/v1)
  - DECIV_API_MODEL (default: gpt-6.1-sol)
  - DECIV_API_EXTRA_HEADERS (optional JSON dict of extra HTTP headers)

Supports single-pair mode (--batch-size 1) and parent-disjoint batched mode (--batch-size 8)
where every item in a batch belongs to a distinct parent cluster (zero cross-variant leakage),
saving incremental checkpoints to results/real_verifiers/gpt61_sol_checkpoint.jsonl and
exporting results/real_verifiers/gpt61_sol_probs.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_real_verifiers_gpu import DATA_PATH, OUT_DIR, STATUSES, load_unique_pairs, text_key

SYSTEM_INSTRUCTIONS = (
    "You are a calibrated factual verification judge for a persistent knowledge store. "
    "Evaluate each independent (Evidence, Claim) item strictly on its own merits. "
    "Because passages come from encyclopedic or scientific source articles about the subject entity, resolve standard "
    "pronouns or definite references in the evidence (e.g., 'she', 'he', 'it', 'the film', 'the city') to "
    "the claim's subject entity when contextually consistent; however, do NOT invent facts absent from the evidence. "
    "Classify each item into four mutually exclusive epistemic states with calibrated probabilities summing to 1.0:\n"
    "- supported: All core and secondary factual assertions (entities, dates, numbers, relationships, and every clause) in the claim are supported by the evidence.\n"
    "- contradicted: The core factual answer, entity, relationship, or truth value in the claim directly conflicts with or is refuted by the evidence.\n"
    "- partially_supported: The core assertion of the claim is supported by the evidence, BUT the claim also introduces a localized numeric/date/scope error OR appends an extra unsupported secondary clause/qualifier not backed by the evidence.\n"
    "- insufficient_evidence: The evidence lacks a key bridge fact or does not contain enough information to verify or refute the core claim.\n"
    "Output ONLY valid compact JSON with no markdown or explanation."
)


def _norm_vec(obj: dict) -> list[float]:
    vals = []
    for k in STATUSES:
        v = float(obj.get(k, 0.0))
        if not np.isfinite(v) or v < 0.0:
            v = 0.0
        vals.append(v)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.sum() <= 1e-12:
        arr = np.full(4, 0.25, dtype=np.float64)
    else:
        arr = arr / arr.sum()
    arr = np.clip(arr, 1e-4, 1.0)
    arr = arr / arr.sum()
    return [round(float(x), 6) for x in arr]


def _extract_text(raw: dict) -> str:
    out_txt = raw.get("output_text")
    if out_txt:
        return str(out_txt)
    parts = []
    for item in raw.get("output", []):
        for c in item.get("content", []):
            if c.get("type") in ("output_text", "text"):
                parts.append(c.get("text", ""))
    return "\n".join(parts)


def call_batch(
    base_url: str,
    api_key: str,
    model: str,
    batch_items: list[tuple[str, str, str]],
    max_retries: int = 6,
) -> tuple[dict[str, list[float]], dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "OpenAI/Python",
    }
    extra_hdr = os.environ.get("DECIV_API_EXTRA_HEADERS", "").strip()
    if extra_hdr:
        headers.update(json.loads(extra_hdr))
    if len(batch_items) == 1:
        tk, claim, ev = batch_items[0]
        user_input = (
            f"Primary Evidence Passage:\n{ev}\n\nCandidate Claim:\n{claim}\n\n"
            'Return ONLY JSON: {"supported":p1,"contradicted":p2,"partially_supported":p3,"insufficient_evidence":p4}'
        )
        max_toks = 256
    else:
        blocks = []
        id_map = {}
        for idx, (tk, claim, ev) in enumerate(batch_items, 1):
            sid = f"I{idx}"
            id_map[sid] = tk
            blocks.append(f"[{sid}]\nEvidence: {ev}\nClaim: {claim}")
        user_input = (
            f"Evaluate each of the {len(batch_items)} independent items below and return ONLY one JSON object "
            "mapping each item ID (I1..I"
            + str(len(batch_items))
            + ') to {"supported":p1,"contradicted":p2,"partially_supported":p3,"insufficient_evidence":p4}.\n\n'
            + "\n\n".join(blocks)
        )
        max_toks = 160 * len(batch_items) + 256

    payload = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": user_input,
        "reasoning": {"effort": "low"},
        "max_output_tokens": max_toks,
        "store": False,
    }
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(f"{base_url.rstrip('/')}/responses", data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            txt = _extract_text(raw).strip()
            # Strip markdown fences if any
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S).strip()
            start = txt.find("{")
            end = txt.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"No JSON object in output: {txt[:200]}")
            parsed = json.loads(txt[start : end + 1])

            res_map: dict[str, list[float]] = {}
            if len(batch_items) == 1:
                tk = batch_items[0][0]
                if "supported" in parsed:
                    res_map[tk] = _norm_vec(parsed)
                elif "I1" in parsed:
                    res_map[tk] = _norm_vec(parsed["I1"])
                else:
                    raise ValueError(f"Unexpected single JSON keys: {list(parsed.keys())}")
            else:
                for idx, (tk, _, _) in enumerate(batch_items, 1):
                    sid = f"I{idx}"
                    if sid in parsed and isinstance(parsed[sid], dict):
                        res_map[tk] = _norm_vec(parsed[sid])
                if len(res_map) < len(batch_items):
                    raise ValueError(f"Incomplete batch response: got {len(res_map)}/{len(batch_items)}")

            usage = raw.get("usage") or {}
            return res_map, {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "returned_model": str(raw.get("model", model)),
            }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:300]
            last_err = f"HTTP {e.code}: {err_body}"
            time.sleep(min(30.0, (1.8 ** attempt) + 0.5))
        except Exception as e:
            last_err = repr(e)
            time.sleep(min(20.0, (1.5 ** attempt) + 0.5))
    raise RuntimeError(f"Batch failed after {max_retries} attempts: {last_err}")


def make_parent_disjoint_batches(
    todo_items: list[tuple[str, str, str]],
    tk_to_parent: dict[str, str],
    batch_size: int,
) -> list[list[tuple[str, str, str]]]:
    """Group todo items into batches where NO two items in a batch share the same parent_id."""
    if batch_size <= 1:
        return [[it] for it in todo_items]
    by_parent: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for it in todo_items:
        by_parent[tk_to_parent[it[0]]].append(it)
    # Round-robin peel one item per parent per pass -> guaranteed parent-disjoint batches!
    ordered: list[tuple[str, str, str]] = []
    parent_keys = sorted(by_parent.keys())
    while True:
        added = False
        for pk in parent_keys:
            if by_parent[pk]:
                ordered.append(by_parent[pk].pop(0))
                added = True
        if not added:
            break
    batches = []
    cur: list[tuple[str, str, str]] = []
    cur_pids: set[str] = set()
    for it in ordered:
        pid = tk_to_parent[it[0]]
        if len(cur) >= batch_size or pid in cur_pids:
            batches.append(cur)
            cur = []
            cur_pids = set()
        cur.append(it)
        cur_pids.add(pid)
    if cur:
        batches.append(cur)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", type=str, default=os.environ.get("DECIV_API_MODEL", "gpt-6.1-sol"))
    ap.add_argument("--base-url", type=str, default=os.environ.get("DECIV_API_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--verifier-id", type=str, default="gpt61_sol")
    args = ap.parse_args()

    api_key = os.environ.get("DECIV_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set DECIV_API_KEY or OPENAI_API_KEY in environment before running.")

    rows, unique_pairs = load_unique_pairs(DATA_PATH)
    tk_to_parent = {text_key(r["claim_text"], r["evidence_text"]): r["parent_id"] for r in rows}
    ckpt_path = OUT_DIR / f"{args.verifier_id}_checkpoint.jsonl"
    out_path = OUT_DIR / f"{args.verifier_id}_probs.json"

    cached: dict[str, list[float]] = {}
    tot_in_tok = 0
    tot_out_tok = 0
    returned_model = args.model
    if ckpt_path.exists():
        for line in ckpt_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            cached[rec["text_key"]] = rec["probs"]
            tot_in_tok += int(rec.get("input_tokens", 0))
            tot_out_tok += int(rec.get("output_tokens", 0))
            if rec.get("returned_model"):
                returned_model = rec["returned_model"]

    target_pairs = unique_pairs[: args.limit] if args.limit > 0 else unique_pairs
    todo = [item for item in target_pairs if item[0] not in cached]
    batches = make_parent_disjoint_batches(todo, tk_to_parent, args.batch_size)
    print(
        f"[{args.verifier_id}] model={args.model} | target={len(target_pairs)} | "
        f"cached={len(target_pairs)-len(todo)} | todo={len(todo)} | batches={len(batches)} (bs={args.batch_size}, workers={args.workers})",
        flush=True,
    )

    lock = threading.Lock()
    t0 = time.perf_counter()
    done_new = 0
    failed_items: list[tuple[str, str, str]] = []

    if batches:
        with ckpt_path.open("a", encoding="utf-8") as ckpt_f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(call_batch, args.base_url, api_key, args.model, b): b for b in batches}
                for fut in concurrent.futures.as_completed(futs):
                    b = futs[fut]
                    try:
                        res_map, meta = fut.result()
                        per_in = meta["input_tokens"] // max(1, len(res_map))
                        per_out = meta["output_tokens"] // max(1, len(res_map))
                        with lock:
                            for tk_res, probs in res_map.items():
                                cached[tk_res] = probs
                                ckpt_f.write(
                                    json.dumps(
                                        {
                                            "text_key": tk_res,
                                            "probs": probs,
                                            "input_tokens": per_in,
                                            "output_tokens": per_out,
                                            "returned_model": meta["returned_model"],
                                        }
                                    )
                                    + "\n"
                                )
                            ckpt_f.flush()
                            tot_in_tok += meta["input_tokens"]
                            tot_out_tok += meta["output_tokens"]
                            returned_model = meta["returned_model"]
                            done_new += len(res_map)
                            dt = time.perf_counter() - t0
                            rate = done_new / max(1e-6, dt)
                            eta = (len(todo) - done_new) / max(1e-6, rate)
                            print(
                                f"  [{args.verifier_id}] {len(cached)}/{len(target_pairs)} total "
                                f"(+{done_new}/{len(todo)} new in {dt:.1f}s, {rate:.2f} items/s, ETA {eta:.0f}s | "
                                f"tok in={tot_in_tok} out={tot_out_tok})",
                                flush=True,
                            )
                    except Exception as e:
                        with lock:
                            print(f"  [BATCH WARN] {len(b)} items failed ({e}); queuing single-item fallback", flush=True)
                            failed_items.extend(b)

            # Fallback single-item retries for any failed batch items
            if failed_items:
                print(f"  [{args.verifier_id}] Running single-item fallback for {len(failed_items)} items...", flush=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, args.workers)) as ex2:
                    futs2 = {ex2.submit(call_batch, args.base_url, api_key, args.model, [it]): it for it in failed_items}
                    for fut in concurrent.futures.as_completed(futs2):
                        it = futs2[fut]
                        try:
                            res_map, meta = fut.result()
                            with lock:
                                for tk_res, probs in res_map.items():
                                    cached[tk_res] = probs
                                    ckpt_f.write(
                                        json.dumps(
                                            {
                                                "text_key": tk_res,
                                                "probs": probs,
                                                "input_tokens": meta["input_tokens"],
                                                "output_tokens": meta["output_tokens"],
                                                "returned_model": meta["returned_model"],
                                            }
                                        )
                                        + "\n"
                                    )
                                ckpt_f.flush()
                                tot_in_tok += meta["input_tokens"]
                                tot_out_tok += meta["output_tokens"]
                                done_new += len(res_map)
                        except Exception as e2:
                            print(f"  [FINAL ERROR] {it[0]}: {e2}", flush=True)

    elapsed = time.perf_counter() - t0
    covered = [r for r in rows if text_key(r["claim_text"], r["evidence_text"]) in cached]
    y_true = np.array([STATUSES.index(r["semantic_status"]) for r in covered], dtype=np.int64)
    probs_arr = np.array([cached[text_key(r["claim_text"], r["evidence_text"])] for r in covered], dtype=np.float64)
    y_pred = np.argmax(probs_arr, axis=1)
    acc = float(np.mean(y_pred == y_true))
    recalls = {}
    for idx, st in enumerate(STATUSES):
        mask = y_true == idx
        recalls[st] = round(float(np.mean(y_pred[mask] == idx)) if np.any(mask) else 0.0, 4)
    print(
        f"[{args.verifier_id}] Covered {len(covered)}/1200 packets ({len(cached)}/{len(unique_pairs)} unique) | "
        f"4-Class Acc = {acc*100:.2f}% | Per-class recall = {recalls} | wall={elapsed:.1f}s",
        flush=True,
    )

    if len(cached) == len(unique_pairs):
        out_obj = {
            "verifier_id": args.verifier_id,
            "model_name": returned_model,
            "family": "Frontier Closed-Weight LLM (API)",
            "n_unique_pairs": len(unique_pairs),
            "n_packets": len(rows),
            "accuracy_4class": round(acc, 4),
            "rec_supported": recalls["supported"],
            "rec_contradicted": recalls["contradicted"],
            "rec_partial": recalls["partially_supported"],
            "rec_insufficient": recalls["insufficient_evidence"],
            "inference_seconds": round(elapsed, 2),
            "throughput_pairs_per_s": round(len(unique_pairs) / max(1e-6, elapsed), 2),
            "peak_vram_gb": 0.0,
            "total_input_tokens": tot_in_tok,
            "total_output_tokens": tot_out_tok,
            "probs": {k: cached[k] for k, _, _ in unique_pairs},
        }
        out_path.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")
        print(f"Saved complete verifier output -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
