"""Run 4 genuinely heterogeneous open-source neural verifiers on DeCiv-Real-1200 (RTX 3090 24GB).

Models (4 distinct architectures & model families):
  1. deberta_v3_large : MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (435M encoder-only NLI, fp16)
  2. bart_large_mnli  : facebook/bart-large-mnli (406M encoder-decoder NLI, fp16)
  3. llama31_8b       : Meta Llama-3.1-8B-Instruct (8.0B decoder LLM via local CUDA Ollama, exact logprobs)
  4. qwen35_9b        : Alibaba Qwen3.5-9B (9.0B decoder LLM via local CUDA Ollama, exact logprobs)

Each verifier emits continuous 4-class probability vectors over:
    (supported, contradicted, partially_supported, insufficient_evidence)
with exact deduplication over identical (claim_text, evidence_text) pairs so v0 and v1
receive identical verifier posteriors without floating-point drift.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "deciv_real_1200.jsonl"
OUT_DIR = ROOT / "results" / "real_verifiers"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_URL = "http://127.0.0.1:11434"

STATUSES = ("supported", "contradicted", "partially_supported", "insufficient_evidence")


def text_key(claim: str, evidence: str) -> str:
    return hashlib.sha256((claim + "\n|||\n" + evidence).encode("utf-8")).hexdigest()[:24]


def extract_core_clause(claim: str) -> str:
    """Label-free syntactic extraction of the primary clause for sub-clause NLI decomposition."""
    for sep in (", and ", "; ", ", while ", ", although ", ", whereas "):
        if sep in claim:
            left = claim.split(sep, 1)[0].strip()
            if len(left) >= 18:
                return left.rstrip(".,;:") + "."
    m_year = re.search(r"\b(?:in |according to the |the )?(1[789]\d\d|20[012]\d)\b", claim)
    if m_year:
        core = re.sub(r"\s+", " ", claim[: m_year.start()] + claim[m_year.end() :]).strip()
        if len(core) >= 15:
            return core
    m_pct = re.search(r"\b\d+(?:\.\d+)?%", claim)
    if m_pct:
        core = re.sub(r"\s+", " ", claim[: m_pct.start()] + "a proportion" + claim[m_pct.end() :]).strip()
        if len(core) >= 15:
            return core
    return claim


def load_unique_pairs(path: Path) -> tuple[list[dict], list[tuple[str, str, str]]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    unique_map: dict[str, tuple[str, str]] = {}
    for r in rows:
        k = text_key(r["claim_text"], r["evidence_text"])
        if k not in unique_map:
            unique_map[k] = (r["claim_text"], r["evidence_text"])
    unique_list = [(k, v[0], v[1]) for k, v in unique_map.items()]
    return rows, unique_list


def _resolve_nli_label_indices(id2label: dict[int, str]) -> tuple[int, int, int]:
    ent = neu = con = -1
    for idx, name in id2label.items():
        low = str(name).lower()
        if "entail" in low:
            ent = int(idx)
        elif "neutral" in low:
            neu = int(idx)
        elif "contra" in low:
            con = int(idx)
    assert ent >= 0 and neu >= 0 and con >= 0, f"Unrecognized NLI labels: {id2label}"
    return ent, neu, con


def ollama_post(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_vram_gb(model_name: str) -> float:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        for m in data.get("models", []):
            if m.get("name", "").startswith(model_name.split(":")[0]):
                return float(m.get("size_vram", 0)) / 1e9
    except Exception:
        pass
    return 0.0


def ollama_unload(model_name: str) -> None:
    try:
        ollama_post("/api/chat", {"model": model_name, "messages": [], "keep_alive": 0}, timeout=30)
    except Exception:
        pass


SYSTEM_PROMPT = (
    "You are a strict fact-verification judge evaluating whether an Evidence passage supports a Claim. "
    "Classify the Claim against the Evidence into exactly one of four categories:\n"
    "A: SUPPORTED — Every factual assertion, entity, date, number, and clause in the Claim is fully supported by the Evidence.\n"
    "B: CONTRADICTED — The core factual assertion or main answer in the Claim is directly contradicted by the Evidence.\n"
    "C: PARTIALLY_SUPPORTED — The core proposition is supported by the Evidence, BUT a localized detail (such as a shifted year, wrong percentage, or an extra unverified secondary clause) is unsupported or inaccurate.\n"
    "D: INSUFFICIENT_EVIDENCE — The Evidence does not contain enough information to verify or refute the core Claim.\n"
    "Respond with ONLY a single capital letter: A, B, C, or D."
)


def extract_abcd_probs_from_logprobs(res: dict) -> np.ndarray:
    """Extract normalized 4-class probabilities over {A, B, C, D} from Ollama top_logprobs."""
    lp_list = res.get("logprobs") or []
    content = (res.get("message", {}).get("content") or "").strip().upper()
    letter_lps: dict[str, list[float]] = {ch: [] for ch in ("A", "B", "C", "D")}
    min_seen = -15.0

    # Find first token entry carrying letter candidates
    if lp_list:
        first_entry = lp_list[0]
        # Skip leading whitespace token if model emitted whitespace before letter
        if len(lp_list) > 1 and not first_entry.get("token", "").strip():
            first_entry = lp_list[1]
        cands = first_entry.get("top_logprobs") or [first_entry]
        for item in cands:
            tok = str(item.get("token", "")).strip().upper()
            lp = float(item.get("logprob", -20.0))
            min_seen = min(min_seen, lp)
            if tok in letter_lps:
                letter_lps[tok].append(lp)

    floor_lp = min_seen - 3.0
    logits = np.zeros(4, dtype=np.float64)
    has_any = any(bool(v) for v in letter_lps.values())
    if has_any:
        for idx, ch in enumerate(("A", "B", "C", "D")):
            vals = letter_lps[ch]
            if vals:
                m = max(vals)
                logits[idx] = m + math.log(sum(math.exp(v - m) for v in vals))
            else:
                logits[idx] = floor_lp
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
    else:
        # Fallback if logprobs lacked A/B/C/D tokens: parse generated letter
        probs = np.full(4, 0.05, dtype=np.float64)
        for idx, ch in enumerate(("A", "B", "C", "D")):
            if content.startswith(ch):
                probs[idx] = 0.85
                break
        probs /= np.sum(probs)

    probs = np.clip(probs, 1e-6, 1.0)
    return probs / probs.sum()


def score_ollama_pairs(
    model_name: str,
    pairs: list[tuple[str, str, str]],
    label_prefix: str = "",
) -> tuple[dict[str, list[float]], float, float]:
    """Score (key, claim, evidence) pairs via local CUDA Ollama with exact top-token logprobs."""
    # Warm-up load
    _ = ollama_post(
        "/api/chat",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply A."}],
            "stream": False,
            "think": False,
            "keep_alive": "15m",
            "options": {"temperature": 0.0, "num_predict": 1},
        },
        timeout=120,
    )
    vram_gb = ollama_vram_gb(model_name)
    probs_map: dict[str, list[float]] = {}
    t0 = time.perf_counter()

    for idx, (k, clm, ev) in enumerate(pairs):
        user_msg = f"Evidence:\n{ev}\n\nClaim:\n{clm}\n\nClassification (A, B, C, or D):"
        res = ollama_post(
            "/api/chat",
            {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "think": False,
                "logprobs": True,
                "top_logprobs": 20,
                "keep_alive": "15m",
                "options": {"temperature": 0.0, "num_predict": 2},
            },
            timeout=120,
        )
        probs_map[k] = extract_abcd_probs_from_logprobs(res).tolist()
        if (idx + 1) % 100 == 0 or (idx + 1) == len(pairs):
            dt = time.perf_counter() - t0
            rate = (idx + 1) / max(1e-6, dt)
            print(f"    [{label_prefix or model_name}] {idx+1}/{len(pairs)} ({dt:.1f}s, {rate:.1f} pairs/s)", flush=True)

    inf_s = time.perf_counter() - t0
    vram_gb = max(vram_gb, ollama_vram_gb(model_name))
    ollama_unload(model_name)
    return probs_map, inf_s, vram_gb


@torch.no_grad()
def run_nli_verifier(
    verifier_id: str,
    model_name: str,
    unique_pairs: list[tuple[str, str, str]],
    batch_size: int = 32,
) -> dict:
    out_file = OUT_DIR / f"{verifier_id}_probs.json"
    if out_file.exists():
        print(f"[CACHE HIT] {verifier_id} already saved at {out_file.name}", flush=True)
        return json.loads(out_file.read_text(encoding="utf-8"))

    print(f"\n=== Running NLI Verifier: {verifier_id} ({model_name}) ===", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float16).cuda().eval()
    load_s = time.perf_counter() - t_load0
    ent_i, neu_i, con_i = _resolve_nli_label_indices(model.config.id2label)

    def score_batch(premises: list[str], hyps: list[str]) -> np.ndarray:
        out = np.zeros((len(premises), 3), dtype=np.float64)
        for start in range(0, len(premises), batch_size):
            end = min(start + batch_size, len(premises))
            enc = tokenizer(
                premises[start:end], hyps[start:end], padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
            out[start:end, 0] = probs[:, ent_i]
            out[start:end, 1] = probs[:, neu_i]
            out[start:end, 2] = probs[:, con_i]
        return out

    evidences = [ev for _, _, ev in unique_pairs]
    claims = [clm for _, clm, _ in unique_pairs]
    cores = [extract_core_clause(clm) for clm in claims]

    t0 = time.perf_counter()
    p_full = score_batch(evidences, claims)
    p_core = score_batch(evidences, cores)
    torch.cuda.synchronize()
    inf_s = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    probs_map = {}
    for idx, (k, clm, _) in enumerate(unique_pairs):
        ef, nf, cf = float(p_full[idx, 0]), float(p_full[idx, 1]), float(p_full[idx, 2])
        es = float(p_core[idx, 0])
        has_sub = cores[idx] != clm
        pw = (0.75 * es) if has_sub else (0.20 * min(ef, 1.0 - ef) * 2.0)
        vec = np.array([ef, cf * (1.0 - pw), (nf + cf) * pw, nf * (1.0 - pw)], dtype=np.float64)
        vec = np.clip(vec, 1e-6, 1.0)
        probs_map[k] = (vec / vec.sum()).tolist()

    result = {
        "verifier_id": verifier_id,
        "model_name": model_name,
        "architecture": "nli_sequence_classifier_fp16",
        "n_unique_pairs": len(unique_pairs),
        "load_seconds": round(load_s, 2),
        "inference_seconds": round(inf_s, 2),
        "throughput_pairs_per_s": round(len(unique_pairs) / max(1e-6, inf_s), 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "probs": probs_map,
    }
    out_file.write_text(json.dumps(result), encoding="utf-8")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_ollama_verifier(verifier_id: str, ollama_model: str, display_name: str, unique_pairs: list[tuple[str, str, str]]) -> dict:
    out_file = OUT_DIR / f"{verifier_id}_probs.json"
    if out_file.exists():
        print(f"[CACHE HIT] {verifier_id} already saved at {out_file.name}", flush=True)
        return json.loads(out_file.read_text(encoding="utf-8"))

    print(f"\n=== Running LLM Verifier: {verifier_id} ({display_name} via Ollama CUDA) ===", flush=True)
    probs_map, inf_s, vram_gb = score_ollama_pairs(ollama_model, unique_pairs, label_prefix=verifier_id)
    result = {
        "verifier_id": verifier_id,
        "model_name": display_name,
        "ollama_tag": ollama_model,
        "architecture": "decoder_llm_cuda_gguf",
        "n_unique_pairs": len(unique_pairs),
        "inference_seconds": round(inf_s, 2),
        "throughput_pairs_per_s": round(len(unique_pairs) / max(1e-6, inf_s), 2),
        "peak_vram_gb": round(vram_gb, 3),
        "probs": probs_map,
    }
    out_file.write_text(json.dumps(result), encoding="utf-8")
    print(
        f"  Done {verifier_id}: {inf_s:.2f}s ({result['throughput_pairs_per_s']:.1f} pairs/s), VRAM={vram_gb:.2f} GB",
        flush=True,
    )
    return result


def main():
    rows, unique_pairs = load_unique_pairs(DATA_PATH)
    print(f"Loaded {len(rows)} packets -> {len(unique_pairs)} unique (claim, evidence) pairs.", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    results = [
        run_nli_verifier(
            "deberta_v3_large",
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            unique_pairs,
        ),
        run_nli_verifier(
            "bart_large_mnli",
            "facebook/bart-large-mnli",
            unique_pairs,
        ),
        run_ollama_verifier(
            "llama31_8b",
            "llama3.1:8b",
            "Meta-Llama-3.1-8B-Instruct",
            unique_pairs,
        ),
        run_ollama_verifier(
            "qwen35_9b",
            "qwen3.5:9b",
            "Qwen/Qwen3.5-9B",
            unique_pairs,
        ),
    ]

    summary_rows = []
    for res in results:
        vid = res["verifier_id"]
        pmap = res["probs"]
        correct = 0
        per_class_tot = {s: 0 for s in STATUSES}
        per_class_ok = {s: 0 for s in STATUSES}
        for r in rows:
            k = text_key(r["claim_text"], r["evidence_text"])
            pred = STATUSES[int(np.argmax(pmap[k]))]
            true_s = r["semantic_status"]
            per_class_tot[true_s] += 1
            if pred == true_s:
                correct += 1
                per_class_ok[true_s] += 1
        acc = correct / len(rows)
        row_s = {
            "verifier_id": vid,
            "model_name": res["model_name"],
            "accuracy_4class": round(acc, 4),
            "rec_supported": round(per_class_ok["supported"] / per_class_tot["supported"], 4),
            "rec_contradicted": round(per_class_ok["contradicted"] / per_class_tot["contradicted"], 4),
            "rec_partial": round(per_class_ok["partially_supported"] / per_class_tot["partially_supported"], 4),
            "rec_insufficient": round(per_class_ok["insufficient_evidence"] / per_class_tot["insufficient_evidence"], 4),
            "inference_seconds": res["inference_seconds"],
            "throughput_pairs_per_s": res["throughput_pairs_per_s"],
            "peak_vram_gb": res["peak_vram_gb"],
        }
        summary_rows.append(row_s)
        print(
            f"[SUMMARY] {vid:18s} | 4-class Acc={acc*100:.2f}% | "
            f"Sup={row_s['rec_supported']*100:.1f}% Con={row_s['rec_contradicted']*100:.1f}% "
            f"Part={row_s['rec_partial']*100:.1f}% Ins={row_s['rec_insufficient']*100:.1f}% | "
            f"Time={res['inference_seconds']:.1f}s VRAM={res['peak_vram_gb']:.2f}GB",
            flush=True,
        )

    (OUT_DIR / "verifiers_run_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    print("\n=== ALL 4 REAL HETEROGENEOUS VERIFIERS COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
