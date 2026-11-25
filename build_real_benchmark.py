"""Construct DeCiv-Real-1200: a 4-domain, 1,200-packet real full-text benchmark.

Domains (75 parent clusters x 4 variants = 300 packets per domain; 1,200 total):
  1. FEVER      : Real Wikipedia claims + full gold Wikipedia sentences (copenlu/fever_gold_evidence).
  2. SciFact    : Real biomedical claims + full PubMed abstracts (allenai/scifact).
  3. HotpotQA   : Real multi-hop factual QA + full Wikipedia supporting paragraphs.
  4. TruthfulQA : Real adversarial misconception QA + factual answers & real sources.

Also exports 150 real open-ended RAG QA queries (75 HotpotQA + 75 TruthfulQA),
each carrying exact (qa_question, gold_answer, poisoned_answer) for the
end-to-end dense RAG contamination defense benchmark.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw_sources"
OUT_JSONL = ROOT / "data" / "deciv_real_1200.jsonl"
OUT_QA_JSONL = ROOT / "data" / "deciv_real_rag_qa_150.jsonl"
HARD_1800 = ROOT.parent / "others" / "build_engine" / "data" / "derived" / "deciv_hard_1800.csv"

FORBIDDEN_LEAK_TOKENS = (
    "fever label",
    "not enough info",
    "gold evidence",
    "dataset supplies",
    "hotpotqa supporting facts",
    "truthfulqa best answer",
    "omitted_by_control",
    "conflict_control",
)


def clean_wiki_text(text: str) -> str:
    text = text.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]")
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_partially_supported_claim(true_claim: str, domain: str, idx: int) -> tuple[str, str]:
    """Deterministically build a fluent partially_supported claim from a verified true claim.

    Strategy (zero verifier-LLM involvement to avoid circularity):
      1. If a 4-digit year (1700-2023) appears in FEVER/HotpotQA, shift it by +/-3 years
         (localized temporal inaccuracy while the core proposition remains supported).
      2. Else if a multi-digit number or percentage appears in FEVER/SciFact, perturb that
         single numeric figure while keeping the primary entity/relationship intact.
      3. Else append one of 16 diverse, domain-tailored unverified secondary attribution/scope
         clauses so the core proposition is directly entailed by primary evidence while the
         appended secondary detail is unsupported (locally repairable against primary evidence).
    """
    m_year = re.search(r"\b(1[789]\d\d|20[01]\d|202[0-3])\b", true_claim)
    if m_year and domain in ("FEVER", "HotpotQA"):
        orig_y = int(m_year.group(1))
        shifted_y = orig_y + (3 if (idx % 2 == 0) else -3)
        corrupted = true_claim[: m_year.start()] + str(shifted_y) + true_claim[m_year.end() :]
        return corrupted, "localized_year_shift"

    m_pct = re.search(r"\b(\d+(?:\.\d+)?)%", true_claim)
    if m_pct and domain in ("SciFact", "FEVER"):
        val = float(m_pct.group(1))
        new_val = round(val + 15.0, 1) if val < 70 else round(max(5.0, val - 15.0), 1)
        rep = f"{int(new_val)}%" if new_val.is_integer() else f"{new_val}%"
        corrupted = true_claim[: m_pct.start()] + rep + true_claim[m_pct.end() :]
        return corrupted, "localized_percent_shift"

    base = true_claim.rstrip(". ")
    if domain == "SciFact":
        clauses = [
            ", and this effect was formally replicated in a mandatory 2022 multicenter phase-IV registry across 14 European hospitals",
            ", and subsequent 2021 World Health Organization guidelines mandated routine population-wide screening based on this result",
            ", and follow-up telemetry confirmed the exact same magnitude exclusively in pediatric cohorts under age six",
            ", and a 2020 meta-analysis of 48,000 patients reported zero biological variance across all ethnic subgroups",
            ", and regulatory agencies in twelve countries codified this threshold as a binding clinical diagnostic standard in 2019",
            ", and parallel veterinary trials in 2018 demonstrated identical quantitative kinetics across all mammalian species",
            ", and the finding was restricted exclusively to subjects receiving high-dose intravenous supplementation for 90 days",
            ", and an archival 2023 audit established that the mechanism completely reverses under standard ambient humidity",
        ]
    else:
        clauses = [
            ", and this fact was officially commemorated in a 2019 UNESCO archival registry",
            ", and the original documentation was exclusively preserved in the Zurich Federal Archive in 2016",
            ", and a 2021 parliamentary commission formally revised the official historical record on this point",
            ", and contemporary accounts note it occurred simultaneously across seven neighboring jurisdictions",
            ", and the underlying citation was later codified in an international 2018 standardization charter",
            ", and public census bulletins from 2020 attributed the outcome solely to a municipal referendum",
            ", and a 2017 retrospective exhibition in Vienna displayed the sole surviving manuscript confirming it",
            ", and official trade statistics from 2015 further linked the event to a bilateral maritime treaty",
        ]
    clause = clauses[idx % len(clauses)]
    return base + clause + ".", "unverified_secondary_clause"


def make_gov_context(variant: str, idx: int) -> dict[str, str]:
    if variant == "v0_clean_verified" or variant == "v0_clean_nei":
        return {
            "provenance_status": "complete",
            "evidence_auditability": "auditable",
            "human_review_required": "false",
            "repairability": "no_repair",
        }
    if variant in ("v1_provenance_dropout", "v1_nei_provenance_dropout"):
        mode = idx % 3
        return {
            "provenance_status": "incomplete" if mode == 0 else "complete",
            "evidence_auditability": "unauditable" if mode == 1 else "auditable",
            "human_review_required": "true" if mode == 2 else "false",
            "repairability": "no_repair",
        }
    if variant == "v2_localized_repairable":
        return {
            "provenance_status": "complete",
            "evidence_auditability": "auditable",
            "human_review_required": "false",
            "repairability": "verified_repair",
        }
    if variant == "v2_partial_unrepairable":
        return {
            "provenance_status": "complete",
            "evidence_auditability": "auditable",
            "human_review_required": "false",
            "repairability": "no_repair",
        }
    # v3_adversarial_contradiction
    return {
        "provenance_status": "complete",
        "evidence_auditability": "auditable",
        "human_review_required": "false",
        "repairability": "no_repair",
    }


def reference_action(status: str, ctx: dict[str, str]) -> str:
    accept_ok = (
        ctx["provenance_status"] == "complete"
        and ctx["evidence_auditability"] == "auditable"
        and ctx["human_review_required"] != "true"
    )
    if status == "supported":
        return "accept" if accept_ok else "defer"
    if status == "contradicted":
        return "reject"
    if status == "partially_supported" and ctx["repairability"] == "verified_repair":
        return "revise"
    return "defer"


def build_fever_domain() -> list[dict]:
    """75 FEVER parents (50 supported + 25 NEI) x 4 variants = 300 packets."""
    rows = []
    with (RAW / "fever_valid.jsonl").open(encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]

    def extract_ev(r):
        ev = r.get("evidence", [])
        sents = []
        seen = set()
        for item in ev:
            if isinstance(item, list) and len(item) >= 3 and isinstance(item[2], str):
                s = clean_wiki_text(item[2])
                if s and s not in seen and len(s) >= 25:
                    seen.add(s)
                    sents.append(s)
        return " ".join(sents[:3])

    sup_pool, ref_pool, nei_pool = [], [], []
    seen_claims = set()
    for r in raw:
        c = clean_wiki_text(r.get("claim", ""))
        ev = extract_ev(r)
        if not c or not ev or c in seen_claims or len(c) < 18 or len(ev) < 35:
            continue
        if any(tok in ev.lower() or tok in c.lower() for tok in FORBIDDEN_LEAK_TOKENS):
            continue
        seen_claims.add(c)
        lbl = r.get("label")
        rec = {"id": r["id"], "claim": c, "evidence": ev[:900], "url": f"https://en.wikipedia.org/wiki/FEVER_{r['id']}"}
        if lbl == "SUPPORTS":
            sup_pool.append(rec)
        elif lbl == "REFUTES":
            ref_pool.append(rec)
        elif lbl == "NOT ENOUGH INFO":
            nei_pool.append(rec)

    assert len(sup_pool) >= 50 and len(ref_pool) >= 75 and len(nei_pool) >= 25

    for p_idx in range(75):
        pid = f"FEV_R_{p_idx+1:03d}"
        ref_item = ref_pool[p_idx]
        if p_idx < 50:
            base = sup_pool[p_idx]
            v2_claim, v2_rule = make_partially_supported_claim(base["claim"], "FEVER", p_idx)
            variants = [
                ("v0_clean_verified", base["claim"], base["evidence"], "supported", base["claim"], "intact_supported"),
                ("v1_provenance_dropout", base["claim"], base["evidence"], "supported", base["claim"], "provenance_dropout"),
                ("v2_localized_repairable", v2_claim, base["evidence"], "partially_supported", base["claim"], v2_rule),
                ("v3_adversarial_contradiction", ref_item["claim"], ref_item["evidence"], "contradicted", ref_item["claim"], "natural_fever_refutes"),
            ]
        else:
            nei = nei_pool[p_idx - 50]
            sup_helper = sup_pool[50 + (p_idx - 50)]
            v2_claim, v2_rule = make_partially_supported_claim(sup_helper["claim"], "FEVER", p_idx)
            variants = [
                ("v0_clean_nei", nei["claim"], nei["evidence"], "insufficient_evidence", nei["claim"], "natural_fever_nei"),
                ("v1_nei_provenance_dropout", nei["claim"], nei["evidence"], "insufficient_evidence", nei["claim"], "nei_provenance_dropout"),
                ("v2_partial_unrepairable", v2_claim, sup_helper["evidence"], "partially_supported", sup_helper["claim"], "partial_unrepairable"),
                ("v3_adversarial_contradiction", ref_item["claim"], ref_item["evidence"], "contradicted", ref_item["claim"], "natural_fever_refutes"),
            ]

        for vname, clm, ev, status, gold_clean_clm, rule in variants:
            ctx = make_gov_context(vname, p_idx)
            rows.append(
                {
                    "packet_id": f"{pid}__{vname[:2]}",
                    "parent_id": pid,
                    "source_dataset": "FEVER",
                    "variant_name": vname,
                    "corruption_rule": rule,
                    "claim_text": clm,
                    "evidence_text": ev,
                    "gold_clean_claim": gold_clean_clm,
                    "source_url": base["url"] if p_idx < 50 else nei["url"],
                    "semantic_status": status,
                    "reference_action": reference_action(status, ctx),
                    **ctx,
                }
            )
    return rows


def build_scifact_domain() -> list[dict]:
    """75 SciFact parents (50 supported + 25 NEI) x 4 variants = 300 packets."""
    sf_dir = RAW / "scifact" / "data"
    corpus = {}
    with (sf_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            title = clean_wiki_text(d.get("title", ""))
            sents = [clean_wiki_text(s) for s in d.get("abstract", [])]
            corpus[int(d["doc_id"])] = {"title": title, "sents": sents}

    claims_all = []
    for fname in ("claims_dev.jsonl", "claims_train.jsonl"):
        with (sf_dir / fname).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    claims_all.append(json.loads(line))

    sup_pool, con_pool, nei_pool = [], [], []
    seen = set()
    for r in claims_all:
        clm = clean_wiki_text(r["claim"])
        if clm in seen or len(clm) < 20:
            continue
        ev_map = r.get("evidence", {})
        if not ev_map:
            cids = r.get("cited_doc_ids", [])
            if not cids or int(cids[0]) not in corpus:
                continue
            doc = corpus[int(cids[0])]
            ev_txt = f"{doc['title']}. " + " ".join(doc["sents"][:5])
            if len(ev_txt) < 60:
                continue
            seen.add(clm)
            nei_pool.append({"id": r["id"], "claim": clm, "evidence": ev_txt[:1000], "url": f"https://pubmed.ncbi.nlm.nih.gov/{cids[0]}/"})
        else:
            doc_id_str, rat_list = next(iter(ev_map.items()))
            doc_id = int(doc_id_str)
            if doc_id not in corpus or not rat_list:
                continue
            lbl = rat_list[0]["label"]
            sent_ids = sorted({si for entry in rat_list for si in entry.get("sentences", [])})
            doc = corpus[doc_id]
            # Build focused abstract excerpt containing title + rationale sentences + 1 neighbor
            keep = set(sent_ids)
            for si in list(sent_ids):
                if si - 1 >= 0:
                    keep.add(si - 1)
                if si + 1 < len(doc["sents"]):
                    keep.add(si + 1)
            chosen = [doc["sents"][i] for i in sorted(keep) if 0 <= i < len(doc["sents"])]
            ev_txt = f"{doc['title']}. " + " ".join(chosen)
            if len(ev_txt) < 60:
                continue
            seen.add(clm)
            rec = {"id": r["id"], "claim": clm, "evidence": ev_txt[:1000], "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/"}
            if lbl == "SUPPORT":
                sup_pool.append(rec)
            elif lbl == "CONTRADICT":
                con_pool.append(rec)

    assert len(sup_pool) >= 75 and len(con_pool) >= 75 and len(nei_pool) >= 25
    rows = []
    for p_idx in range(75):
        pid = f"SCI_R_{p_idx+1:03d}"
        con_item = con_pool[p_idx]
        if p_idx < 50:
            base = sup_pool[p_idx]
            v2_claim, v2_rule = make_partially_supported_claim(base["claim"], "SciFact", p_idx)
            variants = [
                ("v0_clean_verified", base["claim"], base["evidence"], "supported", base["claim"], "intact_supported"),
                ("v1_provenance_dropout", base["claim"], base["evidence"], "supported", base["claim"], "provenance_dropout"),
                ("v2_localized_repairable", v2_claim, base["evidence"], "partially_supported", base["claim"], v2_rule),
                ("v3_adversarial_contradiction", con_item["claim"], con_item["evidence"], "contradicted", con_item["claim"], "natural_scifact_contradict"),
            ]
        else:
            nei = nei_pool[p_idx - 50]
            sup_helper = sup_pool[50 + (p_idx - 50)]
            v2_claim, v2_rule = make_partially_supported_claim(sup_helper["claim"], "SciFact", p_idx)
            variants = [
                ("v0_clean_nei", nei["claim"], nei["evidence"], "insufficient_evidence", nei["claim"], "natural_scifact_nei"),
                ("v1_nei_provenance_dropout", nei["claim"], nei["evidence"], "insufficient_evidence", nei["claim"], "nei_provenance_dropout"),
                ("v2_partial_unrepairable", v2_claim, sup_helper["evidence"], "partially_supported", sup_helper["claim"], "partial_unrepairable"),
                ("v3_adversarial_contradiction", con_item["claim"], con_item["evidence"], "contradicted", con_item["claim"], "natural_scifact_contradict"),
            ]

        for vname, clm, ev, status, gold_clean_clm, rule in variants:
            ctx = make_gov_context(vname, p_idx)
            rows.append(
                {
                    "packet_id": f"{pid}__{vname[:2]}",
                    "parent_id": pid,
                    "source_dataset": "SciFact",
                    "variant_name": vname,
                    "corruption_rule": rule,
                    "claim_text": clm,
                    "evidence_text": ev,
                    "gold_clean_claim": gold_clean_clm,
                    "source_url": base["url"] if p_idx < 50 else nei["url"],
                    "semantic_status": status,
                    "reference_action": reference_action(status, ctx),
                    **ctx,
                }
            )
    return rows


def _extract_hpq_distractor(evidence: str, gold_answer: str, question: str) -> str:
    """Extract a realistic same-passage distractor answer from HotpotQA supporting text."""
    # 1. Look for quoted titles/entities different from gold_answer
    quoted = re.findall(r'"([^"]{3,45})"', evidence)
    for q in quoted:
        if q.lower() != gold_answer.lower() and q.lower() not in question.lower() and gold_answer.lower() not in q.lower():
            return q
    # 2. If gold answer is a 4-digit year, shift by +4 years
    if re.fullmatch(r"\d{4}", gold_answer.strip()):
        return str(int(gold_answer.strip()) + 4)
    # 3. If gold answer is yes/no, flip it
    if gold_answer.strip().lower() == "yes":
        return "no"
    if gold_answer.strip().lower() == "no":
        return "yes"
    # 4. Look for capitalized multi-word proper nouns in evidence not in gold_answer or question
    prop_nouns = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", evidence)
    for pn in prop_nouns:
        if (
            pn.lower() != gold_answer.lower()
            and pn.lower() not in question.lower()
            and gold_answer.lower() not in pn.lower()
            and pn.lower() not in ("united states", "new york")
        ):
            return pn
    # 5. If a year exists in gold_answer, shift it
    m_y = re.search(r"\b(1[789]\d\d|20[01]\d)\b", gold_answer)
    if m_y:
        return gold_answer[: m_y.start()] + str(int(m_y.group(1)) + 5) + gold_answer[m_y.end() :]
    return "an unrelated entity not mentioned in the record"


def build_hotpotqa_domain() -> tuple[list[dict], list[dict]]:
    """75 HotpotQA parents (50 supported + 25 held-out-hop NEI) x 4 variants = 300 packets + 75 QA items."""
    with HARD_1800.open(encoding="utf-8-sig") as f:
        hpq_v0 = [
            r
            for r in csv.DictReader(f)
            if r["original_source_dataset"] == "HotpotQA" and r["variant_name"] == "v0_intact_control"
        ]
    assert len(hpq_v0) >= 75

    rows = []
    qa_items = []
    pat = re.compile(r'The answer to the multi-hop question "(.+)" is "(.+)"\.?')

    for p_idx in range(75):
        r = hpq_v0[p_idx]
        pid = f"HPQ_R_{p_idx+1:03d}"
        m = pat.search(r["claim_text"])
        assert m, f"Could not parse HotpotQA claim: {r['claim_text']}"
        question = m.group(1).strip()
        gold_ans = m.group(2).strip()
        ev_full = re.sub(r"^HotpotQA supporting facts:\s*", "", r["evidence_text"]).strip()
        ev_full = clean_wiki_text(ev_full)[:1100]
        clean_claim = f'For the question "{question}", the verified answer is "{gold_ans}".'
        poison_ans = _extract_hpq_distractor(ev_full, gold_ans, question)
        if poison_ans.startswith("an unrelated"):
            contradict_claim = f'For the question "{question}", the answer is NOT "{gold_ans}".'
            poison_ans = f"not {gold_ans}"
        else:
            contradict_claim = f'For the question "{question}", the verified answer is "{poison_ans}".'

        # Split two-hop evidence for NEI (keep the hop that does NOT contain gold_ans!)
        hops = [h.strip() for h in ev_full.split(" | ") if h.strip()]
        nei_ev = hops[0]
        if len(hops) >= 2:
            no_ans_hops = [h for h in hops if gold_ans.lower() not in h.lower()]
            if no_ans_hops:
                nei_ev = no_ans_hops[0]
            else:
                # Mask the second hop sentence containing the answer
                nei_ev = hops[0].split(". ")[0] + "."

        qa_items.append(
            {
                "qa_id": pid,
                "parent_id": pid,
                "source_dataset": "HotpotQA",
                "has_supported_v0": p_idx < 50,
                "question": question,
                "gold_answer": gold_ans,
                "poisoned_answer": poison_ans,
                "clean_claim": clean_claim,
                "poisoned_claim": contradict_claim,
            }
        )

        v2_claim, v2_rule = make_partially_supported_claim(clean_claim, "HotpotQA", p_idx)
        if p_idx < 50:
            variants = [
                ("v0_clean_verified", clean_claim, ev_full, "supported", clean_claim, "intact_supported"),
                ("v1_provenance_dropout", clean_claim, ev_full, "supported", clean_claim, "provenance_dropout"),
                ("v2_localized_repairable", v2_claim, ev_full, "partially_supported", clean_claim, v2_rule),
                ("v3_adversarial_contradiction", contradict_claim, ev_full, "contradicted", clean_claim, "in_passage_entity_poisoning"),
            ]
        else:
            variants = [
                ("v0_clean_nei", clean_claim, nei_ev, "insufficient_evidence", clean_claim, "missing_second_hop_evidence"),
                ("v1_nei_provenance_dropout", clean_claim, nei_ev, "insufficient_evidence", clean_claim, "nei_provenance_dropout"),
                ("v2_partial_unrepairable", v2_claim, ev_full, "partially_supported", clean_claim, "partial_unrepairable"),
                ("v3_adversarial_contradiction", contradict_claim, ev_full, "contradicted", clean_claim, "in_passage_entity_poisoning"),
            ]

        for vname, clm, ev, status, gold_clean_clm, rule in variants:
            ctx = make_gov_context(vname, p_idx)
            rows.append(
                {
                    "packet_id": f"{pid}__{vname[:2]}",
                    "parent_id": pid,
                    "source_dataset": "HotpotQA",
                    "variant_name": vname,
                    "corruption_rule": rule,
                    "claim_text": clm,
                    "evidence_text": ev,
                    "gold_clean_claim": gold_clean_clm,
                    "source_url": r["source_url"],
                    "semantic_status": status,
                    "reference_action": reference_action(status, ctx),
                    **ctx,
                }
            )
    return rows, qa_items


def build_truthfulqa_domain() -> tuple[list[dict], list[dict]]:
    """75 TruthfulQA parents (50 supported + 25 off-topic/insufficient NEI) x 4 variants = 300 pkts + 75 QA items."""
    with (RAW / "TruthfulQA.csv").open(encoding="utf-8-sig") as f:
        tqa = list(csv.DictReader(f))

    # Keep rich entries with non-empty Best Answer, Best Incorrect Answer, and Source URL
    filtered = []
    for r in tqa:
        q = r["Question"].strip()
        best = r["Best Answer"].strip()
        wrong = r["Best Incorrect Answer"].strip()
        corr = r["Correct Answers"].strip()
        src = r["Source"].strip()
        if (
            len(q) >= 15
            and len(best) >= 15
            and len(wrong) >= 15
            and best.lower() != wrong.lower()
            and "i have no comment" not in best.lower()
            and "i have no comment" not in wrong.lower()
            and src.startswith("http")
        ):
            filtered.append(r)

    assert len(filtered) >= 120
    rows = []
    qa_items = []
    for p_idx in range(75):
        r = filtered[p_idx]
        pid = f"TQA_R_{p_idx+1:03d}"
        q = r["Question"].strip()
        best = r["Best Answer"].strip().rstrip(".")
        wrong = r["Best Incorrect Answer"].strip().rstrip(".")
        corr_list = [c.strip().rstrip(".") for c in r["Correct Answers"].split(";") if c.strip() and "i have no comment" not in c.lower()]
        ev_full = f'Reference facts regarding "{q}": {best}.'
        if len(corr_list) > 1:
            extra = [c for c in corr_list if c.lower() != best.lower()][:2]
            if extra:
                ev_full += " Additional verified notes: " + "; ".join(extra) + "."
        clean_claim = f'Regarding "{q}", {best}.'
        contradict_claim = f'Regarding "{q}", {wrong}.'

        # For NEI parents (50..74), pair the claim with an unrelated topic's reference passage from filtered[90+...]
        off_r = filtered[85 + (p_idx - 50)] if p_idx >= 50 else filtered[85]
        nei_ev = f'Reference note on {off_r["Category"]}: {off_r["Best Answer"].strip().rstrip(".")}.'

        qa_items.append(
            {
                "qa_id": pid,
                "parent_id": pid,
                "source_dataset": "TruthfulQA",
                "has_supported_v0": p_idx < 50,
                "question": q,
                "gold_answer": best,
                "poisoned_answer": wrong,
                "clean_claim": clean_claim,
                "poisoned_claim": contradict_claim,
            }
        )

        v2_claim, v2_rule = make_partially_supported_claim(clean_claim, "TruthfulQA", p_idx)
        if p_idx < 50:
            variants = [
                ("v0_clean_verified", clean_claim, ev_full, "supported", clean_claim, "intact_supported"),
                ("v1_provenance_dropout", clean_claim, ev_full, "supported", clean_claim, "provenance_dropout"),
                ("v2_localized_repairable", v2_claim, ev_full, "partially_supported", clean_claim, v2_rule),
                ("v3_adversarial_contradiction", contradict_claim, ev_full, "contradicted", clean_claim, "natural_truthfulqa_misconception"),
            ]
        else:
            variants = [
                ("v0_clean_nei", clean_claim, nei_ev, "insufficient_evidence", clean_claim, "non_entailing_reference_note"),
                ("v1_nei_provenance_dropout", clean_claim, nei_ev, "insufficient_evidence", clean_claim, "nei_provenance_dropout"),
                ("v2_partial_unrepairable", v2_claim, ev_full, "partially_supported", clean_claim, "partial_unrepairable"),
                ("v3_adversarial_contradiction", contradict_claim, ev_full, "contradicted", clean_claim, "natural_truthfulqa_misconception"),
            ]

        for vname, clm, ev, status, gold_clean_clm, rule in variants:
            ctx = make_gov_context(vname, p_idx)
            rows.append(
                {
                    "packet_id": f"{pid}__{vname[:2]}",
                    "parent_id": pid,
                    "source_dataset": "TruthfulQA",
                    "variant_name": vname,
                    "corruption_rule": rule,
                    "claim_text": clm,
                    "evidence_text": ev,
                    "gold_clean_claim": gold_clean_clm,
                    "source_url": r["Source"].strip(),
                    "semantic_status": status,
                    "reference_action": reference_action(status, ctx),
                    **ctx,
                }
            )
    return rows, qa_items


def main():
    fever = build_fever_domain()
    scifact = build_scifact_domain()
    hpq, hpq_qa = build_hotpotqa_domain()
    tqa, tqa_qa = build_truthfulqa_domain()

    all_rows = fever + scifact + hpq + tqa
    all_qa = hpq_qa + tqa_qa

    assert len(all_rows) == 1200
    assert len({r["packet_id"] for r in all_rows}) == 1200
    assert len({r["parent_id"] for r in all_rows}) == 300

    # Strict leak-token verification across ALL 1,200 packets
    for r in all_rows:
        combined = (r["claim_text"] + " " + r["evidence_text"]).lower()
        for tok in FORBIDDEN_LEAK_TOKENS:
            assert tok not in combined, f"Leak token '{tok}' found in {r['packet_id']}"

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with OUT_QA_JSONL.open("w", encoding="utf-8") as f:
        for q in all_qa:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Built {OUT_JSONL.name}: n={len(all_rows)}, parents={len({r['parent_id'] for r in all_rows})}")
    print("  Domains:", dict(Counter(r["source_dataset"] for r in all_rows)))
    print("  Semantic statuses:", dict(Counter(r["semantic_status"] for r in all_rows)))
    print("  Reference actions:", dict(Counter(r["reference_action"] for r in all_rows)))
    print(f"Built {OUT_QA_JSONL.name}: n={len(all_qa)} QA queries")

    for dom in ("FEVER", "SciFact", "HotpotQA", "TruthfulQA"):
        sample = [r for r in all_rows if r["source_dataset"] == dom][:4]
        print(f"\n=== SAMPLE {dom} PARENT ({sample[0]['parent_id']}) ===")
        for s in sample:
            print(f"  [{s['variant_name']:28s}] status={s['semantic_status']:21s} ref={s['reference_action']:6s}")
            print(f"    CLM: {s['claim_text'][:150]}")
            print(f"    EV : {s['evidence_text'][:150]}")


if __name__ == "__main__":
    main()
