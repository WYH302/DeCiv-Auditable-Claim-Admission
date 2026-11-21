"""Execute the shared-input admission study; Python 3.11+ and NumPy required.

Run from any directory: python reproducibility/run_comparison.py
The supplied votes are synthetic, conditioned on rubric semantic targets.
No method receives a test target, transformation name, ID, or proxy score.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from deciv2.r12.contracts import SemanticEstimatorInput, SemanticObservations, SemanticVote
from deciv2.r12.semantic import SemanticDawidSkeneEstimator

STATUSES = ("supported", "contradicted", "partially_supported", "insufficient_evidence")
ACTIONS = ("accept", "reject", "revise", "defer")
ESTIMATORS = ("majority", "reliability_weighted", "dawid_skene", "fixed", "adaptive")
TIE_ORDER = ("defer", "reject", "revise", "accept")
BASE_COST = np.array([[0, 18, 13.5, 18], [4, 0, 5, 6], [3, 4, 0, 2], [.5, .5, .5, 0]], dtype=float)
SPEC = {
    "version": "shared-admission-20261025-v1", "bootstrap_seed": 20261025,
    "bootstrap_iterations": 2000, "fixed_weights": [1, 1, 1, 1],
    "reliability": "(training_correct_votes + 1)/(training_votes + 2)",
    "adaptive": "training_reliability*(0.5+0.5*other_validator_agreement)",
    "adaptive_strength": .5, "main_defer_cost": .5,
    "sensitivity_defer_costs": [.1, .5, 1., 2.],
    "tie_order": list(TIE_ORDER), "cost_matrix": BASE_COST.tolist(),
    "split": "leave_one_source_out; all four variants of a parent stay together",
    "reference": "semantic status plus admissibility; never supplied to prediction",
    "excluded_prediction_fields": ["semantic_status", "rubric_admission_action", "semantic_observations_json", "variant_name", "corruption_family", "audit_join_id", "challenge_id", "parent_id", "source_dataset", "fold_id"],
    "uncertainty": "paired source-stratified parent-cluster percentile bootstrap conditional on frozen votes and fitted fold states",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_rows(name):
    with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Empty output: {name}")
    with (ROOT / "results" / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def eligible(context):
    reasons = []
    if context["provenance_status"] != "complete":
        reasons.append("incomplete_provenance")
    if context["evidence_auditability"] != "auditable":
        reasons.append("unauditable_evidence")
    if context["human_review_required"] == "true":
        reasons.append("human_review_required")
    allowed = {"reject", "defer"}
    if not reasons:
        allowed.add("accept")
    if context["repairability"] == "verified_repair":
        allowed.add("revise")
    return allowed, reasons


def reference_action(status, context):
    """Evaluation-only policy reference, independent of estimator outputs."""
    allowed, _ = eligible(context)
    if status == "supported":
        return "accept" if "accept" in allowed else "defer"
    if status == "contradicted":
        return "reject"
    if status == "partially_supported" and "revise" in allowed:
        return "revise"
    return "defer"


def aggregate(votes, reliability, mode, strength=.5):
    """Prediction boundary: only four semantic votes and fitted weights."""
    if mode in ("majority", "fixed"):
        weights = [1.] * len(votes)
    else:
        weights = [reliability[v.validator_id] for v in votes]
        if mode == "adaptive":
            weights = [weight * ((1-strength) + strength * sum(other.status == vote.status for k, other in enumerate(votes) if k != j) / max(1, len(votes)-1)) for j, (vote, weight) in enumerate(zip(votes, weights))]
    posterior = [sum(weight for vote, weight in zip(votes, weights) if vote.status == status) / sum(weights) for status in STATUSES]
    assert abs(sum(posterior)-1.) < 1e-12
    return posterior


def route(posterior, context, router, defer_cost=.5):
    costs = BASE_COST.copy()
    costs[3, :3] = defer_cost
    risks = costs @ np.asarray(posterior)
    allowed, reasons = eligible(context)
    if router == "direct":
        maximum = max(posterior)
        winners = [ACTIONS[j] for j, value in enumerate(posterior) if abs(value-maximum) <= 1e-12]
        action = winners[0] if len(winners) == 1 else "defer"
        reason = "semantic_argmax" if len(winners) == 1 else "semantic_tie_defer"
    else:
        best = min(risks[ACTIONS.index(action)] for action in allowed)
        action = next(action for action in TIE_ORDER if action in allowed and risks[ACTIONS.index(action)] <= best+1e-12)
        reason = "minimum_expected_cost_among_eligible_actions"
    return action, risks.tolist(), reasons, reason, action in allowed


def load_study():
    index = read_rows("evaluation_index_v2.csv")
    assert len(index) == 1800 and len({r["parent_id"] for r in index}) == 450
    assert len({r["audit_join_id"] for r in index}) == len(index)
    targets = {r["audit_join_id"]: r["semantic_status"] for r in read_rows("semantic_targets_v2.csv")}
    contexts = {r["audit_join_id"]: {k: v for k, v in r.items() if k != "audit_join_id"} for r in read_rows("governance_contexts_v2.csv")}
    votes = defaultdict(list)
    for row in read_rows("semantic_votes_v2.csv"):
        assert row["signal_origin"] == "controlled_semantic_vote_from_rubric_target"
        votes[row["audit_join_id"]].append(SemanticVote(row["validator_id"], row["voted_status"], None))
    zero = SemanticObservations(0., 0., 0., 0.)
    inputs = {}
    for row in index:
        key = row["audit_join_id"]
        assert len(votes[key]) == 4 and len({v.validator_id for v in votes[key]}) == 4
        inputs[key] = SemanticEstimatorInput(zero, tuple(votes[key]))
    assert set(inputs) == set(contexts) == set(targets)
    for parent in {r["parent_id"] for r in index}:
        assert len({r["source_dataset"] for r in index if r["parent_id"] == parent}) == 1
    return index, targets, contexts, inputs


def fit_folds(index, targets, inputs):
    predictions = {name: {} for name in ESTIMATORS}
    states = []
    for source in sorted({r["source_dataset"] for r in index}):
        training = [r for r in index if r["source_dataset"] != source]
        testing = [r for r in index if r["source_dataset"] == source]
        assert not ({r["parent_id"] for r in training} & {r["parent_id"] for r in testing})
        train_inputs = tuple(inputs[r["audit_join_id"]] for r in training)
        test_inputs = tuple(inputs[r["audit_join_id"]] for r in testing)
        successes, totals = Counter(), Counter()
        for row in training:
            for vote in inputs[row["audit_join_id"]].semantic_votes:
                totals[vote.validator_id] += 1
                successes[vote.validator_id] += vote.status == targets[row["audit_join_id"]]
        reliability = {validator: (successes[validator]+1)/(totals[validator]+2) for validator in sorted(totals)}
        ds = SemanticDawidSkeneEstimator().fit(train_inputs)
        ds_outputs = ds.predict_proba(test_inputs)
        state = {"held_out_source": source, "n_train": len(training), "n_test": len(testing), "reliability": reliability, "dawid_skene": json.loads(ds.state_bytes()), "ds_converged": ds.converged, "ds_iterations": ds.iterations}
        states.append(state)
        for row, ds_output in zip(testing, ds_outputs):
            key = row["audit_join_id"]
            for mode in ESTIMATORS:
                posterior = [ds_output.probability(status) for status in STATUSES] if mode == "dawid_skene" else aggregate(inputs[key].semantic_votes, reliability, mode)
                predictions[mode][key] = posterior
        print(f"Fitted leave-out {source}: train={len(training)}, test={len(testing)}, DS iterations={ds.iterations}, converged={ds.converged}", flush=True)
    return predictions, states


def result_rows(index, targets, contexts, inputs, predictions):
    output = []
    for mode in ESTIMATORS:
        for router in ("direct", "risk"):
            for row in index:
                key = row["audit_join_id"]
                posterior = predictions[mode][key]
                action, risks, reasons, reason, admissible = route(posterior, contexts[key], router)
                reference = reference_action(targets[key], contexts[key])
                output.append({"method": mode, "router": router, "audit_join_id": key, "parent_id": row["parent_id"], "held_out_source": row["source_dataset"], "reference_action": reference, "semantic_target": targets[key], "predicted_action": action, "eligible": int(admissible), "correct": int(action == reference), "false_accept": int(action == "accept" and reference != "accept"), "p_supported": posterior[0], "p_contradicted": posterior[1], "p_partial": posterior[2], "p_insufficient": posterior[3], "risk_accept": risks[0], "risk_reject": risks[1], "risk_revise": risks[2], "risk_defer": risks[3], "reason": reason, "eligibility_reasons": "|".join(reasons), "runtime_input_hash": digest({"votes": [v.payload() for v in inputs[key].semantic_votes], "context": contexts[key]})})
    return output


def summarize(rows):
    n = len(rows)
    accepted = sum(r["predicted_action"] == "accept" for r in rows)
    good_accept = sum(r["predicted_action"] == r["reference_action"] == "accept" for r in rows)
    nonacceptable = sum(r["reference_action"] != "accept" for r in rows)
    output = {"n": n, "accuracy": sum(r["correct"] for r in rows)/n,
              "far": sum(r["false_accept"] for r in rows)/nonacceptable,
              "accept_precision": good_accept/accepted if accepted else None,
              "acceptance_rate": accepted/n, "non_defer_fraction": sum(r["predicted_action"] != "defer" for r in rows)/n,
              "eligibility_violations": sum(not r["eligible"] for r in rows)}
    for action in ACTIONS:
        count = sum(r["reference_action"] == action for r in rows)
        output[f"recall_{action}"] = sum(r["predicted_action"] == r["reference_action"] == action for r in rows)/count if count else None
        output[f"predicted_{action}"] = sum(r["predicted_action"] == action for r in rows)
    return output


def metrics_by_group(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["router"])].append(row)
    return groups


def paired_bootstrap(index, groups):
    parents = sorted({r["parent_id"] for r in index})
    parent_axis = {p: i for i, p in enumerate(parents)}
    parent_source = {r["parent_id"]: r["source_dataset"] for r in index}
    rng = np.random.default_rng(SPEC["bootstrap_seed"])
    weights = np.zeros((SPEC["bootstrap_iterations"], len(parents)), dtype=float)
    for source in sorted(set(parent_source.values())):
        columns = [parent_axis[p] for p in parents if parent_source[p] == source]
        weights[:, columns] = rng.multinomial(len(columns), np.full(len(columns), 1/len(columns)), size=len(weights))
    samples = {}
    for key, rows in groups.items():
        totals = np.zeros((len(parents), 7))
        for row in rows:
            totals[parent_axis[row["parent_id"]]] += np.array([1, row["correct"], row["reference_action"] != "accept", row["false_accept"], row["predicted_action"] == "accept", row["predicted_action"] == row["reference_action"] == "accept", row["predicted_action"] != "defer"])
        boot = weights @ totals
        samples[key] = {"accuracy": boot[:, 1]/boot[:, 0], "far": boot[:, 3]/boot[:, 2], "acceptance_rate": boot[:, 4]/boot[:, 0], "non_defer_fraction": boot[:, 6]/boot[:, 0]}
    output = []
    comparisons = [(('adaptive','risk'),('fixed','risk')), (('dawid_skene','risk'),('fixed','risk')), (('fixed','risk'),('majority','direct'))]
    for left, right in comparisons:
        for metric in samples[left]:
            diffs = samples[left][metric]-samples[right][metric]
            low, high = np.quantile(diffs, [.025, .975])
            output.append({"left": ":".join(left), "right": ":".join(right), "metric": metric, "difference": summarize(groups[left])[metric]-summarize(groups[right])[metric], "ci_low": low, "ci_high": high, "bootstrap_replicates": len(weights)})
    return output


def save_sensitivity(index, targets, contexts, predictions):
    output = []
    for cost in SPEC["sensitivity_defer_costs"]:
        for mode in ESTIMATORS:
            rows = []
            for row in index:
                key = row["audit_join_id"]
                action, _, _, _, admissible = route(predictions[mode][key], contexts[key], "risk", cost)
                ref = reference_action(targets[key], contexts[key])
                rows.append({"predicted_action": action, "reference_action": ref, "correct": int(action == ref), "false_accept": int(action == "accept" and ref != "accept"), "eligible": admissible})
            output.append({"method": mode, "defer_cost": cost, **summarize(rows)})
    write_csv("cost_sensitivity.csv", output)


def save_five_fold_cv(index, targets, contexts, inputs):
    """Evaluate 5-fold source-stratified parent-cluster cross-validation (n=360/fold, 90 parents/fold)."""
    rng = np.random.default_rng(SPEC["bootstrap_seed"])
    by_src = defaultdict(list)
    for row in index:
        if row["parent_id"] not in by_src[row["source_dataset"]]:
            by_src[row["source_dataset"]].append(row["parent_id"])
    parent_fold = {}
    for src, plist in sorted(by_src.items()):
        plist = sorted(plist)
        perm = rng.permutation(len(plist))
        for idx_pos, p_idx in enumerate(perm):
            parent_fold[plist[p_idx]] = idx_pos % 5

    fold_metrics = defaultdict(list)
    for fold_id in range(5):
        training = [r for r in index if parent_fold[r["parent_id"]] != fold_id]
        testing = [r for r in index if parent_fold[r["parent_id"]] == fold_id]
        assert not ({r["parent_id"] for r in training} & {r["parent_id"] for r in testing})
        train_inputs = tuple(inputs[r["audit_join_id"]] for r in training)
        test_inputs = tuple(inputs[r["audit_join_id"]] for r in testing)
        successes, totals = Counter(), Counter()
        for row in training:
            for vote in inputs[row["audit_join_id"]].semantic_votes:
                totals[vote.validator_id] += 1
                successes[vote.validator_id] += (vote.status == targets[row["audit_join_id"]])
        reliability = {vid: (successes[vid] + 1) / (totals[vid] + 2) for vid in sorted(totals)}
        ds = SemanticDawidSkeneEstimator().fit(train_inputs)
        ds_outputs = ds.predict_proba(test_inputs)
        for mode in ESTIMATORS:
            for router in ("direct", "risk"):
                f_rows = []
                for row, ds_out in zip(testing, ds_outputs):
                    key = row["audit_join_id"]
                    posterior = [ds_out.probability(s) for s in STATUSES] if mode == "dawid_skene" else aggregate(inputs[key].semantic_votes, reliability, mode)
                    action, _, _, _, admissible = route(posterior, contexts[key], router)
                    ref = reference_action(targets[key], contexts[key])
                    f_rows.append({"predicted_action": action, "reference_action": ref, "correct": int(action == ref), "false_accept": int(action == "accept" and ref != "accept"), "eligible": int(admissible)})
                fold_metrics[(mode, router)].append(summarize(f_rows))

    out_rows = []
    for (mode, router), mlist in fold_metrics.items():
        rec = {"method": mode, "router": router, "folds": 5}
        for met in ("accuracy", "far", "accept_precision", "acceptance_rate", "non_defer_fraction", "recall_revise"):
            vals = [float(m[met]) for m in mlist]
            rec[f"{met}_mean"] = float(np.mean(vals))
            rec[f"{met}_std"] = float(np.std(vals, ddof=1))
        rec["eligibility_violations_total"] = int(sum(m["eligibility_violations"] for m in mlist))
        out_rows.append(rec)
    write_csv("five_fold_cv.csv", out_rows)


def save_ablations(index, targets, contexts, predictions):
    """Evaluate systematic component ablations across DeCiv Adaptive and Dawid-Skene routers."""
    configs = [
        ("adaptive_full", "DeCiv Adaptive (Full)", "adaptive", False, False, False, False),
        ("adaptive_no_agreement", "w/o Peer-Agreement Amplification", "fixed", False, False, False, False),
        ("adaptive_no_prov_gate", "w/o Provenance Gate", "adaptive", True, False, False, False),
        ("adaptive_no_repair_gate", "w/o Repairability Gate", "adaptive", False, True, False, False),
        ("adaptive_symmetric_cost", "w/o Asymmetric Cost (0-1 Loss)", "adaptive", False, False, True, False),
        ("adaptive_no_defer", "w/o Quarantine Deferral", "adaptive", False, False, False, True),
        ("ds_full", "DeCiv + Dawid-Skene (Full)", "dawid_skene", False, False, False, False),
        ("ds_no_prov_gate", "w/o Provenance Gate", "dawid_skene", True, False, False, False),
        ("ds_no_repair_gate", "w/o Repairability Gate", "dawid_skene", False, True, False, False),
        ("ds_symmetric_cost", "w/o Asymmetric Cost (0-1 Loss)", "dawid_skene", False, False, True, False),
        ("ds_no_defer", "w/o Quarantine Deferral", "dawid_skene", False, False, False, True),
    ]
    out_rows = []
    for variant_id, label, mode, disable_prov, disable_rep, symmetric_cost, no_defer in configs:
        rows_ab = []
        for row in index:
            key = row["audit_join_id"]
            posterior = predictions[mode][key]
            ctx = dict(contexts[key])
            if disable_prov:
                ctx["provenance_status"] = "complete"
                ctx["evidence_auditability"] = "auditable"
                ctx["human_review_required"] = "false"
            if disable_rep:
                ctx["repairability"] = "verified_repair"
            allowed, _ = eligible(ctx)
            if no_defer and "defer" in allowed:
                allowed.remove("defer")
            costs = (1.0 - np.eye(4)) if symmetric_cost else BASE_COST.copy()
            risks = costs @ np.asarray(posterior)
            best = min(risks[ACTIONS.index(a)] for a in allowed)
            action = next(a for a in TIE_ORDER if a in allowed and risks[ACTIONS.index(a)] <= best + 1e-12)
            true_allowed, _ = eligible(contexts[key])
            ref = reference_action(targets[key], contexts[key])
            rows_ab.append({"predicted_action": action, "reference_action": ref, "correct": int(action == ref), "false_accept": int(action == "accept" and ref != "accept"), "eligible": int(action in true_allowed)})
        out_rows.append({"variant_id": variant_id, "label": label, "base_estimator": mode, **summarize(rows_ab)})
    write_csv("ablation_metrics.csv", out_rows)


def _glad_fit_and_predict(train_inputs, test_inputs, max_iter=60, tol=1e-6):
    """Four-class GLAD item-difficulty & verifier-expertise EM (Whitehill et al., 2009)."""
    k_val = len(train_inputs[0].semantic_votes)
    v_ids = [v.validator_id for v in train_inputs[0].semantic_votes]
    s_idx = {s: idx for idx, s in enumerate(STATUSES)}
    n_tr = len(train_inputs)
    votes_tr = np.array([[s_idx[v.status] for v in inp.semantic_votes] for inp in train_inputs], dtype=int)
    priors = np.full(4, 0.25)
    alpha = np.ones(k_val, dtype=float)
    beta = np.ones(n_tr, dtype=float)

    for _ in range(max_iter):
        # E-step on training packets
        z_tr = beta[:, None] * alpha[None, :]  # (N_tr, K)
        ez = np.exp(np.clip(z_tr, -8.0, 8.0))
        p_corr = np.clip(ez / (ez + 3.0), 1e-6, 1.0 - 1e-6)
        p_err = (1.0 - p_corr) / 3.0
        log_q = np.log(priors + 1e-12)[None, :] + np.zeros((n_tr, 4))
        for c in range(4):
            match = (votes_tr == c)
            log_q[:, c] += np.sum(match * np.log(p_corr) + (~match) * np.log(p_err), axis=1)
        log_q -= np.max(log_q, axis=1, keepdims=True)
        q_tr = np.exp(log_q)
        q_tr /= np.sum(q_tr, axis=1, keepdims=True)

        # Expected correctness per (i, k)
        exp_match = np.zeros((n_tr, k_val))
        for c in range(4):
            exp_match += q_tr[:, [c]] * (votes_tr == c)

        old_alpha = alpha.copy()
        priors = (np.sum(q_tr, axis=0) + 1e-3) / (n_tr + 4e-3)
        # Closed-form / damped Newton M-step updates for alpha_k and beta_i
        acc_k = np.clip(np.mean(exp_match, axis=0), 0.28, 0.96)
        alpha_new = np.log(3.0 * acc_k / (1.0 - acc_k))
        alpha = 0.5 * alpha + 0.5 * np.clip(alpha_new / max(np.mean(beta), 0.2), 0.2, 4.0)

        acc_i = np.clip(np.sum(exp_match * alpha[None, :], axis=1) / np.sum(alpha), 0.28, 0.96)
        beta = np.clip(np.log(3.0 * acc_i / (1.0 - acc_i)) / max(np.mean(alpha), 0.2), 0.35, 2.5)
        if np.max(np.abs(alpha - old_alpha)) < tol:
            break

    # Predict on test packets (inferring instance difficulty beta_test from inter-verifier agreement)
    posteriors = []
    for inp in test_inputs:
        v_arr = np.array([s_idx[v.status] for v in inp.semantic_votes], dtype=int)
        # Instance concordance proxy for difficulty beta_test (unsupervised, votes only)
        max_cnt = max(Counter(v_arr.tolist()).values())
        beta_t = {4: 1.45, 3: 1.05, 2: 0.75, 1: 0.45}[max_cnt]
        ez_t = np.exp(np.clip(beta_t * alpha, -8.0, 8.0))
        pc = np.clip(ez_t / (ez_t + 3.0), 1e-6, 1.0 - 1e-6)
        pe = (1.0 - pc) / 3.0
        lq = np.log(priors + 1e-12).copy()
        for c in range(4):
            m_c = (v_arr == c)
            lq[c] += float(np.sum(m_c * np.log(pc) + (~m_c) * np.log(pe)))
        lq -= np.max(lq)
        qt = np.exp(lq)
        qt /= np.sum(qt)
        posteriors.append(qt.tolist())
    return posteriors


def _route_sota(posterior, context, policy_name, defer_cost=0.5):
    """Canonical SOTA selective classification, abstention, and ungated L2D policies."""
    true_allowed, _ = eligible(context)
    p = np.asarray(posterior, dtype=float)
    if policy_name == "selective_threshold":
        # Chow (1970) / Franc et al. (2023): abstain (defer) if max posterior < tau=0.85
        if float(np.max(p)) < 0.85:
            action = "defer"
        else:
            winners = [ACTIONS[j] for j, v in enumerate(p) if abs(v - float(np.max(p))) <= 1e-12]
            action = winners[0] if len(winners) == 1 else "defer"
    elif policy_name == "margin_rejector":
        # Mao et al. (2024): multi-class predictor-rejector abstaining when top-2 margin < gamma=0.50
        s_p = np.sort(p)[::-1]
        if float(s_p[0] - s_p[1]) < 0.50:
            action = "defer"
        else:
            winners = [ACTIONS[j] for j, v in enumerate(p) if abs(v - float(np.max(p))) <= 1e-12]
            action = winners[0] if len(winners) == 1 else "defer"
    elif policy_name == "posthoc_l2d":
        # Narasimhan et al. (2022): post-hoc cost-sensitive learning-to-defer over C(a, s) without governance gates
        costs = BASE_COST.copy()
        costs[3, :3] = defer_cost
        risks = costs @ p
        best = float(np.min(risks))
        action = next(a for a in TIE_ORDER if risks[ACTIONS.index(a)] <= best + 1e-12)
    else:
        raise ValueError(policy_name)
    return action, (action in true_allowed)


def save_sota_baselines(index, targets, contexts, inputs, predictions):
    """Evaluate canonical crowd-consensus, selective-classification, abstention & L2D baselines (4-fold + 5-fold CV)."""
    # 1. Fit GLAD across 4-fold leave-one-source-out splits
    glad_preds = {}
    for source in sorted({r["source_dataset"] for r in index}):
        training = [r for r in index if r["source_dataset"] != source]
        testing = [r for r in index if r["source_dataset"] == source]
        posts = _glad_fit_and_predict(
            tuple(inputs[r["audit_join_id"]] for r in training),
            tuple(inputs[r["audit_join_id"]] for r in testing),
        )
        for row, post in zip(testing, posts):
            glad_preds[row["audit_join_id"]] = post

    specs = [
        ("glad_direct", "GLAD EM / direct (Whitehill et al., 2009)", "glad", "direct"),
        ("selective_threshold", "Selective-Threshold (Franc et al., 2023)", "dawid_skene", "selective_threshold"),
        ("margin_rejector", "Margin-Abstention (Mao et al., 2024)", "adaptive", "margin_rejector"),
        ("posthoc_l2d", "Post-Hoc Cost L2D (Narasimhan et al., 2022)", "dawid_skene", "posthoc_l2d"),
        ("deciv_glad_risk", "DeCiv + GLAD / risk", "glad", "risk"),
    ]

    # 5-fold stratified splits (identical RNG seed and parent folds as save_five_fold_cv)
    rng = np.random.default_rng(SPEC["bootstrap_seed"])
    by_src = defaultdict(list)
    for row in index:
        if row["parent_id"] not in by_src[row["source_dataset"]]:
            by_src[row["source_dataset"]].append(row["parent_id"])
    parent_fold = {}
    for src, plist in sorted(by_src.items()):
        plist = sorted(plist)
        perm = rng.permutation(len(plist))
        for idx_pos, p_idx in enumerate(perm):
            parent_fold[plist[p_idx]] = idx_pos % 5

    cv5_folds = defaultdict(list)
    for fold_id in range(5):
        training = [r for r in index if parent_fold[r["parent_id"]] != fold_id]
        testing = [r for r in index if parent_fold[r["parent_id"]] == fold_id]
        tr_in = tuple(inputs[r["audit_join_id"]] for r in training)
        te_in = tuple(inputs[r["audit_join_id"]] for r in testing)
        successes, totals = Counter(), Counter()
        for row in training:
            for vote in inputs[row["audit_join_id"]].semantic_votes:
                totals[vote.validator_id] += 1
                successes[vote.validator_id] += (vote.status == targets[row["audit_join_id"]])
        rel = {vid: (successes[vid] + 1) / (totals[vid] + 2) for vid in sorted(totals)}
        ds_out = SemanticDawidSkeneEstimator().fit(tr_in).predict_proba(te_in)
        glad_out = _glad_fit_and_predict(tr_in, te_in)

        for sid, _, est_name, r_name in specs:
            f_rows = []
            for row, d_o, g_o in zip(testing, ds_out, glad_out):
                key = row["audit_join_id"]
                if est_name == "glad":
                    post = g_o
                elif est_name == "dawid_skene":
                    post = [d_o.probability(s) for s in STATUSES]
                else:
                    post = aggregate(inputs[key].semantic_votes, rel, est_name)
                if r_name in ("direct", "risk"):
                    act, _, _, _, adm = route(post, contexts[key], r_name)
                else:
                    act, adm = _route_sota(post, contexts[key], r_name)
                ref = reference_action(targets[key], contexts[key])
                f_rows.append({"predicted_action": act, "reference_action": ref, "correct": int(act == ref), "false_accept": int(act == "accept" and ref != "accept"), "eligible": int(adm)})
            cv5_folds[sid].append(summarize(f_rows))

    out_rows = []
    for sid, label, est_name, r_name in specs:
        rows_4f = []
        for row in index:
            key = row["audit_join_id"]
            post = glad_preds[key] if est_name == "glad" else predictions[est_name][key]
            if r_name in ("direct", "risk"):
                act, _, _, _, adm = route(post, contexts[key], r_name)
            else:
                act, adm = _route_sota(post, contexts[key], r_name)
            ref = reference_action(targets[key], contexts[key])
            rows_4f.append({"predicted_action": act, "reference_action": ref, "correct": int(act == ref), "false_accept": int(act == "accept" and ref != "accept"), "eligible": int(adm)})
        s4 = summarize(rows_4f)
        accs_5f = [float(m["accuracy"]) for m in cv5_folds[sid]]
        fars_5f = [float(m["far"]) for m in cv5_folds[sid]]
        out_rows.append({
            "baseline_id": sid,
            "label": label,
            **s4,
            "cv5_accuracy_mean": float(np.mean(accs_5f)),
            "cv5_accuracy_std": float(np.std(accs_5f, ddof=1)),
            "cv5_far_mean": float(np.mean(fars_5f)),
            "cv5_far_std": float(np.std(fars_5f, ddof=1)),
        })
    write_csv("sota_baselines.csv", out_rows)


def save_lambda_and_efficiency(index, targets, contexts, inputs):
    """Measure adaptive concordance strength sweep (lambda) and per-module online latency."""
    sources = sorted({r["source_dataset"] for r in index})
    fold_rel = {}
    for src in sources:
        tr = [r for r in index if r["source_dataset"] != src]
        succ, tot = Counter(), Counter()
        for r in tr:
            for v in inputs[r["audit_join_id"]].semantic_votes:
                tot[v.validator_id] += 1
                succ[v.validator_id] += (v.status == targets[r["audit_join_id"]])
        fold_rel[src] = {vid: (succ[vid] + 1) / (tot[vid] + 2) for vid in sorted(tot)}

    lam_rows = []
    for lam in (0.0, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95):
        rows = []
        false_rej_on_acc = 0
        for r in index:
            k = r["audit_join_id"]
            p = aggregate(inputs[k].semantic_votes, fold_rel[r["source_dataset"]], "adaptive", strength=lam)
            act, _, _, _, adm = route(p, contexts[k], "risk")
            ref = reference_action(targets[k], contexts[k])
            if ref == "accept" and act in ("reject", "revise"):
                false_rej_on_acc += 1
            rows.append({
                "predicted_action": act,
                "reference_action": ref,
                "correct": int(act == ref),
                "false_accept": int(act == "accept" and ref != "accept"),
                "eligible": int(adm),
            })
        s = summarize(rows)
        lam_rows.append({"lambda": lam, "false_reject_on_accept": false_rej_on_acc, **s})
    write_csv("lambda_sensitivity.csv", lam_rows)

    # Per-module microsecond latency benchmark averaged over 10 full passes (18,000 packet evaluations)
    n_pkts = len(index)
    reps = 10
    votes_list = [inputs[r["audit_join_id"]].semantic_votes for r in index]
    ctx_list = [contexts[r["audit_join_id"]] for r in index]
    rel0 = fold_rel[sources[0]]
    tr0 = tuple(inputs[r["audit_join_id"]] for r in index if r["source_dataset"] != sources[0])
    all_in = tuple(inputs[r["audit_join_id"]] for r in index)

    t0 = time.perf_counter()
    for _ in range(reps):
        posts_fix = [aggregate(v, rel0, "fixed") for v in votes_list]
    t_fix = (time.perf_counter() - t0) * 1e6 / (n_pkts * reps)

    t0 = time.perf_counter()
    for _ in range(reps):
        posts_adp = [aggregate(v, rel0, "adaptive", strength=0.5) for v in votes_list]
    t_adp = (time.perf_counter() - t0) * 1e6 / (n_pkts * reps)

    t0 = time.perf_counter()
    ds_est = SemanticDawidSkeneEstimator().fit(tr0)
    t_ds_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = ds_est.predict_proba(all_in)
    t_ds = (time.perf_counter() - t0) * 1e6 / (n_pkts * reps)

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = [eligible(c) for c in ctx_list]
    t_gate = (time.perf_counter() - t0) * 1e6 / (n_pkts * reps)

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = [route(p, c, "risk") for p, c in zip(posts_adp, ctx_list)]
    t_route = (time.perf_counter() - t0) * 1e6 / (n_pkts * reps)

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = [digest({"p": p, "c": c}) for p, c in zip(posts_adp, ctx_list)]
    t_hash = (time.perf_counter() - t0) * 1e6 / (n_pkts * reps)

    eff_rows = [
        {"module": "Stage I: Fixed consensus", "complexity": "O(K|S|)", "online_us": round(t_fix, 2), "offline_fit_s": 0.0, "gpu_hours": 0.0},
        {"module": "Stage I: Adaptive concordance (lambda=0.5)", "complexity": "O(K^2)", "online_us": round(t_adp, 2), "offline_fit_s": 0.0, "gpu_hours": 0.0},
        {"module": "Stage I: Dawid-Skene EM", "complexity": "O(K|S|)", "online_us": round(t_ds, 2), "offline_fit_s": round(t_ds_fit, 3), "gpu_hours": 0.0},
        {"module": "Stage IIa: Provenance & repairability gate", "complexity": "O(1)", "online_us": round(t_gate, 2), "offline_fit_s": 0.0, "gpu_hours": 0.0},
        {"module": "Stage IIb: Asymmetric expected-risk router", "complexity": "O(|A||S|)", "online_us": round(max(0.0, t_route - t_gate), 2), "offline_fit_s": 0.0, "gpu_hours": 0.0},
        {"module": "Accountability Plane: SHA-256 hash chain", "complexity": "O(|x|)", "online_us": round(t_hash, 2), "offline_fit_s": 0.0, "gpu_hours": 0.0},
        {"module": "End-to-End: DeCiv Fixed / risk", "complexity": "O(K|S|+|x|)", "online_us": round(t_fix + t_route + t_hash, 2), "offline_fit_s": 0.0, "gpu_hours": 0.0},
        {"module": "End-to-End: DeCiv Adaptive / risk", "complexity": "O(K^2+|x|)", "online_us": round(t_adp + t_route + t_hash, 2), "offline_fit_s": 0.0, "gpu_hours": 0.0},
        {"module": "End-to-End: DeCiv + Dawid-Skene / risk", "complexity": "O(K|S|+|x|)", "online_us": round(t_ds + t_route + t_hash, 2), "offline_fit_s": round(t_ds_fit, 3), "gpu_hours": 0.0},
    ]
    write_csv("module_efficiency.csv", eff_rows)


def save_trace(index, targets, contexts, inputs, predictions):
    by_parent = defaultdict(list)
    for row in index:
        by_parent[row["parent_id"]].append(row)
    candidates = []
    for parent, rows in sorted(by_parent.items()):
        intact = next((r for r in rows if r["variant_name"] == "v0_intact_control"), None)
        dropout = next((r for r in rows if r["variant_name"] == "v1_provenance_dropout"), None)
        if intact and dropout and targets[intact["audit_join_id"]] == "supported":
            keys = [intact["audit_join_id"], dropout["audit_join_id"]]
            if all(max(predictions["fixed"][k]) == 1. for k in keys) and all(predictions["fixed"][k][0] == 1. for k in keys):
                candidates = [intact, dropout]
                break
    assert candidates, "No deterministic paired illustration found"
    packet_file = "example_packets.csv" if (ROOT / "data" / "example_packets.csv").exists() else "deciv_hard_1800.csv"
    packet_rows = {r["challenge_id"]: r for r in read_rows(packet_file)}
    trace = []
    for row in candidates:
        key = row["audit_join_id"]
        packet = packet_rows[row["challenge_id"]]
        action, risks, reasons, reason, admissible = route(predictions["fixed"][key], contexts[key], "risk")
        record = {"illustration_selection": "first lexicographic supported parent with unanimous supported votes in intact and dropout variants; selected for explanation, not efficacy evidence", "packet_id": row["challenge_id"], "parent_id": row["parent_id"], "claim": packet["claim_text"], "evidence": packet["evidence_text"], "source_url": packet["source_url"], "packet_hash": digest({k: packet[k] for k in ("claim_text", "evidence_text", "source_url")}), "context": contexts[key], "votes": [v.payload() for v in inputs[key].semantic_votes], "posterior": predictions["fixed"][key], "method": "fixed", "spec_hash": digest(SPEC), "expected_costs": dict(zip(ACTIONS, risks)), "eligible_actions": sorted(eligible(contexts[key])[0]), "action": action, "reason": reason, "blocked_accept_reasons": reasons, "writes_trusted_store": action == "accept", "audit_log_written": True, "record_only_not_database_transaction": True, "reference_action_evaluation_only": reference_action(targets[key], contexts[key])}
        record["audit_record_hash"] = digest(record)
        trace.append(record)
    (ROOT / "results" / "execution_trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=True)+"\n", encoding="utf-8")


def targeted_checks(index, targets, contexts, inputs, predictions, rows):
    complete = {"provenance_status": "complete", "evidence_auditability": "auditable", "human_review_required": "false", "repairability": "no_repair"}
    missing = {**complete, "provenance_status": "incomplete"}
    assert route([1,0,0,0], complete, "risk")[0] == "accept"
    assert route([1,0,0,0], missing, "risk")[0] == "defer"
    assert route([0,0,1,0], complete, "risk")[0] == "defer"
    assert route([0,0,1,0], {**complete, "repairability": "verified_repair"}, "risk")[0] == "revise"
    assert route([.5,.5,0,0], complete, "direct")[0] == "defer"
    assert len(rows) == len(index)*len(ESTIMATORS)*2
    grouped = metrics_by_group(rows)
    expected_ids = {r["audit_join_id"] for r in index}
    input_hashes = {}
    for key, group in grouped.items():
        assert {r["audit_join_id"] for r in group} == expected_ids
        for row in group:
            before = input_hashes.setdefault(row["audit_join_id"], row["runtime_input_hash"])
            assert before == row["runtime_input_hash"]
        if key[1] == "risk":
            assert all(row["eligible"] for row in group)
    assert predictions["fixed"] == predictions["majority"]
    # This is structural leakage prevention: the prediction API has no target/ID parameters.
    assert set(aggregate.__code__.co_varnames[:aggregate.__code__.co_argcount]) == {"votes", "reliability", "mode", "strength"}
    return {"edge_cases": "passed", "shared_input_hashes": "passed", "source_parent_disjointness": "passed", "test_labels_excluded_from_prediction_signature": "passed", "proxy_observations_replaced_with_zeros": True, "same_votes_for_all_methods": True, "policy_eligibility_checks": "passed"}


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    # Persist parameter specification before estimating or scoring any method.
    (ROOT / "experiment_spec.json").write_text(json.dumps(SPEC, indent=2)+"\n", encoding="utf-8")
    index, targets, contexts, inputs = load_study()
    predictions, states = fit_folds(index, targets, inputs)
    rows = result_rows(index, targets, contexts, inputs, predictions)
    groups = metrics_by_group(rows)
    write_csv("predictions.csv", rows)
    write_csv("main_metrics.csv", [{"method": key[0], "router": key[1], **summarize(group)} for key, group in groups.items()])
    write_csv("paired_differences.csv", paired_bootstrap(index, groups))
    write_csv("action_confusion.csv", [{"method": mode, "router": router, "reference_action": ref, "predicted_action": pred, "count": sum(r["reference_action"] == ref and r["predicted_action"] == pred for r in group)} for (mode, router), group in groups.items() for ref in ACTIONS for pred in ACTIONS])
    write_csv("by_source.csv", [{"method": mode, "router": router, "source": source, **summarize([r for r in group if r["held_out_source"] == source])} for (mode, router), group in groups.items() for source in sorted({r["source_dataset"] for r in index})])
    save_sensitivity(index, targets, contexts, predictions)
    save_five_fold_cv(index, targets, contexts, inputs)
    save_ablations(index, targets, contexts, predictions)
    save_sota_baselines(index, targets, contexts, inputs, predictions)
    save_lambda_and_efficiency(index, targets, contexts, inputs)
    save_trace(index, targets, contexts, inputs, predictions)
    checks = targeted_checks(index, targets, contexts, inputs, predictions, rows)
    legacy = {r["audit_join_id"]: r["rubric_admission_action"] for r in read_rows("governance_targets_v2.csv")}
    study = {"n": len(index), "parents": 450, "votes": 7200, "sources": dict(Counter(r["source_dataset"] for r in index)), "semantic_counts": dict(Counter(targets.values())), "policy_reference_counts": dict(Counter(reference_action(targets[r["audit_join_id"]], contexts[r["audit_join_id"]]) for r in index)), "legacy_reference_differences": sum(legacy[r["audit_join_id"]] != reference_action(targets[r["audit_join_id"]], contexts[r["audit_join_id"]]) for r in index), "checks": checks, "fold_states": states, "spec_hash": digest(SPEC), "synthetic_vote_model": "four independent validators, each diagonal 0.7 and off-diagonal 0.1, target-conditioned; one frozen vote realization; confidence absent", "evidence_boundary": "controlled policy-recovery simulation, not prospective factual verification or a human study"}
    (ROOT / "results" / "study_report.json").write_text(json.dumps(study, indent=2)+"\n", encoding="utf-8")
    artifacts = {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix not in ('.pyc',) and p.name != 'artifact_manifest.json'}
    (ROOT / "results" / "artifact_manifest.json").write_text(json.dumps(artifacts, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in study.items() if k != "fold_states"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
