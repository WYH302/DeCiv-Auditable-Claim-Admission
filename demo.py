"""Quick-start demonstration of DeCiv: Epistemic-Governance Decoupling & Four-Way Triage."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    print("=" * 88)
    print("DeCiv: Auditable Claim Admission for Persistent Knowledge Stores")
    print("Epistemic-Governance Decoupling | Four-Way Epistemic Triage | Dual-Plane Commitment")
    print("=" * 88)

    trace = json.loads((RESULTS / "execution_trace.json").read_text(encoding="utf-8"))
    print("\n[1] Paired Execution Trace: Identical Semantic Votes under Divergent Provenance")
    print("-" * 88)
    for item in trace:
        pid = item["packet_id"]
        votes = [v["status"] for v in item["votes"]]
        ctx = item["context"]
        elig = item["eligible_actions"]
        print(f"Packet ID         : {pid} (Parent: {item['parent_id']})")
        print(f"Claim Text        : {item['claim']}")
        print(f"Validator Votes   : {votes}")
        print(
            f"Provenance Flags  : status={ctx['provenance_status']}, "
            f"auditability={ctx['evidence_auditability']}, "
            f"human_review={ctx['human_review_required']}"
        )
        print(f"Eligible Set E(x) : {elig}")
        print(
            f"Risk Router       : action={item['action']} "
            f"| Retrieval Plane Admit={item['writes_trusted_store']} "
            f"| Audit SHA-256={item['audit_record_hash'][:16]}..."
        )
        if item["blocked_accept_reasons"]:
            print(f"Blocked Reasons   : {item['blocked_accept_reasons']}")
        print("-" * 88)

    print("\n[2] Shared-Input Benchmark Summary (N = 1,800 Packets, 4-Fold Source-Held-Out CV)")
    print("-" * 88)
    print(f"{'Method / Router':<30} {'Acc (%)':>9} {'FAR (%)':>9} {'Prec (%)':>10} {'Accept (%)':>11} {'Non-Defer (%)':>14}")
    print("-" * 88)
    with (RESULTS / "main_metrics.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = f"{row['method']} / {row['router']}"
            acc = 100 * float(row["accuracy"])
            far = 100 * float(row["far"])
            prec = 100 * float(row["accept_precision"]) if row["accept_precision"] else 0.0
            acc_rate = 100 * float(row["acceptance_rate"])
            non_def = 100 * float(row["non_defer_fraction"])
            print(f"{label:<30} {acc:>9.2f} {far:>9.2f} {prec:>10.2f} {acc_rate:>11.2f} {non_def:>14.2f}")
    print("-" * 88)

    print("\n[3] Key Paired Cluster-Bootstrap Contrasts (2,000 Parent-Cluster Resamples)")
    print("-" * 88)
    with (RESULTS / "paired_differences.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            comp = f"{row['left']} vs {row['right']}"
            metric = row["metric"]
            est = 100 * float(row["difference"])
            lo = 100 * float(row["ci_low"])
            hi = 100 * float(row["ci_high"])
            print(f"{comp:<36} | {metric:<18}: {est:+7.2f} pp  (95% CI: [{lo:+.2f}%, {hi:+.2f}%])")
    print("=" * 88)


if __name__ == "__main__":
    main()
