"""Quick-start demonstration of DeCiv: Epistemic-Governance Decoupling & Four-Way Triage."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    print("=" * 92)
    print("DeCiv: Auditable Claim Admission for Persistent Knowledge Stores")
    print("Epistemic-Governance Decoupling | Four-Way Epistemic Triage | Dual-Plane Commitment")
    print("=" * 92)

    trace = json.loads((RESULTS / "execution_trace.json").read_text(encoding="utf-8"))
    print("\n[1] Paired Execution Trace: Identical Semantic Votes under Divergent Provenance")
    print("-" * 92)
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
        print("-" * 92)

    cv5 = {(r["method"], r["router"]): r for r in csv.DictReader((RESULTS / "five_fold_cv.csv").open(encoding="utf-8"))}
    print("\n[2] Shared-Input Benchmark Summary (N = 1,800 Packets, 4-Fold Cross-Domain + 5-Fold CV)")
    print("-" * 92)
    print(f"{'Method / Router':<28} {'Acc (%)':>8} {'FAR (%)':>8} {'Prec (%)':>9} {'Non-Def (%)':>12} {'5-Fold Acc (%)':>18}")
    print("-" * 92)
    with (RESULTS / "main_metrics.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = f"{row['method']} / {row['router']}"
            r5 = cv5[(row["method"], row["router"])]
            acc = 100 * float(row["accuracy"])
            far = 100 * float(row["far"])
            prec = 100 * float(row["accept_precision"]) if row["accept_precision"] else 0.0
            non_def = 100 * float(row["non_defer_fraction"])
            s5 = f"{100*float(r5['accuracy_mean']):.2f} +/- {100*float(r5['accuracy_std']):.2f}"
            print(f"{label:<28} {acc:>8.2f} {far:>8.2f} {prec:>9.2f} {non_def:>12.2f} {s5:>18}")
    print("-" * 92)

    print("\n[3] Component Ablation Highlights (N = 1,800 Packets)")
    print("-" * 92)
    with (RESULTS / "ablation_metrics.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(
                f"{row['label']:<36} | Acc={100*float(row['accuracy']):>5.2f}% "
                f"| FAR={100*float(row['far']):>5.2f}% "
                f"| RevRec={100*float(row['recall_revise']):>5.2f}% "
                f"| Viol={int(row['eligibility_violations']):>2d}"
            )
    print("=" * 92)


if __name__ == "__main__":
    main()
