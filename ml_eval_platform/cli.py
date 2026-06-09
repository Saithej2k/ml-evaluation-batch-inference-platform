import argparse
import json
from pathlib import Path

from sqlalchemy.orm import Session

from ml_eval_platform.database import get_session_factory, init_db
from ml_eval_platform.services.demo_data import run_demo_batch
from ml_eval_platform.services.gates import check_metric_rows
from ml_eval_platform.services.serializers import run_to_metric_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage batch NLP evaluation runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create database tables if they do not exist.")

    demo = subparsers.add_parser("run-demo", help="Run the 4 dataset x 6 model evaluation workload.")
    demo.add_argument("--records-per-dataset", type=int, default=3_200)
    demo.add_argument("--output", type=Path, default=Path("artifacts/evaluation-results.json"))

    gate = subparsers.add_parser("check-gate", help="Compare current metrics with baseline thresholds.")
    gate.add_argument("--baseline", type=Path, required=True)
    gate.add_argument("--current", type=Path, required=True)
    gate.add_argument("--max-accuracy-drop", type=float, default=0.02)
    gate.add_argument("--max-latency-increase", type=float, default=0.15)

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
        print("database initialized")
        return

    if args.command == "run-demo":
        init_db()
        with get_session_factory()() as session:
            rows = _run_demo(session, args.records_per_dataset)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {len(rows)} evaluation summaries to {args.output}")
        return

    if args.command == "check-gate":
        baseline_rows = json.loads(args.baseline.read_text(encoding="utf-8"))
        current_rows = json.loads(args.current.read_text(encoding="utf-8"))
        outcome = check_metric_rows(
            baseline_rows=baseline_rows,
            current_rows=current_rows,
            max_accuracy_drop=args.max_accuracy_drop,
            max_latency_increase=args.max_latency_increase,
        )
        print(json.dumps(outcome, indent=2))
        if not outcome["passed"]:
            raise SystemExit(1)


def _run_demo(session: Session, records_per_dataset: int) -> list[dict[str, object]]:
    runs = run_demo_batch(session, records_per_dataset=records_per_dataset)
    return [run_to_metric_row(run) for run in runs]


if __name__ == "__main__":
    main()

