"""
Analyze DBBench task-limit errors from local AgentRuns JSONL logs.

Usage:
    python analyze_v4_errors.py --baseline logs/baseline/DB-Runs.jsonl --candidate logs/v4/dbbench-runs.jsonl

The public snapshot excludes the logs themselves. This helper expects sanitized
local logs to be supplied by the reviewer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_logs(file_path: Path) -> list[str]:
    runs = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines()]
    status_counts = Counter(run["output"]["status"] for run in runs)

    lines = [f"--- Analyzing: {file_path} ---", f"Status distribution: {status_counts}", ""]
    lines.append("Most common DB errors leading to task limit reached:")

    limit_errors: list[str] = []
    for run in runs:
        if run["output"]["status"] != "task limit reached":
            continue
        history = run["output"]["history"]
        user_messages = [m["content"] for m in history if m["role"] == "user"]
        if len(user_messages) >= 2:
            limit_errors.append(user_messages[-1][:100])

    for error, count in Counter(limit_errors).most_common(10):
        lines.append(f"{count} times: {error}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("analyze_v4_errors_output.txt"))
    args = parser.parse_args()

    report = analyze_logs(args.baseline) + [""] + analyze_logs(args.candidate)
    args.output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
