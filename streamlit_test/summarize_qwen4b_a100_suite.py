#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize fixed-worker Qwen 4B suite outputs.")
    p.add_argument("suite_dir")
    args = p.parse_args(argv)

    suite_dir = Path(args.suite_dir).resolve()
    suite_result = json.loads((suite_dir / "suite_result.json").read_text(encoding="utf-8"))

    total_requests = 0
    success_requests = 0
    output_tokens_total = 0
    request_elapsed_total = 0.0

    for req_path in suite_dir.glob("worker_*_*/agents_*/memory_w_*/seed_distribution_*/memory_w_*/requests.jsonl"):
        with req_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec: Dict[str, Any] = json.loads(line)
                total_requests += 1
                parsed_token = rec.get("parsed_token")
                if parsed_token is not None:
                    success_requests += 1
                usage = rec.get("usage") or {}
                output_tokens_total += int(usage.get("output_tokens") or 0)
                request_elapsed_total += float(rec.get("request_elapsed_s") or 0.0)

    wall_time_s = float(suite_result.get("wall_time_s") or 0.0)
    summary = {
        **suite_result,
        "total_requests": total_requests,
        "success_requests": success_requests,
        "success_rate": (success_requests / total_requests) if total_requests else 0.0,
        "output_tokens_total": output_tokens_total,
        "requests_per_s": (total_requests / wall_time_s) if wall_time_s else 0.0,
        "output_tokens_per_s": (output_tokens_total / wall_time_s) if wall_time_s else 0.0,
        "mean_request_elapsed_s": (request_elapsed_total / total_requests) if total_requests else 0.0,
    }

    out_path = suite_dir / "suite_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
