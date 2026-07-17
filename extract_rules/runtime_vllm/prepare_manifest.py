#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


RESET_FIELDS = {
    "status": "pendente",
    "process_id": "",
    "hora_inicio": "",
    "hora_fim": "",
    "caminho_csv_saida": "",
    "caminho_log_saida": "",
    "caminho_log_prompt_saida": "",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an immutable Phase 1 run manifest.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-alias", default="gemma3-4b-temp0")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames

    if not fields or len(rows) != 54:
        raise SystemExit(f"Expected 54 canonical cells, found {len(rows)}")

    for row in rows:
        row.update(RESET_FIELDS)
        row["modelo"] = args.model_alias
        row["modelo_executado"] = args.model_alias

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
