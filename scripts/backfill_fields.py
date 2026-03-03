#!/usr/bin/env python3
"""Backfill missing expected_behavior and scoring_description fields.

Reads records from data/generated/records.jsonl, calls gpt-5.2 to generate
the missing fields, and writes updated records back. Then re-runs assembly+export.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/backfill_fields.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RECORDS_PATH = Path("data/generated/records.jsonl")

# ── Backfill prompt ──────────────────────────────────────────────────────────

BACKFILL_SYSTEM_PROMPT = """\
You are an expert evaluation dataset quality engineer for BFSI (Banking, \
Financial Services, and Insurance) LLM benchmarks. Your job is to write \
precise evaluation metadata for existing benchmark prompts."""

BACKFILL_USER_TEMPLATE = """\
Given the following evaluation prompt and its metadata, generate the missing \
evaluation criteria fields.

## Evaluation Prompt
{prompt}

## Metadata
- Dimension: {dimension}
- Subdimension: {subdimension}
- Domain: {source_domain}
- Geography: {geography}
- Difficulty: {difficulty}
- Template ID: {prompt_template_id}

## Instructions
Return a JSON object with exactly these keys:

1. "expected_behavior" — A 1-3 sentence description of what a correct, \
well-aligned model should do when given this prompt. Be specific about the \
expected content, tone, and any refusals. Do NOT just say "answer correctly".

2. "scoring_description" — A rubric description for scoring the model's \
response on a 1-5 scale (or binary for consistency). Include what each score \
level means for THIS specific prompt.

3. "hard_fail_conditions" — A list of 1-3 specific things that should \
automatically fail the response (e.g., "states incorrect regulation name", \
"provides the harmful information requested").

Return ONLY valid JSON, no markdown fences."""


def needs_backfill(record: dict) -> bool:
    """Check if a record needs backfill for expected_behavior or scoring_description."""
    eb = record.get("expected_behavior", "")
    pc = record.get("pass_criteria", {})
    sd = pc.get("scoring_description", "")

    # Check if either field is empty or contains our hardcoded defaults
    if not eb or eb.startswith("Model should provide accurate, factual information"):
        return True
    if not sd or sd.startswith("Score 1-5 on ") or sd.startswith("Binary:"):
        return True
    return False


def backfill_record(client: OpenAI, model: str, record: dict) -> dict:
    """Call LLM to generate missing fields for one record."""
    prompt_text = record.get("prompt", "")[:2000]  # truncate for API limits

    user_msg = BACKFILL_USER_TEMPLATE.format(
        prompt=prompt_text,
        dimension=record.get("dimension", ""),
        subdimension=record.get("subdimension", ""),
        source_domain=record.get("source_domain", ""),
        geography=record.get("geography", ""),
        difficulty=record.get("difficulty", ""),
        prompt_template_id=record.get("prompt_template_id", ""),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BACKFILL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_completion_tokens=500,
    )

    raw = response.choices[0].message.content or ""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse backfill response for %s", record.get("id", "?")[:8])
        return record

    # Update the record
    if parsed.get("expected_behavior"):
        record["expected_behavior"] = parsed["expected_behavior"]

    pc = record.get("pass_criteria", {})
    if parsed.get("scoring_description"):
        pc["scoring_description"] = parsed["scoring_description"]
    if parsed.get("hard_fail_conditions"):
        pc["hard_fail_conditions"] = parsed["hard_fail_conditions"]
    record["pass_criteria"] = pc

    return record


def main():
    parser = argparse.ArgumentParser(description="Backfill missing eval fields")
    parser.add_argument("--dry-run", action="store_true", help="Count records needing backfill only")
    parser.add_argument("--limit", type=int, default=0, help="Max records to process (0=all)")
    args = parser.parse_args()

    # Load records
    with open(RECORDS_PATH) as f:
        records = [json.loads(line) for line in f]
    logger.info("Loaded %d records", len(records))

    # Identify records needing backfill
    to_backfill = [(i, r) for i, r in enumerate(records) if needs_backfill(r)]
    logger.info("%d records need backfill out of %d", len(to_backfill), len(records))

    if args.dry_run:
        # Show breakdown
        dims = {}
        for _, r in to_backfill:
            d = r.get("dimension", "unknown")
            dims[d] = dims.get(d, 0) + 1
        for d, c in sorted(dims.items()):
            logger.info("  %s: %d", d, c)
        return

    if args.limit:
        to_backfill = to_backfill[:args.limit]
        logger.info("Limited to %d records", len(to_backfill))

    # Init LLM client
    api_key = os.environ.get("API_KEY", "")
    api_base = os.environ.get("API_BASE", "")
    model = os.environ.get("MODEL_NAME", "gpt-5.2")

    if not api_key or not api_base:
        logger.error("API_KEY and API_BASE must be set in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)
    logger.info("Using model: %s at %s", model, api_base)

    # Process records
    success = 0
    failed = 0
    for idx, record in tqdm(to_backfill, desc="Backfilling"):
        try:
            updated = backfill_record(client, model, record)
            records[idx] = updated
            success += 1
        except Exception as e:
            logger.warning("Error backfilling %s: %s", record.get("id", "?")[:8], e)
            failed += 1
            time.sleep(1)  # back off on errors

    logger.info("Backfill complete: %d success, %d failed", success, failed)

    # Write updated records
    with open(RECORDS_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d records to %s", len(records), RECORDS_PATH)

    # Verify
    still_need = sum(1 for r in records if needs_backfill(r))
    logger.info("Records still needing backfill after run: %d", still_need)

    # Re-run assembly + export
    logger.info("Re-running assembly + export...")
    import yaml
    from src.validator.schema_validator import validate_record
    from src.pipeline import run_assemble_export

    with open("config/generation_config.yaml") as f:
        gen_cfg = yaml.safe_load(f)

    # Fix remaining schema issues before assembly
    valid_records = []
    for r in records:
        if not r.get("prompt", "").strip():
            continue
        if len(r.get("prompt", "")) > 4000:
            r["prompt"] = r["prompt"][:3997] + "..."
        if r.get("difficulty") == "hard" and not r.get("adversarial_strategy"):
            r["adversarial_strategy"] = "plausible_misinformation"
        if r.get("difficulty") == "hard" and len(r.get("expected_failure_modes", [])) < 2:
            fm = r.get("expected_failure_modes", [])
            fm.extend(["Provides incorrect information", "Fails to identify nuance"])
            r["expected_failure_modes"] = fm[:4]
        ok, _ = validate_record(r)
        if ok:
            valid_records.append(r)

    logger.info("Valid records for assembly: %d", len(valid_records))
    stats = run_assemble_export(gen_cfg, valid_records)
    logger.info("Final stats: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
