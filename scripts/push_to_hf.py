#!/usr/bin/env python3
"""Push the BFSI LLM evaluation dataset to HuggingFace Hub.

Usage:
    # First login:
    huggingface-cli login

    # Then push:
    PYTHONPATH=. .venv/bin/python scripts/push_to_hf.py --repo sabyasm/bfsi-llm-eval

    # Dry run (validate only):
    PYTHONPATH=. .venv/bin/python scripts/push_to_hf.py --repo sabyasm/bfsi-llm-eval --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/final")


def main():
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="HF repo ID, e.g. sabyasm/bfsi-llm-eval")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show stats only")
    parser.add_argument("--private", action="store_true", help="Create a private dataset")
    args = parser.parse_args()

    # Load from parquet
    parquet_path = DATA_DIR / "dataset.parquet"
    readme_path = DATA_DIR / "README.md"

    if not parquet_path.exists():
        logger.error("dataset.parquet not found at %s", parquet_path)
        return

    ds = Dataset.from_parquet(str(parquet_path))
    logger.info("Loaded %d records from %s", len(ds), parquet_path)

    # Show summary
    logger.info("Schema: %s", ds.features)
    logger.info("Columns: %s", ds.column_names)

    dims = {}
    for row in ds:
        d = row["dimension"]
        dims[d] = dims.get(d, 0) + 1
    for d, c in sorted(dims.items()):
        logger.info("  %s: %d", d, c)

    if args.dry_run:
        logger.info("Dry run — not pushing. Dataset looks good!")
        return

    # Push dataset
    logger.info("Pushing to %s ...", args.repo)
    ds.push_to_hub(
        args.repo,
        split="test",
        private=args.private,
    )
    logger.info("Dataset pushed successfully!")

    # Upload README separately to ensure it's the dataset card
    api = HfApi()
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
        )
        logger.info("Dataset card (README.md) uploaded!")

    logger.info("Done! View at: https://huggingface.co/datasets/%s", args.repo)


if __name__ == "__main__":
    main()
