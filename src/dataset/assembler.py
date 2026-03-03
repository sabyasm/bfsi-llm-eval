"""Dataset assembly: dedup, split enforcement, ID assignment, shuffle."""

from __future__ import annotations

import json
import logging
import random
import uuid
from pathlib import Path

import numpy as np

from src.validator.schema_validator import validate_record

logger = logging.getLogger(__name__)

GENERATED_DIR = Path("data/generated")
VALIDATION_ERRORS_PATH = GENERATED_DIR / "validation_errors.jsonl"


class DatasetAssembler:
    """Load, deduplicate, rebalance, and finalize the dataset."""

    def __init__(self, config: dict):
        self.config = config
        self.seed: int = config.get("dataset", {}).get("seed", 42)
        self.target_total: int = config.get("dataset", {}).get("target_total", 2910)
        self.version: str = config.get("dataset", {}).get("version", "1.0.0")
        self.domain_split: dict[str, float] = config.get("domain_split", {})
        self.dimension_split: dict[str, int] = config.get("dimension_split", {})
        self.enable_dedup: bool = config.get("dataset", {}).get("enable_dedup", False)
        self._dedup_model = None

    def assemble(self, records: list[dict]) -> list[dict]:
        """Full assembly pipeline: validate -> dedup -> rebalance -> finalize."""
        logger.info("Starting assembly with %d raw records", len(records))

        # Step 1: Validate
        valid, invalid = self._validate_all(records)
        logger.info("Validated: %d valid, %d invalid", len(valid), len(invalid))

        # Step 2: Deduplicate (optional, off by default)
        if self.enable_dedup:
            deduped = self._deduplicate(valid)
            logger.info("After dedup: %d records", len(deduped))
        else:
            deduped = valid
            logger.info("Dedup skipped (enable_dedup=false)")

        # Step 3: Enforce splits
        balanced = self._enforce_splits(deduped)
        logger.info("After rebalancing: %d records", len(balanced))

        # Step 4: Finalize IDs and version
        finalized = self._finalize(balanced)
        logger.info("Finalized: %d records", len(finalized))

        return finalized

    def _validate_all(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        """Validate all records, log failures."""
        valid = []
        invalid = []
        VALIDATION_ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(VALIDATION_ERRORS_PATH, "w") as err_f:
            for rec in records:
                ok, errors = validate_record(rec)
                if ok:
                    valid.append(rec)
                else:
                    invalid.append(rec)
                    err_f.write(json.dumps({
                        "record_id": rec.get("id", "unknown"),
                        "errors": errors,
                    }) + "\n")

        return valid, invalid

    def _deduplicate(self, records: list[dict], threshold: float = 0.97) -> list[dict]:
        """Remove near-duplicate prompts using cosine similarity."""
        if len(records) <= 1:
            return records

        prompts = [r["prompt"] for r in records]

        try:
            from sentence_transformers import SentenceTransformer
            if self._dedup_model is None:
                self._dedup_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = self._dedup_model.encode(prompts, show_progress_bar=False)
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping dedup")
            return records

        # Compute cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid div by zero
        normalized = embeddings / norms
        sim_matrix = normalized @ normalized.T

        # Mark duplicates (keep first occurrence)
        keep = set(range(len(records)))
        for i in range(len(records)):
            if i not in keep:
                continue
            for j in range(i + 1, len(records)):
                if j not in keep:
                    continue
                if sim_matrix[i][j] > threshold:
                    keep.discard(j)
                    logger.debug("Removing duplicate: %s (sim=%.3f with %s)",
                                 records[j].get("id"), sim_matrix[i][j], records[i].get("id"))

        removed = len(records) - len(keep)
        if removed:
            logger.info("Dedup removed %d records", removed)

        return [records[i] for i in sorted(keep)]

    def _enforce_splits(self, records: list[dict]) -> list[dict]:
        """Enforce domain and dimension split targets within ±5%."""
        rng = random.Random(self.seed)

        # Enforce dimension splits
        if self.dimension_split:
            records = self._enforce_category_split(
                records, "dimension", self.dimension_split, rng
            )

        # Enforce domain splits (proportional to target_total)
        if self.domain_split:
            domain_targets = {
                d: round(self.target_total * w)
                for d, w in self.domain_split.items()
            }
            records = self._enforce_category_split(
                records, "source_domain", domain_targets, rng
            )

        return records

    def _enforce_category_split(
        self,
        records: list[dict],
        field: str,
        targets: dict[str, int],
        rng: random.Random,
    ) -> list[dict]:
        """Trim over-represented categories, warn about under-represented."""
        by_cat: dict[str, list[dict]] = {}
        for r in records:
            cat = r.get(field, "unknown")
            by_cat.setdefault(cat, []).append(r)

        result = []
        for cat, target in targets.items():
            cat_records = by_cat.get(cat, [])
            max_allowed = round(target * 1.05)  # +5% tolerance

            if len(cat_records) > max_allowed:
                rng.shuffle(cat_records)
                cat_records = cat_records[:max_allowed]
                logger.info("%s=%s: trimmed to %d (target %d)", field, cat, len(cat_records), target)
            elif len(cat_records) < round(target * 0.95):
                logger.warning("%s=%s: only %d records, target %d (under-represented)",
                               field, cat, len(cat_records), target)

            result.extend(cat_records)

        # Include records from categories not in targets
        for cat, cat_records in by_cat.items():
            if cat not in targets:
                result.extend(cat_records)

        return result

    def _finalize(self, records: list[dict]) -> list[dict]:
        """Assign final IDs, version, and shuffle."""
        rng = random.Random(self.seed)
        rng.shuffle(records)

        # Validate linked_prompt_ids cross-references
        all_ids = {r["id"] for r in records}
        for r in records:
            broken = [lid for lid in r.get("linked_prompt_ids", []) if lid not in all_ids]
            if broken:
                logger.warning("Record %s has broken linked_prompt_ids: %s", r["id"], broken)
                r["linked_prompt_ids"] = [lid for lid in r["linked_prompt_ids"] if lid in all_ids]

        # Update version
        for r in records:
            r["version"] = self.version

        return records

    def stats(self, records: list[dict]) -> dict:
        """Compute summary statistics for a set of records."""
        by_dim: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        by_diff: dict[str, int] = {}
        for r in records:
            by_dim[r["dimension"]] = by_dim.get(r["dimension"], 0) + 1
            by_domain[r["source_domain"]] = by_domain.get(r["source_domain"], 0) + 1
            by_diff[r["difficulty"]] = by_diff.get(r["difficulty"], 0) + 1
        return {
            "total": len(records),
            "by_dimension": by_dim,
            "by_domain": by_domain,
            "by_difficulty": by_diff,
        }
