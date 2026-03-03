# BFSI LLM Eval

> Generate adversarial evaluation datasets for testing LLMs in Banking, Financial Services & Insurance domains.

**Status:** WIP → first release

## What This Does

Scrapes publicly available BFSI content (Wikipedia, bank/insurer/regulator websites, SEC EDGAR, OSFI) and uses LLM-driven generation to produce a structured prompt dataset (~2,910 records) that stress-tests model behavior across four dimensions:

| Dimension | What It Probes | Subdimensions |
|-----------|---------------|---------------|
| **Hallucination** | Factual accuracy, faithfulness to context, citation integrity | closed-book truthfulness, open-book faithfulness, uncertainty calibration, self-contradiction, citation fidelity |
| **Consistency** | Stable outputs across runs, temps, and linked questions | repeat stability, temperature sensitivity, position coherence, refusal stability |
| **Robustness** | Resilience to phrasing changes, format constraints, ambiguity | phrasing variants, instruction following, zero-shot vs few-shot, ambiguity handling |
| **Safety** | Refusal accuracy, over-refusal, bias, toxicity, tone | should-refuse, over-refusal triggers, toxicity susceptibility, BBQ-style bias, tone professionalism |

### Difficulty Levels

| Level | Share | Design |
|-------|-------|--------|
| Easy | 20% | Straightforward, tests basic competence |
| Medium | 40% | Requires domain knowledge + careful reasoning |
| Hard | 40% | Applies named adversarial strategies (plausible misinformation, conflicting context injection, leading question framing, regulatory confusion, multi-step reasoning traps, etc.) with 2+ expected failure modes per prompt |

## Quick Start

```bash
# clone & install
git clone https://github.com/sabyasm/bfsi-llm-eval.git
cd bfsi-llm-eval
pip install -r requirements.txt

# configure
cp .env.example .env
# edit .env → set API_BASE, API_KEY, MODEL_NAME

# run full pipeline (scrape → generate → validate → assemble → export)
python src/pipeline.py --mode full_refresh

# or incremental (fill gaps only)
python src/pipeline.py --mode incremental

# dry run (preview generation plan, no API calls)
python src/pipeline.py --dry-run

# filter by domain/dimension
python src/pipeline.py --domain banking --dimension hallucination

# skip scraping (use cached data)
python src/pipeline.py --no-scrape
```

Output lands in `data/final/` as Parquet + JSONL.

### Push to HuggingFace

```bash
python scripts/push_to_hf.py --repo your-org/bfsi-llm-eval
```

### Run Tests

```bash
pytest tests/
```

## How the Dataset Is Built

```
Public Sources (Wikipedia, bank sites, regulators, SEC EDGAR)
  ↓  scrape + respect robots.txt
Raw Text → 500-word chunks
  ↓  22 prompt templates × LLM generation
Candidate Prompts
  ↓  Pydantic validation + dedup (cosine similarity)
Final Dataset (Parquet/JSONL) → HuggingFace Hub
```

Each record includes: prompt, dimension, subdimension, difficulty, source domain, geography (Canada/USA), expected behavior, pass criteria, adversarial strategy (if hard), expected failure modes, and linked prompt IDs for consistency checks.

## Project Structure

```
config/          # generation_config.yaml, source_config.yaml
src/
  scraper/       # Wikipedia, web, API scrapers + chunker
  generator/     # LLM client, prompt builder, 22 YAML templates
  validator/     # Pydantic schema validation
  dataset/       # Assembler, exporter, HF card generator
  pipeline.py    # Main entry point
scripts/         # backfill, push-to-hf, shell wrappers
tests/           # Unit + acceptance tests
data/            # Runtime data (gitignored)
```

## License

MIT
