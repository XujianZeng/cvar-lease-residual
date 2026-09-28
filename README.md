# CVaR lease residual analysis

[简体中文](README.zh-CN.md) | English

Python code for collecting public SEC ABS-EE lease records and studying return-aware residual risk buffers. This repository contains source code only. It does not include input data, generated results or figures, or manuscript materials.

## Structure

| Directory | Purpose |
| --- | --- |
| `collection/` | `collect_sec.py`: collect lease termination records from SEC filings. |
| `analysis/` | `analyze.py`: original GM pilot; `buffer_study.py`: issuer studies; `revision_study.py`: additional benchmarks and stress analyses; `cycle_study.py`: cross-year evaluation; `summarize.py`: cross-issuer summary. |

`requirements.txt` lists the Python dependencies. Use Python 3.10 or newer. Run the commands below from the repository root with `python -m` so package imports work.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Data and execution

The analysis modules expect JSONL inputs under `data/` with the filenames defined in `analysis/buffer_study.py` and `analysis/cycle_study.py`. Supply those files locally or collect the relevant SEC filings. Data and generated outputs are ignored by Git.

Set an identifying SEC user agent with a contact address before collecting data:

```bash
export SEC_USER_AGENT='Your Name your.email@example.com'
.venv/bin/python -m collection.collect_sec \
  --cik 0001935132 --start 2024-01-01 --end 2024-12-31 \
  --output data/gm_2022_3_2024.jsonl
```

Once the required cohort files are present, run the studies:

```bash
.venv/bin/python -m analysis.buffer_study --issuer gm
.venv/bin/python -m analysis.buffer_study --issuer nissan
.venv/bin/python -m analysis.buffer_study --issuer vw
.venv/bin/python -m analysis.buffer_study --issuer ford
.venv/bin/python -m analysis.revision_study
.venv/bin/python -m analysis.cycle_study
.venv/bin/python -m analysis.summarize
```

`summarize` reads the four `study_results*.json` files. `cycle_study` also reads `revision_results.json`. Results are written to the repository root, and charts to `figures/`.

For the original GM pilot, provide its three cohorts explicitly:

```bash
.venv/bin/python -m analysis.analyze \
  --train data/gm_2022_3_2024.jsonl \
  --test data/gm_2023_1_2025.jsonl \
  --final data/gm_2023_3_2025_26.jsonl \
  --output results.json
```
