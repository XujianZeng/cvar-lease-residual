# CVaR 汽车租赁残值分析

简体中文 | [English](README.md)

本仓库提供 Python 源码，用于从美国 SEC 公开的 ABS-EE 申报文件采集汽车租赁记录，并研究考虑车辆退租风险的残值缓冲方案。仓库只包含代码，不包含输入数据、生成的结果和图表，也不包含论文材料。

## 目录结构

| 目录 | 用途 |
| --- | --- |
| `collection/` | `collect_sec.py`：从 SEC 申报文件采集租赁终止记录。 |
| `analysis/` | `analyze.py`：最初的 GM 分析；`buffer_study.py`：各发行人的缓冲方案研究；`revision_study.py`：补充基准和压力分析；`cycle_study.py`：跨年份检验；`summarize.py`：跨发行人汇总。 |

Python 依赖列在 `requirements.txt` 中，需要 Python 3.10 或更新版本。请在仓库根目录使用 `python -m` 运行下列命令，以正确加载模块。

## 安装

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 数据与运行

分析模块需要将 JSONL 数据放在 `data/` 目录下，文件名见 `analysis/buffer_study.py` 和 `analysis/cycle_study.py`。可以自行提供这些文件，或从对应的 SEC 申报文件采集。数据和生成文件已被 Git 忽略。

采集前请设置包含姓名和联系邮箱的 SEC User-Agent：

```bash
export SEC_USER_AGENT='Your Name your.email@example.com'
.venv/bin/python -m collection.collect_sec \
  --cik 0001935132 --start 2024-01-01 --end 2024-12-31 \
  --output data/gm_2022_3_2024.jsonl
```

准备好所需的各期数据文件后，按需运行：

```bash
.venv/bin/python -m analysis.buffer_study --issuer gm
.venv/bin/python -m analysis.buffer_study --issuer nissan
.venv/bin/python -m analysis.buffer_study --issuer vw
.venv/bin/python -m analysis.buffer_study --issuer ford
.venv/bin/python -m analysis.revision_study
.venv/bin/python -m analysis.cycle_study
.venv/bin/python -m analysis.summarize
```

`summarize` 需要前四个命令生成的 `study_results*.json`；`cycle_study` 还需要 `revision_results.json`。结果写入仓库根目录，图表写入 `figures/`。

如需运行最初的 GM 分析，请显式指定三期数据：

```bash
.venv/bin/python -m analysis.analyze \
  --train data/gm_2022_3_2024.jsonl \
  --test data/gm_2023_1_2025.jsonl \
  --final data/gm_2023_3_2025_26.jsonl \
  --output results.json
```
