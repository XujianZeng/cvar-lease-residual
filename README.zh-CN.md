# CVaR 汽车租赁残值分析

简体中文 | [English](README.md)

本项目研究**汽车租赁资产证券化中的计价残值如何设定**：车辆退租后，实际处置收入可能低于为其设定的残值。较高的计价残值可以保留更多资产池价值，但也可能放大不利情形下的损失。项目要回答的问题是：考虑退租风险的规则，能否在控制这类损失的同时保留更多价值？

对每笔符合条件的已终止租赁，代码将退租车辆的缺口定义为 `max(计价残值 − 净处置收入, 0)`；未退租车辆的缺口记为零。尾部风险使用 **99% CVaR** 衡量，即缺口最严重的 1% 租赁的平均缺口。代码在训练样本中，以发行人报告的基础残值对应的 CVaR 为约束，拟合不同的残值计价比例，再将规则原样应用到后续样本。比较规则时还会固定平均计价残值，以便在相同价值水平下比较尾部风险。

## 代码主要完成什么

1. **采集和整理样本。** 从美国 SEC 公开的 ABS-EE 资产级申报文件提取租赁终止字段，按资产编号合并月度记录，并将退租车辆与后续净处置收入关联。
2. **拟合残值计价规则。** 比较统一比例、发行人基础残值、品牌分组比例和考虑退租风险的分桶规则。分桶方法利用租赁起始信息估计退租概率及潜在处置缺口，并在 CVaR 约束下优化各桶比例。
3. **检验规则能否推广。** 将已拟合规则用于 GM、日产、大众和福特信贷的后续样本，并进行汇总比较、压力情景、梯度提升模型基准，以及覆盖 2019—2024 年的 GM 跨年份检验。

这些代码用于历史租赁样本的研究分析，不是生产环境的残值定价服务，也不是完整的资产池损失模型。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `collection/collect_sec.py` | 从 SEC 申报文件采集所需租赁记录。 |
| `analysis/analyze.py` | 最初的 GM 品牌分组试验。 |
| `analysis/buffer_study.py` | 跨发行人的主要退租风险缓冲研究。 |
| `analysis/revision_study.py` | 汇总证据、压力分析及梯度提升模型基准。 |
| `analysis/cycle_study.py` | GM 相邻年份之间的检验。 |
| `analysis/summarize.py` | 根据研究结果生成跨发行人表格和图表。 |

## 运行方式

需要 Python 3.10 或更新版本。在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

仓库**不包含输入数据**。分析模块需要将 JSONL 样本放在 `data/` 下；所需文件名和时间窗口定义在 `analysis/buffer_study.py` 与 `analysis/cycle_study.py` 中。可以自行提供数据文件，也可以采集对应的 SEC 申报文件。采集前要设置包含姓名及联系邮箱的 `SEC_USER_AGENT`：

```bash
export SEC_USER_AGENT='Your Name your.email@example.com'
.venv/bin/python -m collection.collect_sec \
  --cik 0001935132 --start 2024-01-01 --end 2024-12-31 \
  --output data/gm_2022_3_2024.jsonl
```

备齐某发行人所需的全部样本文件后，在仓库根目录运行：

```bash
.venv/bin/python -m analysis.buffer_study --issuer gm
```

其他发行人参数为 `nissan`、`vw` 和 `ford`。补充分析运行 `analysis.revision_study`，跨年份检验运行 `analysis.cycle_study`；四家发行人的结果均生成后，可运行 `analysis.summarize`。`cycle_study` 还需要 `revision_results.json`。最初的 GM 试验可用 `python -m analysis.analyze --train ... --test ... --final ... --output results.json` 运行。

本仓库只保存代码，不收录数据、生成的结果与图表，也不收录论文材料。结果写入仓库根目录，图表写入 `figures/`。
