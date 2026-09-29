# CVaR 汽车租赁残值分析

简体中文 | [English](README.md)

本项目研究**汽车租赁资产证券化中的计价残值如何设定**：车辆退租后，实际处置收入可能低于为其设定的残值。较高的计价残值可以保留更多资产池价值，但也可能放大不利情形下的损失。项目要回答的问题是：考虑退租风险的规则，能否在控制这类损失的同时保留更多价值？

对每笔符合条件的已终止租赁，代码将退租车辆的缺口定义为 `max(计价残值 − 净处置收入, 0)`；未退租车辆的缺口记为零。尾部风险使用 **99% CVaR** 衡量，即缺口最严重的 1% 租赁的平均缺口。代码在训练样本中，以发行人报告的基础残值对应的 CVaR 为约束，拟合不同的残值计价比例，再将规则原样应用到后续样本。比较规则时还会固定平均计价残值，以便在相同价值水平下比较尾部风险。

## 2026-09-29 修订

基础残值锚定实现是风险可行的封顶启发式，不保证封顶目标最优。主样本与原计价规则保留；AUC 改为并列平均秩，bootstrap 尾部量加入有限模拟修正，两张百分比图与正文表格使用同一组比例 bootstrap 区间。

新增 `analysis.robustness_study`，在四种回款口径下完整重新拟合，报告样本流转、美元差额、尾部笔数、无定义比例次数、逐年剔除、月份整簇及 GB/LR 配对比较。新增分析明确为事后分析。保留三个月非正记录回款后，基础锚定确认性合并估计从 -2.2% 变为 +13.7%；该情景不代表已知最终回收。不能再宣称普遍稳健或非劣。

原方案形成于 2019 年 6 月试点观察之后、批量采集之前。哈希值只固定文本，不独立认证历史时间。本次新增内容详见带日期的修订方案。

四家发行方主分析之后，依次运行 `python -m analysis.revision_study`、`python -m analysis.cycle_study`、`python -m analysis.robustness_study`、`python -m analysis.summarize`；科学校验运行 `python -m unittest analysis.test_invariants -v`。可设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。论文投稿材料另附含保存结果的可复现快照。

## 代码主要完成什么

1. **采集和整理样本。** 从美国 SEC 公开的 ABS-EE 资产级申报文件提取租赁终止字段，按资产编号合并月度记录，并将退租车辆与后续净处置收入关联。
2. **拟合残值计价规则。** 比较统一比例、发行人基础残值、品牌分组比例和考虑退租风险的分桶规则。分桶方法利用已报告的租赁属性估计退租概率及潜在处置缺口，并在 CVaR 约束下拟合各桶比例；这些属性在发行时点是否可得仍需另行验证。
3. **检验规则能否推广。** 将已拟合规则用于 GM、日产、大众和福特信贷的后续样本，并进行汇总比较、压力情景、梯度提升模型基准，以及覆盖 2019—2024 年的 GM 跨年份检验。

这些代码用于历史租赁样本的研究分析，不是生产环境的残值定价服务，也不是完整的资产池损失模型。

## 研究记录

- [`analysis_plan_extension.md`](analysis_plan_extension.md) 记录 GM 在 2019—2024 年间的跨年份检验方案。源文件最后修改于北京时间 2026-09-25 10:51:37，SHA-256 为 `74266ffe5d0c36986e78ad13118e101294fb0c36b972936bfdff382ffba55e97`。
- [`filings_used.csv`](filings_used.csv) 列出 17 个数据集所用的 185 份 SEC XML 文件，包括申报日期、accession 编号、来源链接、文件大小和提取的事件记录数。它是文件清单，不含逐笔租赁记录。
- [`revision_protocol_2026-09-29.md`](revision_protocol_2026-09-29.md) 记录事后修正和敏感性分析。除包导入方式及仓库相对路径外，分析与采集实现和修订稿补充文件 S3 一致。

## 与修订稿的对应关系

| 方法或结果 | 代码实现 |
| --- | --- |
| AUC 对并列预测使用平均秩 | `analysis.buffer_study.auc` 使用 `rankdata(..., method="average")`。 |
| 有限 bootstrap 符号尾部量 | `analysis.revision_study.summarize` 使用 `min(1, 2 * (1 + min(N_plus, N_minus)) / (B_valid + 1))`；`analysis.cycle_study.summarize` 共用该实现。零同时计入两侧，非有限抽样剔除但明确报告数量。 |
| 表 4、图 3–4 的百分比区间 | `analysis.revision_study.forest_rows` 和 `analysis.summarize.collect` 共用 `revision_results.json` 中的百分比 bootstrap 区间。 |
| 等价值发行方基准 | `analysis.revision_study.scaled_base_equal` 在合同残值封顶下，将发行方基础残值缩放至待评规则的平均计入残值，与未缩放的原始基础残值不同。 |
| 表 7 的压力下多计入价值 | `analysis.revision_study.equal_stress_release` 匹配的是压力损失总金额（美元），而非损失率。 |
| 基础锚定规则及回款敏感性 | `analysis.buffer_study.RiskBucket` 报告封顶和可行性诊断；`analysis.robustness_study` 对四种带日期的回款口径完整重拟合。 |

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `collection/collect_sec.py` | 从 SEC 申报文件采集所需租赁记录。 |
| `collection/rebuild_from_manifest.py` | 重建确切文件清单并核对事件记录数。 |
| `analysis/analyze.py` | 最初的 GM 品牌分组试验。 |
| `analysis/buffer_study.py` | 跨发行人的主要退租风险缓冲研究。 |
| `analysis/revision_study.py` | 汇总证据、压力分析及梯度提升模型基准。 |
| `analysis/cycle_study.py` | GM 相邻年份之间的检验。 |
| `analysis/robustness_study.py` | 事后回款、稀疏尾部、整簇及模型配对检查。 |
| `analysis/summarize.py` | 根据研究结果生成跨发行人表格和图表。 |
| `analysis/test_invariants.py` | 科学回归检查，含 AUC 并列秩和有限 bootstrap 尾部量。 |

## 运行方式

论文计算已在 Python 3.13.13 及 `requirements-reproducible.txt` 锁定的依赖版本中核验，其他环境未经核验。在仓库根目录执行：

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproducible.txt
.venv/bin/python -m unittest analysis.test_invariants -v
```

仓库**不包含输入数据**。分析模块需要将 JSONL 样本放在 `data/` 下；所需文件名和时间窗口定义在 `analysis/buffer_study.py` 与 `analysis/cycle_study.py` 中。复现论文时，先核对确切的 185 份文件清单，再采集到全新的数据目录。采集前要设置包含本人姓名及联系邮箱的 `SEC_USER_AGENT`：

```bash
.venv/bin/python -m collection.rebuild_from_manifest
export SEC_USER_AGENT='Your Name your.email@example.com'
.venv/bin/python -m collection.rebuild_from_manifest --download
```

第一条命令不发起网络请求。下载命令读取已保存的文件链接，并拒绝覆盖已有数据文件。`collection.collect_sec` 仍可用于发现新文件，但 SEC 当前文件列表未必能返回论文使用的历史选取。补充文件 S3 保存了原始输入哈希和数值结果。

备齐全部 17 个输入数据集后，依次运行：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m analysis.buffer_study --alpha 0.95 --bootstrap 200 --refit-bootstrap 50 --output study_results_alpha0.95.json
.venv/bin/python -m analysis.buffer_study --alpha 0.975 --bootstrap 200 --refit-bootstrap 50 --output study_results_alpha0.975.json
.venv/bin/python -m analysis.buffer_study --issuer gm
.venv/bin/python -m analysis.buffer_study --issuer nissan
.venv/bin/python -m analysis.buffer_study --issuer vw
.venv/bin/python -m analysis.buffer_study --issuer ford
.venv/bin/python -m analysis.revision_study
.venv/bin/python -m analysis.cycle_study
.venv/bin/python -m analysis.robustness_study
.venv/bin/python -m analysis.summarize
```

主分析采用 1,000 次条件于已拟合模型的检验集抽样和 200 次训练重拟合；两种 alpha 敏感性采用 200/50 次抽样，并在 GM 主分析之前运行，使最终 GM 图形使用 alpha 0.99。回款敏感性在每种口径下重新拟合，再做 1,000 次条件抽样，不是 1,000 次训练重拟合。独立的原始 GM 试验可用 `python -m analysis.analyze --train ... --test ... --final ... --output results.json` 运行。

本仓库不收录逐笔租赁输入数据、生成的结果与图表，也不收录论文草稿和生成脚本。结果写入仓库根目录，图表写入 `figures/`。

无定义的稳健性百分比保存为 JSON `null`，无定义 bootstrap 次数明确报告。解释稀疏尾部区间及基础锚定收益反转时，应同时阅读补充文件 S2 和带日期的修订方案。
