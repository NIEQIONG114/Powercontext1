# Retrieval 02：seekdb 混合检索的单 SQL 候选查询

| 项目 | 说明 |
| --- | --- |
| 项目 | PowerContext |
| 编号 | `retrieval-02-hybrid-search` |
| 总分 | 20 分 |
| 任务 | 用一条 SQL 获取 seekdb 全文与向量候选，保持过滤、当前版本引用和检索质量 |
| 提交产物 | 产品仓库 HTTPS URL 和完整 commit SHA |
| 冻结版本 | [PowerContext v1.2.0](https://github.com/oceanbase/powercontext/tree/9db1753434162c0aa7c9f1189a1722e617880a52)，commit `9db1753434162c0aa7c9f1189a1722e617880a52` |

## 一、题目背景

全文检索可以定位关键词和术语，向量检索可以补充表达不同但语义相关的内容。两路检索各自选出可能相关的记录，作为后续融合排序的候选。PowerContext 的 seekdb 路径已使用原生全文索引和向量索引，当前通过两次 SQL 查询分别取得这些候选。

本题利用同一数据系统的原生查询能力，把两路候选及其过滤条件放进一条 SQL，减少应用层分别发起和组织查询的工作。[seekdb 的官方项目说明](https://github.com/oceanbase/seekdb)展示了这种统一表达方式。取得候选后，应用层仍可完成融合和精确正文读取。验收既检查候选查询次数和两类索引的实际使用，也检查作用域隔离、当前版本引用，以及逐查询的检索质量是否保持。

## 二、题目内容

### 2.1 通用要求与范围

改造现有 seekdb 后端的候选查询，复用已有 schema、产品传入的 `AsyncConnection` 和共享环境中的 `pylibseekdb==1.3.0.post4`。不得启动独立查询服务或替换数据库、查询引擎。

`mode="hybrid"` 的一次候选检索须恰好执行一条 seekdb statement，同时包含全文与向量候选。该限制只针对候选检索：候选之前的当前 head 校验与向量完整性校验沿用现有 backend 行为，不计入候选 SQL 数量；候选之后的精确正文读取仍遵循产品流程。不能先执行两条候选 SQL，再在应用层合并。

FTS-only 和 vector-only 的公开行为保持不变，embedding 不完整时的处理不属于本题范围。禁止按固定文档、query、主题或测试顺序硬编码结果。

### 2.2 接口与数据

沿用冻结 schema 的 `pc_memory_entry_heads` 和 `pc_memory_vector_entries`。结果可使用 `MemorySearchChannels` 或等价的后端中立结构，保留 `memory_artifact_id`、`head_revision`、`entry_id` 与 `entry_version_id`。单 statement 可返回两路有序候选后再融合，也可采用其他符合契约的结果结构和融合方式；候选 SQL 可以同时带回正文，也可以只返回身份再由产品流程精确读取，两种做法都不改变候选数量与两路的原生次序。

质量输入包含 173 篇真实文档、20 条原始 query，使用固定的 384 维单位向量和 L2 `EmbeddingProfile`，返回上限为 `limit=10`。人工 qrels 只用于评分。

### 2.3 关键行为与约束

| 行为 | 要求 |
| --- | --- |
| 原生索引 | 实际 SQL 的 EXPLAIN 同时显示 `TEXT RETRIEVAL SCAN` 和 `VECTOR INDEX`，不能全表扫描后在 Python 计算匹配或距离 |
| 数据库内过滤 | scope、Memory artifact、tag filter 与 candidate limit 在数据库内生效 |
| 候选上限 | `candidate_limit` 对两路分别生效，全文与向量各取至多 `candidate_limit` 条候选。 |
| 候选顺序 | 两路候选各自保持原生索引给出的顺序，融合、截断与正文读取在应用层完成；候选 SQL 内的连接与排序不得打乱两路的原生次序。 |
| 当前 head | 沿用现有 backend 校验；历史 head 按现有 capability contract 拒绝 |
| 增量投影 | 未修改条目的 projection 可保留旧 `head_revision`，不能因此排除仍活跃的条目；返回引用使用请求中已验证的当前 Memory head |
| 状态变化 | 追加、正文修订、停用和恢复后，索引、正文及当前状态一致 |
| 命中有效性 | 每个命中属于当前 scope/head，引用当前有效 entry version，正文与该版本逐字一致，不能重复返回同一身份 |
| 通道标识 | `matched_by` 非空，标识取自 `fts`、`vector` |
| 稳定性 | 同一实现路径的重复查询返回稳定的身份与顺序；单通道查询正常 |
| 逐查询质量 | 同库、同 scope/head/query/embedding 条件下，每条 query 的 Recall@10 和 MRR@10 均不低于冻结双查询路径 |

允许改变融合算法和同分处理。与冻结路径的完整身份、正文、通道和排序 tuple 是否相同仅用于观察，不要求完整结果相同。

## 三、提交方式

提交基于表中冻结版本的产品仓库 HTTPS URL 和完整 40 位 commit SHA，平台须能获取该仓库及 commit。同一仓库和 commit 可以包含多题实现，按所选题目分别验收。

产品内部实现可以自由调整，包括新增、重构或删除源码文件；不设逐题文件白名单。保留本题声明的公共接口及兼容行为。

判题仅加载产品的 `src/`，使用固定依赖和资源配置。产品自有测试、文档及其他附带文件可以随 commit 提交，但不进入判题运行环境，也不能替换平台的测试、输入数据或评分逻辑。产品源码在验收期间只读。

## 四、评分标准

全部功能、质量和成本要求通过得 20 分，任一要求失败得 0 分。功能门通过后才执行质量与成本测量；缺失接口、跳过必测项或空收集均不通过。

| 验收类别 | 通过条件 |
| --- | --- |
| 功能 | 原生索引、过滤、当前版本、状态变化、引用、重复查询稳定性和单通道行为符合契约 |
| 候选访问成本 | 每条 hybrid query 恰好执行一条包含全文与向量候选的 SQL；当前 head 校验与向量完整性校验不计入 |
| 命中有效性 | 20 个 query、三个独立 scope 共 60 次查询，所有命中的当前版本、正文及通道标识有效，且没有重复身份 |
| 逐查询质量 | 与同库冻结双查询路径相比，每条 query 的 Recall@10、MRR@10 均不降低 |
| 平均召回 | 60 次查询的 `evidence_recall_at_10 >= 0.95` |
| 延迟 | 完整 `Memory.search` 调用的 `p50_latency_ms <= 500` |

每对查询共用同一数据库中的 scope、head、embedding 和已入库身份。两条路径各预热一次，交替先运行候选或对照。每条 query 的 Recall@10 为前 10 个结果中唯一相关 document ID 数除以该 query 的 qrel document ID 数，再对 60 次查询求算术平均。MRR@10 为首个相关文档的倒数排名，没有相关文档时为 0；逐 query 均须满足不退化条件，绝对平均 MRR 只报告为观测值。

p50 是预热后 60 次完整候选 `Memory.search` 调用延迟的中位数。完整排序一致率和同库延迟比也作为观测值，不是通过门槛。500 ms 是回归上限，单 SQL 不代表本题已证明端到端加速。

## 五、公开测试与提交前检查

公开测试检查原生混合检索、过滤和引用身份，并提供 SciFact 的一条 claim、八篇论文摘要及公开查询配置。

公开测试、样例及辅助文件随本目录发布。按[公开自测说明](../../README.md)准备 Python 3.12 和固定依赖后，在产品仓库根目录运行：

```bash
make -C competition test-hybrid
```

提交前检查：

- 一条候选 SQL 实际使用两类索引，过滤和 candidate limit 正确。
- 旧投影中的活跃条目仍可检索，修订、停用和恢复后的引用指向当前版本。
- 每个命中的正文、身份和通道有效，重复查询稳定且无重复身份。
- 单通道行为不变，仓库 URL 和完整 commit 可获取，公开接口可用。

公开样例通过不代替真实语料上的逐查询质量及完整搜索延迟验收。

公开样例来自 [SciFact](https://github.com/allenai/scifact/tree/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e)，固定 commit 为 `68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`。公开样例包含一条 claim 和八篇论文摘要，不包含相关性标签或 evidence 区间。claims 与 evidence 标注采用 CC BY 4.0，论文摘要遵循 S2ORC 的 ODC-By 1.0，公开样例保留署名和对应许可。
