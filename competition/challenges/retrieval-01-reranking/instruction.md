# Retrieval 01：HTTP 重排服务接入与跨批次排序

| 项目 | 说明 |
| --- | --- |
| 项目 | PowerContext |
| 编号 | `retrieval-01-reranking` |
| 总分 | 20 分 |
| 任务 | 接入 HTTP 评分服务，按批次上限请求、跨批次稳定排序，并保留原候选身份和实际请求数 |
| 提交产物 | 产品仓库 HTTPS URL 和完整 commit SHA |
| 冻结版本 | [PowerContext v1.2.0](https://github.com/oceanbase/powercontext/tree/9db1753434162c0aa7c9f1189a1722e617880a52)，commit `9db1753434162c0aa7c9f1189a1722e617880a52` |

## 一、题目背景

初步检索得到候选后，重排服务可以结合查询重新判断候选的相关程度。PowerContext 已有重排接口和 LLM 实现，通过候选序号选择原记录，保留条目及版本身份。本题增加 HTTP 评分服务适配器，让真实 Memory 搜索使用服务分数进行选择。

重排的输入是 Memory 搜索粗排后的候选池：重排只决定从这个池里选哪些候选、按什么顺序返回，不扩大候选池，也不改写候选正文。

服务的批次上限要求客户端拆分请求，最终结果仍须由完整候选池的分数统一决定。[Cohere Rerank](https://docs.cohere.com/docs/rerank-overview)展示了查询与候选评分的使用方式；本题采用给定的离线协议，交付的是分批请求、响应校验、跨批次排序与身份映射，不要求训练排名模型或生成答案。

以查询「数据库连接超时怎么排查」为例，下面的候选和分数均为说明协议的合成示例。分数由服务提供，四个候选的原始 `rank` 为 1 至 4，每批最多两个；一种分批如下：

| 原始 rank | 候选内容 | 请求批次 | 服务返回的示例分数 |
| --- | --- | --- | --- |
| 1 | 数据库备份计划 | 第一批 | 0.10 |
| 2 | 客户端连接超时参数说明 | 第一批 | 0.90 |
| 3 | 连接池耗尽排查记录 | 第二批 | 0.95 |
| 4 | 网络连通性检查方法 | 第二批 | 0.90 |

取前两个结果应得到 `rank=3, 2`：先按分数降序，分数相同按原始 rank 升序，因此 rank 2 排在 rank 4 前。实际请求数为 2，返回结果仍引用原 Memory 条目和版本。这样，第二批的高分候选能进入最终结果，调用方也能沿引用核对原记录。正式评测使用 SciFact 输入核对服务分数决定的排序及下文的质量门槛。

## 二、题目内容

### 2.1 通用要求与范围

在 `powercontext.builtin.artifacts.memory.http_reranker` 实现 `HTTPMemoryReranker`，满足现有 `MemoryReranker` 契约，并能通过 `memory_reranker` 接入 Runtime。请求使用调用方提供的 `httpx.AsyncClient`；验收不访问在线服务。

重排只改变候选的选择和顺序，不改写正文、Memory artifact、entry ID 或 entry version ID。分批后仍使用完整候选池中的原始序号，将服务分数对应回原候选；实际请求数通过 `InferenceUsage.requests` 报告。实现可以自行决定批次调度、批次发送顺序与批内候选顺序，串行或并发发送皆可，最终结果仍须满足下文的全局排序和身份要求。

### 2.2 接口与数据

| 接口 | 契约 |
| --- | --- |
| 构造 | `HTTPMemoryReranker(endpoint: str, *, client: httpx.AsyncClient, batch_size: int = 32, policy_id: str = "http-cross-encoder-v1")` |
| 重排 | 异步 `rerank(query, candidates, limit)`，返回现有 `MemoryRerankDecision` |
| 请求 | POST `endpoint`，JSON 正文包含 `query` 与 `candidates`；每个 candidate 包含 `rank` 和 `text` |
| 响应 | JSON 恰好包含 `scores`；每个 score 记录恰好包含 `rank` 和 `score` |
| 策略标识 | `policy_id` 按构造参数暴露为实例属性，供 Memory 搜索记录本次重排策略 |

```json
{"query":"sensor calibration","candidates":[{"rank":1,"text":"Calibration record."}]}
```

```json
{"scores":[{"rank":1,"score":0.75}]}
```

`rank` 是候选在完整输入中的一基整数序号，分批后不能重新编号。每条请求正文的 `text` 须是该 rank 原候选的正文，原样发送，不得截断或改写。`score` 必须是有限数值，不接受 NaN 或 Infinity。

返回值中的 `selected_ranks` 恰好包含 `limit` 个互不相同的全局 rank，按最终顺序排列。由于 `limit` 不超过候选条数，结果不会因候选用尽而变短。

### 2.3 关键行为与约束

| 行为 | 要求 |
| --- | --- |
| 分批与覆盖 | 每批最多 `batch_size` 个候选，每个全局 rank 恰好发送一次；正常完成时请求数为 `ceil(len(candidates) / batch_size)` |
| 响应校验 | 响应可乱序，但必须恰好覆盖本批所有 rank 一次；缺失、重复、本批之外的 rank 和额外字段均无效 |
| 全局选择 | 按 score 降序选择前 `limit` 个 rank，平分时按原始全局 rank 升序；各批结果须参与同一次全局选择，跨批次同分也按全局 rank 决定先后 |
| 参数边界 | 候选非空，`batch_size > 0`（构造时即校验），`1 <= limit <= len(candidates)`；非法输入以 `ValueError` 失败 |
| 非 2xx 响应 | 先按状态码失败，抛出 httpx 状态错误，不解析响应体；即使正文看似合法也不能当作分数使用 |
| 网络错误 | 连接、超时等传输错误保留 httpx 原始异常，不做包装 |
| 响应格式错误 | 非法 JSON、响应 schema 不符、`score` 非有限数值时抛出 `ValueError`，不限制错误文案 |
| 部分失败 | 任一批次失败即整个调用失败，不返回部分结果；调用方重试时重新发送全部批次 |
| usage | `InferenceUsage.requests` 等于实际发出的 HTTP 请求数；服务不报告 token，`input_tokens` 与 `output_tokens` 留空 |
| 原生集成 | Memory 搜索使用重排决策，返回身份与对应原候选一致 |

## 三、提交方式

提交基于表中冻结版本的产品仓库 HTTPS URL 和完整 40 位 commit SHA，平台须能获取该仓库及 commit。同一仓库和 commit 可以包含多题实现，按所选题目分别验收。

产品内部实现可以自由调整，包括新增、重构或删除源码文件；不设逐题文件白名单。保留本题声明的公共接口及兼容行为。

判题仅加载产品的 `src/`，使用固定依赖和资源配置。产品自有测试、文档及其他附带文件可以随 commit 提交，但不进入判题运行环境，也不能替换平台的测试、输入数据或评分逻辑。产品源码在验收期间只读。

## 四、评分标准

全部功能与质量要求通过得 20 分，任一要求失败得 0 分。缺失接口、跳过必测项或空收集均不通过。

| 验收类别 | 通过条件 |
| --- | --- |
| 功能 | 请求响应校验、批次上限、覆盖、全局排序、状态码与异常传播、usage 和真实 Memory 集成符合契约 |
| 质量输入 | 173 篇 SciFact 原始摘要、20 条原始 claim，固定 384 维 MiniLM 离线向量；`batch_size=7`，返回前 10 个结果 |
| 排序 | 返回结果与服务分数决定的全局稳定排序一致 |
| Recall@10 | 20 个 query 的平均值至少为 0.95 |
| MRR@10 | 20 个 query 的平均值至少为 0.80 |

离线服务以冻结文档和 query 向量的余弦分数评分，qrels 不参与服务打分。每个 query 的 Recall@10 是前 10 个结果中唯一相关文档数除以该 query 的 qrels 文档数；MRR@10 是首个相关文档的倒数排名，无相关结果时为 0。两项都对 20 个 query 求算术平均。批次顺序不参与判分。

## 五、公开测试与提交前检查

公开测试检查 HTTP 批次、响应校验、全局排序与原生 Memory 集成，并提供 SciFact 的一条 claim 和八篇论文摘要。

公开测试、样例及辅助文件随本目录发布。按[公开自测说明](../../README.md)准备 Python 3.12 和固定依赖后，在产品仓库根目录运行：

```bash
make -C competition test-rerank
```

提交前检查：

- 分批或重新安排发送顺序后，全局 rank、正文及请求计数仍正确。
- 乱序响应、跨批同分及非法 schema 的结果和异常符合契约。
- 服务返回非 2xx 时即使正文可解析也以 httpx 状态错误失败，且不产生部分结果。
- 原生 Memory 搜索保持原 entry version 身份，usage 与实际请求一致。
- 仓库 URL 和完整 commit 可获取，公开接口可用。

公开测试通过不代替正式质量验收。

公开样例来自 [SciFact](https://github.com/allenai/scifact/tree/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e)，固定 commit 为 `68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`。公开样例包含一条 claim 和八篇论文摘要，不包含相关性标签或 evidence 区间。claims 与 evidence 标注采用 CC BY 4.0，论文摘要遵循 S2ORC 的 ODC-By 1.0，公开样例保留署名和对应许可。
