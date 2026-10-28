# Processing 01：按新增来源条目数触发记忆处理

| 项目 | 内容 |
| --- | --- |
| 项目 | PowerContext |
| 编号 | `processing-01-count-trigger` |
| 总分 | 20 分 |
| 任务 | 按待处理 Source 条目数自动启动 Memory 与 Topic Memory |
| 提交产物 | 产品仓库 HTTPS URL 和完整 commit SHA |
| 冻结版本 | [PowerContext v1.2.0](https://github.com/oceanbase/powercontext/tree/9db1753434162c0aa7c9f1189a1722e617880a52)，commit `9db1753434162c0aa7c9f1189a1722e617880a52` |

## 一、题目背景

Memory 与 Topic Memory 已能按时间自动处理 Source，也能由调用方显式 flush。低频输入可能需要等待时间条件，高频输入则希望积累足够条目后就启动处理。现有窗口上限控制一次读取多少条，不能表达启动门槛。

数量门槛让输入繁忙的 scope 及时启动处理，少量输入继续由既有时间条件兜底。启动时机与单次处理规模分别可调，调用方不必为每条新输入立即请求生成。

## 二、题目内容

### 2.1 通用要求与范围

验收使用默认 `artifact_processing_role="all"`、全局 supervisor、单写入者 SQLite Runtime。使用原生 worker、本地确定性 OpenAI 兼容 provider 和许可公开会话片段，检查真实 Memory/Topic Memory 的输出、检索与精确引用。无需在线模型；本题不评估生成文本质量。

### 2.2 接口与数据

在 `RuntimeConfig` 中新增以下配置。它们默认均为 `None`，表示关闭数量触发；非空值必须是严格的正整数，拒绝布尔值、字符串、浮点数、零和负数。

```python
memory_source_count_threshold: int | None = None
topic_memory_source_count_threshold: int | None = None
```

公开输入是 [Trace Commons](https://huggingface.co/datasets/trace-commons/agent-traces) 固定 revision `147684b2fdacbbdcd473e92962f565902b2739d5` 的会话文字采样，通过现有 `CaptureSource` 接口写入，不要求解析上游会话格式。离线 provider 根据原生请求携带的证据产生可核对的确定性输出。

### 2.3 关键行为与约束

数量按每个作用域（scope）、每个处理族（family）独立计算：从该处理器最后一次成功提交的 Source 游标（cursor）之后，到当前 journal 末尾的新条目数。待处理条数等于该 scope 的 journal 位置减去该处理族的成功游标序号，尚未提交过游标时按游标 0 计算。首次处理从 cursor 0 开始。计数依据已持久化的 journal 身份，不依据正文长度、token 数或生成结果数。同一身份的重复 capture、已有 Source 的 receipt 信任升级不增加计数。只允许用于 lineage 的服务端 Source 也占一个 journal 条目；它参与启动计数，仍按原有规则从生成证据中排除。

达到门槛后，Runtime 应自动接受并执行一次原有处理调用，无需显式 flush 或等待时间条件。新条目写入后，数量条件在 Supervisor 的下一次检查时判定，该检查周期不超过 1 秒。数量条件与已有时间条件为 OR；任一条件满足即可启动。`None` 保持原有时间行为。显式 `memory.for_scope(...).flush()` 和 `topic_memory.for_scope(...).flush()` 绕过数量门槛。

沿用既有处理器的消费方式。`source_window_limit` 与 `topic_memory_source_window_limit` 仍是每次读取的上限。Memory 成功后按新 cursor 重新判断剩余数量；不足门槛时等待新增输入、时间条件或显式 flush。Topic Memory 已接受的处理批次（wave）继续按原有分窗行为完成，不把门槛重新施加到它的后续窗口。

例如，Memory 门槛为 3、窗口上限为 1，捕获 3 个新条目后自动处理 1 个，剩余 2 个暂不再次数量触发。再捕获 1 个新条目后，待处理数回到 3，自动处理下一窗，剩余 2 个继续等待。不得把一次达到门槛解释成 Memory 必须排空全部输入。

已经接受的调用失败后沿用原有重试，不因重新检查门槛丢弃工作；失败不消耗成功 cursor。关闭并重开同一数据库后，既要保留不足门槛的待处理输入，也要恢复已达到门槛但尚未成功的处理：重启后不足门槛的输入保持待处理，已被接受的工作继续按既有重试与恢复规则处理。成功后不能把已消费的历史条目重新计入门槛。不同 scope 及 Memory、Topic Memory 的 cursor 与门槛互不替代。

处理后的 Source 引用必须能通过公开接口读取，重启后仍可读。实现可选择等价的调度与持久化方式，不要求新增特定 checkpoint、队列或 helper。

## 三、提交方式

提交基于表中冻结版本的产品仓库 HTTPS URL 和完整 40 位 commit SHA，平台须能获取该仓库及 commit。同一仓库和 commit 可以包含多题实现，按所选题目分别验收。

产品内部实现可以自由调整，包括新增、重构或删除源码文件；不设逐题文件白名单。保留本题声明的公共接口及兼容行为。

判题仅加载产品的 `src/`，使用固定依赖和资源配置。产品自有测试、文档及其他附带文件可以随 commit 提交，但不进入判题运行环境，也不能替换平台的测试、输入数据或评分逻辑。产品源码在验收期间只读。

## 四、评分标准

公开测试和平台必测组全部通过得 20 分；任一必测项失败得 0 分，不按子项计部分分。

| 必测行为 | 通过要求 |
| --- | --- |
| 配置与计数 | 严格正整数、默认关闭、按 journal 身份计数，重复 capture 不增加待处理数。 |
| 启动与消费边界 | 数量与时间条件 OR，显式 flush 绕过门槛；窗口仍为单次上限，Memory 按剩余数量重判，Topic 沿用已接受 wave。 |
| 隔离与恢复 | scope/family 独立，成功 cursor 后的输入才计数，不足门槛及已接受工作可在重启后继续，失败不丢输入。 |
| 原生结果 | 真实 worker 产出可检索 Memory/Topic Memory，精确读取及 Source 引用正确，重启后可读。 |

评测不使用模型裁判或在线服务。每次等待自动处理的上限为 45 秒；不足门槛及单窗后的静默观察为 5 秒，故障后的原生生成尝试等待上限为 30 秒。缺失功能不得跳过。

## 五、公开测试与提交前检查

公开测试覆盖配置验证、门槛前后、重复 capture、普通与 lineage 来源的不同写入顺序、单窗边界、scope/family 独立性、时间条件及显式 flush、重启与真实持久化故障恢复。

公开测试、样例及辅助文件随本目录发布。按[公开自测说明](../../README.md)准备 Python 3.12 和固定依赖后，在产品仓库根目录运行：

```bash
make -C competition test-count-trigger
```

提交前检查不足门槛时没有提前输出，达到门槛后无需 flush 即可检索结果；比较单窗处理后的剩余数量、重启前后的引用和真实写入失败后的恢复。公开测试通过后仍须满足全部必测行为。

公开输入包含 11 条会话片段，只采样参与者自身的可见工作叙述，排除思考、图片和工具读取的第三方正文。归属 Trace Commons contributors，公开样例保留 CC BY 4.0 署名；数据卡对汇编、结构和元数据的许可声明不替第三方材料重新授权。
