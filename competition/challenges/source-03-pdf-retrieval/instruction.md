# Source 03：财报 PDF 全文解析与证据页检索

| 项目 | 说明 |
| --- | --- |
| 项目 | PowerContext |
| 编号 | `source-03-pdf-retrieval` |
| 总分 | 50 分 |
| 任务 | 完整解析财报 PDF 的全部物理页，通过自然语言问题检索可精确回查的证据页 |
| 提交产物 | 产品仓库 HTTPS URL 和完整 commit SHA |
| 冻结版本 | [PowerContext v1.2.0](https://github.com/oceanbase/powercontext/tree/9db1753434162c0aa7c9f1189a1722e617880a52)，commit `9db1753434162c0aa7c9f1189a1722e617880a52` |

## 一、题目背景

财报中的检索结果需要能够定位到原文件页面，便于核对数字、表格和相关说明。全文解析负责保留全部页面的可提取内容，问题检索负责从这些页面中找到证据。前者完整，不代表后者一定召回正确页面；遗漏内容或混淆页边界也会影响检索和核对。

PowerContext 已有文本检索和来源回查能力，本题增加完整 PDF 解析及页级来源，把财报接入这套流程。每页记录文档身份、原文件摘要和物理页码，物理页码从 PDF 的第一页开始计数；检索结果通过精确引用回查相应页面。[Haystack 的 PDF 转换文档](https://docs.haystack.deepset.ai/docs/pypdftodocument)展示了文本接入的基本流程；本题使用真实财报和 FinanceBench 问题，分别检验全页覆盖、解析质量、证据页召回及引用准确性。

## 二、题目内容

### 2.1 通用要求

沿用现有 Source Definition、标准文本投影、Runtime 和 scoped Memory 公共接口。scoped Memory 将检索限制在调用方指定的数据范围。共享环境提供锁定的 pypdf；评测全程离线，不调用模型或在线服务。

输入为完整原 PDF，须解析全部物理页，不能以预解析文本替代，也不能只解析证据页。问题检索返回可回查的页级引用；本题不涉及扫描件 OCR，不评价金融计算或答案生成。

候选实现只接收 PDF 字节、文档 ID 和问题，不能读取答案、标注证据、预解析正文、证据页清单或判题资产，不得硬编码文档或问题答案。

### 2.2 接口与数据

在 `powercontext.sources.pdf_pages` 提供以下接口：

```python
class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes

class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: str
    page_number: int
    text: str

def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]: ...

PDF_PAGE_SOURCE_DEFINITION: SourceDefinition

async def search_pdf_pages(
    memory, question: str, limit: int = 10
) -> tuple[MemoryCitation, ...]: ...
```

### 2.3 全文解析和页级 Source

`content` 是完整原 PDF，不能用预解析文本替代，也不能只解析证据页。以物理页为分块边界，按顺序返回所有页，页码从 1 开始，包括空白页。`text` 保存本页可提取的正文，保留文字、表格数字与所属页关系。页面布局空白可以不同，不要求复刻 PDF 视觉排版。本题不涉及扫描件 OCR。共享环境提供锁定的 pypdf。

空白页指本页确实没有可提取文本，此时 `text` 为空字符串；文本提取过程本身报错不属于空白页，而属于无法完整解析。空文件、非 PDF、损坏、读不出任何页（含零页文档）或需要用户提供密码才能读取的文件以 `ValueError` 失败，不返回部分页。带有空密码权限标记、无需用户提供密码即可读取的 PDF 属于合法输入。

文档 ID 不得为空或纯空白。摘要为整个输入 PDF 的 SHA-256 小写十六进制字符串。

Definition 的名称为 `pdf-page`、版本为 `1`，输入类型为 `PDFPageInput`。每页解析为 `CAPTURED` Source，`read` 返回该页输入。Source 身份由 `(document_id, document_sha256, page_number)` 决定，稳定、无碰撞且在任何文档 ID 长度下都不超过 256 字符；文档 ID 过长时实现可以改用其摘要承载身份，只要同一三元组始终得到同一身份、不同三元组不共享身份。PDF 修订后产生新身份，旧引用继续指向原修订。

标准文本投影的 `source_type="pdf-page"`，`source_id` 为 Source `name`，`content` 为 `DOCUMENT_ID\nPAGE_TEXT`。metadata 恰好包含 `document_id`、`document_sha256` 和 `page_number`。调用方用文档摘要找到原 PDF，再用物理页码定位，不能混用财报印刷页码。

### 2.4 真实入库、检索及引用

判题将所有页面 observation 提交至真实 Runtime，并使用固定 pipeline 将其文本放入 Memory。重复提交不增加 Source。关闭并重新打开 Runtime 后，从同一数据划分的文档集合检索；不同文档集合分别建立数据库。

`search_pdf_pages` 接收已打开的 scoped Memory、FinanceBench 原始自然语言问题以及返回上限，三者均可按关键字传入。只允许通过公开 Memory API 搜索，返回最多 `limit` 个指向不同页面的精确 `MemoryCitation`；每个 citation 必须能由 `memory.get` 回查到该页文本及其 Source。允许离线查询处理及重排；不能绕过 Memory 存储创建虚假引用，也不能直接读取数据库、原 PDF 或任何判题资产。判题逐条 `memory.get` 回查文本及 Source，核对完整文档摘要和物理页码。

### 2.5 解析质量

标注证据使用 FinanceBench 原有 `evidence_text` 和 `evidence_text_full_page`；原标注页码从 0 开始，验收转换为物理页码。文本转小写，以正则 `[a-z]+|\d+` 分词，按词频多重集合计交集。

平台还从完整原 PDF 离线冻结每页词项数，检查未被问答标注的页面也有完整输出。每页词项数须落在冻结值的 90% 至 110% 之间；原本无可提取文本的页面仍应为空。空白排版不参与词项统计，不能用只有页码和空正文的占位页替代全文解析。

每条证据的词项召回率为“解析页与证据交集词数 / 证据词数”，不得低于 0.95。每张标注页的全文词项 F1 为“2 × 解析页与标注全文交集词数 / 两者总词数”，不得低于 0.95。两项分别约束证据丢失和整页混入无关文本。

### 2.6 证据页召回

验收以 `limit=20` 检索每个原始问题，完整评测各组的 evidence-page Recall@20 均不得低于 0.70。分子为各问题前 20 条引用命中的不同标注文档页数之和，分母为各问题的不同标注文档页数之和。同一页被多个问题标注时，按各问题分别计数；同一问题的重复标注页只计一次。只有文档和物理页码同时一致才算命中。

## 三、提交方式

提交基于表中冻结版本的产品仓库 HTTPS URL 和完整 40 位 commit SHA，平台须能获取该仓库及 commit。同一仓库和 commit 可以包含多题实现，按所选题目分别验收。

产品内部实现可以自由调整，包括新增、重构或删除源码文件；不设逐题文件白名单。保留本题声明的公共接口及兼容行为。

判题仅加载产品的 `src/`，使用固定依赖和资源配置。产品自有测试、文档及其他附带文件可以随 commit 提交，但不进入判题运行环境，也不能替换平台的测试、输入数据或评分逻辑。产品源码在验收期间只读。

## 四、评分标准

本题共 50 分。全部必测行为通过得 50 分，任一失败得 0 分，不按子项计部分分。

| 必测行为 | 通过条件 |
| --- | --- |
| 完整解析与身份 | 所有物理页按序返回；空白页、合法权限、原子失败（含零页与提取报错）、摘要及修订身份符合契约，身份长度对任意文档 ID 都有界 |
| 持久化与引用 | 全部页面真实入库，重复提交幂等；重启后的返回上限、不同页和精确 citation 正确 |
| 覆盖及质量 | 完整评测各组分别满足每页词项数 90% 至 110%、标注证据召回 0.95、标注整页 F1 0.95 及 evidence-page Recall@20 0.70 |

## 五、公开测试与提交前检查

公开测试使用完整的 PepsiCo 2022 年 10-K PDF，包含 503 个物理页和 5 个原始问题。

公开测试、样例及辅助文件随本目录发布。按[公开自测说明](../../README.md)准备 Python 3.12 和固定依赖后，在产品仓库根目录运行：

```bash
make -C competition test-pdf
```

公开自测完整解析 503 页、持久化全部页面，并检查 5 个原始问题的精确 citation。完整评测还会检查题面公布的质量门槛。

提交前检查：

- 检查全部物理页、空白页顺序、合法空密码权限和解析失败的原子性；空文件、非 PDF、零页、需用户口令和正文提取报错都整篇失败。
- 核对完整字节摘要、页级修订身份、重启后的持久化及精确引用。
- 用超长文档 ID 检查身份仍不超过 256 字符，且同一三元组重复解析得到同一身份。
- 检查 limit 和不同页要求，回查正文、文档摘要及物理页码。
- 用公开样例检查全页覆盖和引用，并确认实现满足题面公布的解析与检索质量要求。
- 核对仓库 URL、完整 commit 及公开接口。公开自测通过后仍须满足全部必测行为。

正式评测使用 [FinanceBench](https://github.com/patronus-ai/financebench/tree/cc39aeb4afdf33909ee1412188bf89035950c2eb) 固定 revision `cc39aeb4afdf33909ee1412188bf89035950c2eb` 的 5 份完整财报、1151 页和 24 个原始问题。公开自测提供其中的完整 PepsiCo PDF 和五个问题，文件不裁页或重编码。数据卡声明 CC BY-NC 4.0，原公司财报保留各自权利。公开样例不包含答案、解释、证据文本或证据页码。损坏、空白页和修订另用明确构造的 PDF 覆盖。
