# Python-teamwork

《Python 数据分析编程》小组项目：第六题中文垃圾邮件分类。

执行步骤、三人分工、时间安排和提交要求见 [项目计划](项目计划.md)。

十步总览、当前状态和下一步任务见 [执行清单](执行清单.md)。

阶段完成情况见 [进度记录](进度记录.md)。每个阶段的成果与 Markdown 记录一同提交并推送，具体约定见 [AGENTS.md](AGENTS.md)。

**当前范围**

已完成前六步：环境、索引、全量清洗、去重冻结划分、EDA/分词/TF-IDF，以及基线和两种模型比较。保留 33,988 封代表，训练/验证/测试为 23,792 / 5,099 / 5,097，三套特征共用训练侧学到的 50,000 维词表。七次验证实验选出 LR C=2（验证 macro-F1 98.78%）；下一步做 HTML 清洗对照，最终测试分类成绩尚未计算。公开聚合证据与图表，正文、去重账本、完整词频、矩阵、拟合模型与逐封预测留在本地。

建议主线：邮件解析清洗 → 内容去重与固定划分 → 探索分析 → jieba 与 TF-IDF → 朴素贝叶斯及逻辑回归 → 评估解释。

**在 Windows 建立环境**

使用 64 位 Python 3.12。以下 PowerShell 命令在仓库根目录执行；新电脑需先安装相应 Python 并完成 GitHub 协作者配置。

```powershell
git clone https://github.com/25121596/Python-teamwork.git
cd Python-teamwork
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path config.local.json)) { Copy-Item config.example.json config.local.json }
.\.venv\Scripts\python.exe scripts\check_environment.py
```

若没有 `py` 启动器，使用已安装的 Python 3.12 解释器完整路径执行 `-m venv .venv`。不要覆盖已经建好的虚拟环境或本地配置。当前电脑的项目虚拟环境已建立，可直接运行最后一条检查命令。

不必执行激活脚本，直接调用 `.venv\Scripts\python.exe` 即可，避免 PowerShell 执行策略以及 `pip` 指向其他解释器的问题。

`requirements.in` 列出直接依赖；`requirements.txt` 固定本阶段实际安装并验证的版本，按 Python 3.12、Windows 64 位环境记录。更新依赖时在独立环境重新安装、检查并更新版本清单，不能只改版本号。

**配置本地数据位置**

数据来源为课程提供的 TREC 2006 中文邮件包，原始邮件必须保存在仓库目录之外，通过配置引用。环境脚本会检查这一约束，避免无扩展名邮件被误提交。数据根目录应包含：

```text
trec06c/
  full/index
  delay/index
  data/
```

把 `config.example.json` 复制为 `config.local.json`，修改 `data_root`。示例目录对应“仓库与数据集目录放在同一个父目录”的课程资料布局；其他电脑可填写自己的位置。

```json
{
  "data_root": "../数据集/最终数据集/题6_中文垃圾邮件_trec06c/trec06c"
}
```

上面仅展示需要修改的字段，请保留示例配置的其余字段。JSON 路径使用正斜杠；绝对路径也可用。相对 `data_root` 以配置文件所在目录为基准，不依赖启动脚本的当前目录。

也可通过当前 PowerShell 会话中的 `TREC06C_ROOT` 覆盖位置：

```powershell
$env:TREC06C_ROOT = 'D:/课程数据/trec06c'
.\.venv\Scripts\python.exe scripts\check_environment.py
```

主索引固定为 `full/index`。其中的 `../data/...` 相对于 `full` 目录解析，`delay/index` 不能作为独立测试集。本地配置被 Git 忽略，不上传个人路径。

**环境检查与实际运行顺序**

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\check_environment.py
```

检查脚本验证虚拟环境、Python 版本、9 个依赖导入、配置、数据目录、索引可读性与首条索引文件是否存在；成功退出码为 0，失败为 1。仅检查首条记录以确认路径链路，不代替下一阶段的全量索引审计。

安全摘要写入 `artifacts/audit/environment_check.json`，不含个人绝对路径、收发件地址或邮件正文。可通过 `--config` 和 `--output` 指定其他配置或输出；相对参数均以项目根目录为基准。

目前可执行入口包括环境检查、索引审计、邮件解析清洗、去重划分、EDA/特征，以及验证集模型比较与核验。后续为清洗对照和最终评估。

**第二阶段索引审计**

```powershell
.\.venv\Scripts\python.exe scripts\audit_index.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

审计以 `full/index` 为唯一入口，只读索引和文件系统元数据，不读取邮件正文。检查格式、标签、路径安全、重复与冲突、缺失/非普通/空文件以及索引与目录覆盖情况。辅助文件即使被索引引用，也不会进入有效邮件或开发样本。

| 产物 | 用途与提交范围 |
|---|---|
| `artifacts/processed/mail_manifest.csv` | 全量稳定样本清单，仅保存在本地，被 Git 忽略 |
| `artifacts/processed/index_issues.jsonl` | 异常明细，仅保存在本地；当前审计没有异常，为空文件 |
| `artifacts/audit/index_summary.json` | 聚合数量、文件大小、审计状态与抽样说明，上传仓库 |
| `artifacts/splits/development_sample.csv` | 固定 400 封开发样本的相对路径与标签元数据，上传仓库 |
| `artifacts/audit/stage02_validation.json` | 本阶段实际验证证据，上传仓库 |

默认开发样本为 ham/spam 各 200 封，种子来自配置的 `random_state=42`；先按规范化路径排序，再抽样。它用于覆盖两类解析场景，**不代表原始类别比例，也不是训练/验证/测试划分**。可用 `--development-size` 改变大小；样本不足时按可用量缩减。

`mail_id` 按相对语料根的规范化路径生成 SHA256，与个人绝对路径和索引排列顺序无关；它不是正文内容哈希，不能据此宣称已经完成内容去重。清单中的 `relative_path` 相对于 `data_root`，后续读取方式为 `data_root / relative_path`。

同路径同标签重复只保留一个 canonical 样本并统计额外行；标签冲突整体隔离。索引问题、空文件、覆盖异常等会使审计返回 1；全部通过返回 0；配置/访问等致命错误返回 2。返回 2 时旧产物可能保留，应先排除错误再重新运行。审计输出固定在项目 `artifacts` 下，确保全量清单与异常明细受到忽略规则保护。

当前真实统计和测试限制见 [第二阶段记录](docs/stages/02_索引审计与样本清单.md)。

**第三阶段邮件解析与清洗**

```powershell
.\.venv\Scripts\python.exe scripts\clean_mail.py --scope development
.\.venv\Scripts\python.exe scripts\check_cleaning.py --scope development
.\.venv\Scripts\python.exe scripts\clean_mail.py --scope full
.\.venv\Scripts\python.exe scripts\check_cleaning.py --scope full
```

先使用固定开发样本调试和核验，再运行全量。启动前核对源索引 checksum、canonical 清单、标签、稳定 ID 和固定开发抽样结果；逐封读取异常会被记录，其余邮件继续处理。

解析器先解传输编码，再处理声明 charset、HTML meta 与 UTF-8/GB18030 等回退。明确可读的错误 base64 声明可以有记录地回退为原 payload；未知传输编码、明显二进制或未拆分 MIME 块不作为正文。multipart/alternative 只选一个有效表示，related 只选根部件；跳过附件和转发的整封子邮件。

主题与正文分开清洗，HTML 去 script/style、解实体并保留段落与锚文本；NFKC、英文小写、控制字符清理与常见 URL/邮箱/电话号码归一化使用同一函数。手机号及常见分隔式电话号码可掩码，但这不等于全面匿名化。

| 产物 | 用途与提交范围 |
|---|---|
| `artifacts/processed/cleaned_development.jsonl` | 开发样本的解码/清洗结果，仅本地保存 |
| `artifacts/processed/cleaned_mail.jsonl` | 全量逐封状态、清洗文本、解码文本与保留标签的对照文本，仅本地保存 |
| `artifacts/processed/*cleaning_issues.jsonl` | 非 ok 状态元数据日志，包含可用的 recovered，不等于隔离名单，仅本地保存 |
| `artifacts/audit/*cleaning_summary.json` | 聚合状态、类别保留率、编码、缺陷和字符质量统计，上传仓库 |
| `artifacts/audit/*cleaning_validation.json` | 缓存校验、账目与规则复核汇总，上传仓库 |
| `artifacts/processed/*cleaning_rule_review.json` | 逐封规则复核名单，仅本地保存 |
| `artifacts/audit/stage03_validation.json` | 测试、开发重跑一致性、开发/全量一致性及 Git 忽略验证，上传仓库 |
| `docs/清洗示例.md` | 明确标注的人工演示输入，不含真实邮件摘录 |

互斥状态为 `ok`、`recovered`、`subject_only`、`empty`、`quarantined`、`error`。只有主题可用的邮件保留；无有效主题/正文的邮件隔离。单封运行错误返回并记录 error，全批仍输出账目，但程序退出码为 1；配置等致命错误返回 2。`summary.passed`仅表示输入处理完整且没有 error，不表示每封都有可用文本。

`has_html`指所选有效正文是否经过 HTML 提取；`has_html_part_anywhere`指任意 MIME 部件是否声明 HTML，两者不能混作正文 HTML 率。attempt 字符统计覆盖所考察主题段/正文候选的解码结果（含被拒候选，不含内部 codec 探测），retained 统计只按最终 `text_clean` 计算。

替换符比例为 0 不能证明没有语义乱码；private-use 字符等会标为质量警告。缓存中 `body_decoded`、`text_unstripped_html` 仍可能含个人信息，只供本地后续对照，不能当作已脱敏内容公开。真实数据复核与已知局限见 [第三阶段记录](docs/stages/03_邮件解析与全量清洗.md)，机制演示见 [人工清洗示例](docs/清洗示例.md)。

**第四阶段去重与冻结划分**

```powershell
.\.venv\Scripts\python.exe scripts\split_data.py
.\.venv\Scripts\python.exe scripts\check_splits.py
```

入口先核对清洗缓存、canonical 清单、固定开发样本和解析源码，并重新读取原始邮件核对字节签名。对最终主题＋正文完全相同且同标签的邮件只保留一个确定代表；跨标签的原始字节或最终文本精确重复键整组隔离，不用多数标签覆盖冲突。

共享原始字节、最终文本、有效正文或第七阶段 HTML 对照文本的邮件组成连通分组。正文相同而标题不同的邮件保留各自代表，但整组进入同一集合；仅符号正文不建立正文关系。开发样本触及的整个分组强制进入训练侧，包括被去重的开发样本。

本版冻结契约为 70%/15%/15%、种子 42，配置必须与这一版本相符；按保留代表的两类数量做整组近似分层，不拆组凑比例。生成器会在写入前调用独立图遍历检查候选结果。重复运行必须与已冻结产物一致，否则返回 2 并保留旧版；不能在模型比较期间反复换种子挑划分。

| 产物 | 用途与范围 |
|---|---|
| `artifacts/splits/train.csv`、`validation.csv`、`test.csv` | 固定代表的 ID、标签和集合，上传仓库；不含内容、路径或内容签名 |
| `artifacts/processed/dedup_ledger.csv` | 全量处置、重复映射、内容签名、分组与开发接触标记，仅本地保存 |
| `artifacts/audit/split_summary.json` | 去重、分组、类别、实际比例、来源与产物指纹等聚合信息 |
| `artifacts/audit/split_validation.json` | 独立重建内容关系、代表规则和开发边界的聚合验证 |
| `artifacts/audit/stage04_validation.json` | 测试、重跑、独立复核和忽略规则证据 |

后续脚本按固定 CSV 的 `mail_id` 与本地清洗缓存关联文本，不能把 CSV 的 ID、标签或集合字段作为模型特征。精确去重无法排除所有近重复广告模板；去重后的类别比例也会改变，评估对象是本语料的不同规范化文本。

冻结文件使用 `.gitattributes` 固定 LF 换行，避免 Windows 检出时改变字节指纹。实际结果见 [第四阶段记录](docs/stages/04_去重与冻结数据划分.md)。

**第五阶段 EDA 与文本特征**

```powershell
.\.venv\Scripts\python.exe scripts\analyze_corpus.py
.\.venv\Scripts\python.exe scripts\build_features.py
.\.venv\Scripts\python.exe scripts\check_features.py
```

所有入口通过验证过的 dataset loader 按冻结 CSV 的 ID 对齐文本和标签。EDA 为描述性聚合，包含全语料与固定集合的类别、长度和质量标记；词覆盖统计只用训练代表。公开词图仅展示预设通用词表中的高频词，完整词频表留在本地。

jieba 采用精确模式、HMM=False，固定停用词与词典指纹；保留非停用词中文单字、数字和联系字段占位符。TF-IDF 只在 train 拟合，min_df=2、max_features=50000、sublinear_tf=True、L2、float64；验证/测试只转换。独立脚本重算训练 DF/IDF，并在新进程重载管线核验保存矩阵。

| 产物 | 范围 |
|---|---|
| `artifacts/figures/stage05_*.png` | 五张可直接使用的统计图与训练通用词覆盖图 |
| `artifacts/audit/eda_summary.json`、`feature_summary.json`、`feature_validation.json`、`stage05_validation.json` | 聚合统计、配置、来源与验收记录 |
| `artifacts/processed/features/` | 本地 CSR 矩阵、ID/标签行序与完整词频，被 Git 忽略 |
| `artifacts/models/tfidf_pipeline.joblib` | 本地已拟合特征管线，被 Git 忽略 |
| `config/stopwords_zh.txt`、`public_terms_zh.txt` | 公开的预设静态规则，不从 heldout 推导 |

`feature_summary.passed` 表示构建完成；独立验收以 `feature_validation` 为准。异常时旧产物可能仍存在，使用缓存前必须核对来源与指纹。手动重载 joblib 需要本仓库 `src` 模块路径与固定依赖。

图表解释、实际维度与检查结果见 [第五阶段记录](docs/stages/05_探索分析与文本特征.md)，人工示例与公式见 [分词与特征说明](docs/分词与特征说明.md)。

**第六阶段基线与模型比较**

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
.\.venv\Scripts\python.exe scripts\check_models.py
```

沿用第五步已验证的训练/验证 CSR 特征，先运行多数类、默认 NB 与 LR，再完成 NB alpha=0.1/0.5/1 和 LR C=0.5/1/2 的有限比较。分类器只拟合训练标签，验证集按 macro-F1、正常邮件误报率、spam 召回率的预设顺序选候选，不加载最终测试矩阵或计算其分类指标。

本轮候选为 LR C=2，NB 类内最佳为 alpha=0.1。逐封验证预测和七个分类器、候选完整管线保存在忽略目录；公开 [验证实验表](artifacts/metrics/stage06_validation.csv)、[配置与指标](artifacts/metrics/model_selection.json)、[独立验收](artifacts/audit/model_validation.json) 和比较图。独立脚本新进程重载、按训练集重训、重算验证指标和候选排序；`model_selection.passed` 仅表示构建完成。

复现顺序为第 2—5 步、模型训练、模型独立检查。失败时先解决来源或指纹问题再重跑，旧产物不可直接视为当前有效结果。模型采用固定依赖与单线程限制；耗时仅是本机运行记录。详见 [第六阶段记录](docs/stages/06_基线与两种模型比较.md)。第七步先完成清洗对照，第八步再报告最终测试分类成绩。

**Notebook 使用**

已安装 `ipykernel`。在现有 Notebook 或编辑器中选择本仓库的 `.venv` 解释器。如需在现有 Jupyter 中注册专用内核，可手动执行：

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name python-teamwork --display-name 'Python teamwork (.venv)'
```

本阶段不启动 Jupyter 服务，也未自动修改全局内核列表。Notebook 第一格建议检查 `sys.executable`，确认选中本项目环境。

**项目目录**

```text
src/spam_classifier/    索引、解析、划分、特征与模型模块
scripts/                各阶段执行入口和独立核验
tests/                  人工微型语料的边界测试
notebooks/              后续探索和展示过程
artifacts/audit/        聚合质量与环境记录
artifacts/processed/    本地邮件文本缓存，Git 忽略
artifacts/splits/       开发样本与正式冻结划分元数据
artifacts/metrics/      聚合验证模型指标与配置
artifacts/figures/      EDA、词覆盖与验证比较图
artifacts/models/       本地模型缓存，Git 忽略
deliverables/           后续 Word、PPT 与汇报材料
docs/stages/            每阶段的 Markdown 记录
```

空目录以 `.gitkeep` 保留，目录存在不代表已产出数据或模型。原始数据、压缩包、清洗正文、模型缓存、`.venv`、`.env` 和临时日志均不上传。

**成员与协作**

人员、组号和组长由组员确认并填写 [成员分工](docs/成员分工.md)。A/B/C 是建议职责，不代表实际姓名或已完成课程报名。

各人用自己的 Git 身份提交实际成果。成员取得仓库后分别建立环境并记录检查结果；当前电脑的验证不能代替其他两位成员的环境验收。

**本阶段记录**

环境准备的命令、实际版本、检查证据和待确认事项见 [第一阶段记录](docs/stages/01_项目环境与目录.md)。
