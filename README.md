# Python-teamwork

《Python 数据分析编程》小组项目：第六题中文垃圾邮件分类。

执行步骤、三人分工、时间安排和提交要求见 [项目计划](项目计划.md)。

阶段完成情况见 [进度记录](进度记录.md)。每个阶段的成果与 Markdown 记录一同提交并推送，具体约定见 [AGENTS.md](AGENTS.md)。

**当前范围**

已完成独立 Python 环境、项目骨架和全量索引审计，生成了本地邮件清单与固定开发样本。邮件正文解析、数据划分和模型训练在后续阶段开展。

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

目前可执行入口为环境检查和下述索引审计。后续运行顺序：解析清洗 → 去重划分 → EDA/特征 → 模型比较 → 最终评估；这些后续脚本尚未实现。

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

**Notebook 使用**

已安装 `ipykernel`。在现有 Notebook 或编辑器中选择本仓库的 `.venv` 解释器。如需在现有 Jupyter 中注册专用内核，可手动执行：

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name python-teamwork --display-name 'Python teamwork (.venv)'
```

本阶段不启动 Jupyter 服务，也未自动修改全局内核列表。Notebook 第一格建议检查 `sys.executable`，确认选中本项目环境。

**项目目录**

```text
src/spam_classifier/    可复用的索引审计模块，后续增加解析与建模
scripts/                check_environment.py 与 audit_index.py
tests/                  人工微型语料的边界测试
notebooks/              后续探索和展示过程
artifacts/audit/        聚合质量与环境记录
artifacts/processed/    本地邮件文本缓存，Git 忽略
artifacts/splits/       开发样本元数据；后续增加正式数据划分
artifacts/metrics/      后续聚合模型指标
artifacts/figures/      后续图表
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
