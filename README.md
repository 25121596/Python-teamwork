# Python-teamwork

《Python 数据分析编程》小组项目：第六题中文垃圾邮件分类。

执行步骤、三人分工、时间安排和提交要求见 [项目计划](项目计划.md)。

阶段完成情况见 [进度记录](进度记录.md)。每个阶段的成果与 Markdown 记录一同提交并推送，具体约定见 [AGENTS.md](AGENTS.md)。

**当前范围**

本阶段建立独立 Python 环境、依赖清单、项目目录、数据路径配置与环境检查脚本。邮件解析、全量审计、数据划分和模型训练在后续阶段开展。

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

目前可执行入口只有环境检查。后续运行顺序按项目计划补充：索引审计 → 解析清洗 → 去重划分 → EDA/特征 → 模型比较 → 最终评估。相应脚本尚未实现，不应把本阶段检查当作训练结果。

**Notebook 使用**

已安装 `ipykernel`。在现有 Notebook 或编辑器中选择本仓库的 `.venv` 解释器。如需在现有 Jupyter 中注册专用内核，可手动执行：

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name python-teamwork --display-name 'Python teamwork (.venv)'
```

本阶段不启动 Jupyter 服务，也未自动修改全局内核列表。Notebook 第一格建议检查 `sys.executable`，确认选中本项目环境。

**项目目录**

```text
src/spam_classifier/    后续可复用的解析、特征与模型模块
scripts/                可执行入口，现有 check_environment.py
notebooks/              后续探索和展示过程
artifacts/audit/        聚合质量与环境记录
artifacts/processed/    本地邮件文本缓存，Git 忽略
artifacts/splits/       后续固定样本划分
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
