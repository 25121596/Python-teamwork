# 第十一阶段：项目说明文档与代码演示 PPT

完成日期：2026 年 10 月 25 日。承接第九、十阶段的交付材料，补充一份完整技术说明和一份逐页穿插源码与运行结果的演示 PPT。本阶段不改变任何模型、划分或指标，只做文档与演示材料，并同步上传状态。

## 目标与边界

- 产出一份可独立阅读的项目说明文档，覆盖流程设计、每个阶段的数据与代码要点、全部实验结果、复现方法与局限。
- 产出一份 14 页的演示 PPT，每页必须同时出现**真实源码片段**和**对应运行结果**，而不是只有结论文字。
- 所有数字与代码片段都必须来自仓库内已跟踪的产物与源码，不允许手写复述。

## 实现方式

| 文件 | 说明 |
|---|---|
| `docs/项目说明文档.md` | 完整技术说明，10 章 + 3 附录 |
| `scripts/build_walkthrough_ppt.py` | PPT 生成器：数字从 `artifacts/` 读取，代码片段从 `src/` 按模式提取 |
| `scripts/check_presentation_layout.py` | 独立布局检查：越界与文本溢出风险 |
| `deliverables/项目演示.pptx` | 生成产物，14 页 16:9 |

生成器有两项硬约束，保证演示材料不会与实验结果脱节：

```python
# 数字：只从公开聚合产物读取
eda = load("artifacts/audit/eda_summary.json")
selection = load("artifacts/metrics/model_selection.json")
ablation = load("artifacts/metrics/cleaning_ablation.json")
final = load("artifacts/metrics/final_evaluation.json")
errors = load("artifacts/metrics/error_analysis.json")

# 代码：从跟踪的源文件按模式提取，片段随源码更新
def src_lines(rel: str, pattern: str, before: int = 0, after: int = 0) -> str:
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.search(pattern, line):
            return "\n".join(lines[max(0, index - before):index + after + 1])
    raise SystemExit(f"pattern not found in {rel}: {pattern}")
```

演示中出现的源码涉及 `data_split.py`（并查集合并、整组分配）、`email_parser.py`（传输编码回退、HTML 提取、统一清洗函数）、`text_features.py`（分词规则与 TF-IDF 参数）、`model_selection.py`（选择策略、指标定义、候选排序）、`final_evaluation.py`（锁定模型评估、候选选取）。

## PPT 结构

| 页 | 主题 | 代码/结果演示内容 |
|---:|---|---|
| 01 | 封面与结果摘要 | 语料、方法、模型、最终成绩概览 |
| 02 | 十步流程与防泄漏约束 | 并查集合并源码 + 指纹链校验源码 |
| 03 | 索引审计与语料规模 | 去重前后类别对比表 + 统一清洗函数源码 |
| 04 | 邮件解析：编码、MIME、HTML | base64/qp 回退源码 + `html_to_text` 源码 |
| 05 | 去重与冻结划分 | 三集合规模表 + 整组贪心分配源码 |
| 06 | jieba 分词与训练侧 TF-IDF | 分词规则源码 + TF-IDF 参数源码 + 人工分词示例 |
| 07 | 验证集七次实验 | 七行实验表 + 选择策略源码 |
| 08 | HTML 清洗对照 | 两臂对照表 + 确定性排序源码 |
| 09 | 冻结测试集最终评估 | 三模型指标表 + 混淆矩阵图 + 评估源码 |
| 10 | 混淆矩阵与两个错误方向 | 混淆矩阵图 + 指标定义源码 + FP/FN 业务解释 |
| 11 | 真实错分语义观察 | 候选选取源码 + 诊断统计表 |
| 12 | LR 关联词 | 系数明细表 + 系数分布图 |
| 13 | 可复现性 | 环境建立与完整运行顺序命令 |
| 14 | 结论与局限 | 结论/局限对照 + 证据入口 + 仓库地址 |

## 检查结果

- `scripts/check_presentation_layout.py`：**14 页，0 个问题**（无越界、无文本溢出风险）。
- 用 PowerPoint COM 实际打开确认：**14 页正常打开**，并逐页导出 PNG 目视核对。
- 排版修复过程中发现并修正三类问题：占位空代码框导致文本框高度为负（生成的文件 PowerPoint 无法打开）、代码片段超宽溢出深色面板、第 12 页表格与图重叠导致页码被遮。
- 文档内所有数字与 `artifacts/metrics/final_evaluation.json`、`cleaning_ablation.json`、`error_analysis.json`、`artifacts/audit/eda_summary.json` 逐项核对一致。

> 本阶段**没有**运行任何新实验，也没有改动 `src/` 下的模型代码、划分 CSV 或既有指标产物。PPT 中的成绩全部来自第 8 步已验收的冻结测试结果。

## 同步更新的文件

| 文件 | 改动 |
|---|---|
| `进度记录.md` | 更新日期、新增说明文档与演示 PPT 条目、修正上传状态描述 |
| `deliverables/README.md` | 交付材料清单新增两项，说明生成与检查方式，修正 GitHub 同步状态 |
| `docs/stages/10a_交付复核与上传状态.md` | 记录写入阻塞已解除，保留原始排查过程与 bundle 恢复方式 |

## 遗留事项

- `docs/成员分工.md` 中三名成员的姓名、学号、组号、组长与实际承担仍待小组填写；本阶段未代填，也未在报告中冒充成员贡献。
- 56 个错分实例的语义复核仍是 AI 辅助完成，学生人工签认未完成。
- 正式 5 分钟讲解 MP4 仍待小组成员本人录制。
- 新增的 `python-pptx` 依赖仅用于生成演示材料，未加入 `requirements.txt`（不影响实验复现路径）；如需在他人机器重生成演示材料，需另行安装该包。
