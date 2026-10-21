# 第七步：HTML 清洗对照与方案锁定

日期：2026-10-21。状态：程序和合成检查完成，真实语料实验未运行。

## 目标与边界

固定第六阶段的 LR C=2、训练/验证 ID 和全部其他配置，只比较 `text_clean` 与 `text_unstripped_html`。两种表示各自在同一训练集拟合 TF-IDF，验证集只转换。先按验证 macro-F1、正常邮件误报率、spam 召回率排序；完全相同时选择去标签文本。未收敛的方案不能入选。无提升或退化也如实保留。

`read_records` 为校验完整缓存指纹会读取全部缓存字节，但只返回指定训练/验证 ID，测试文本不进入训练、选择或统计。第七步不打开测试 CSV/矩阵，不计算测试分类效果。人工机制示例沿用 [清洗示例](../清洗示例.md)，不是本次实测案例。

## 文件与执行方法

| 文件 | 用途 |
|---|---|
| `src/spam_classifier/experiment_io.py` | 上游验收、来源与产物指纹、冻结 ID 对齐、最终锁定契约 |
| `src/spam_classifier/cleaning_ablation.py` | 两臂训练/验证、噪声代理统计、选择规则与三种模型冻结 |
| `scripts/compare_cleaning.py` | 重现原清洗臂，保存结果，最后写入不可静默覆盖的锁 |
| `scripts/check_ablation.py` | 仅用训练/验证独立重训、重算指标、检查排序与模型来源 |
| `tests/test_cleaning_ablation.py` | 8 项合成边界测试，包括泄漏、ID/标签、选择和指纹 |

在原依赖与第 2—6 步私有缓存全部恢复并核验后执行：

```bash
python scripts/check_readiness.py
python scripts/compare_cleaning.py
python scripts/check_ablation.py
```

去标签臂必须逐项重现第六步的验证预测、指标和训练词表/IDF，否则不能锁定。两臂分别保存模型；选定表示后，将多数类、NB alpha=0.1 与 LR C=2 的训练侧模型一起冻结。采用默认 `predict`，不调阈值，不合并训练/验证再拟合。若已经存在测试访问记录，禁止删除锁后重新选方案。

## 验证与产物

实际执行 `python -m unittest discover -s tests -p test_cleaning_ablation.py -q`：8 项通过。基线 110 项已在续作准备中核验。本次新增测试使用明确构造的微型文本，不能替代真实语料实验。

实际运行 `python scripts/compare_cleaning.py` 返回 2（blocked），因为本机没有上游私有数据与模型，且运行版本不同；没有写入正式对照成绩或最终锁。缺失明细见 [续作条件](../../artifacts/audit/continuation_readiness.json)。

真实运行后才会生成以下产物：

- 公开：`artifacts/metrics/cleaning_ablation.json`、`stage07_validation.csv`、`final_lock.json` 与 `artifacts/audit/ablation_validation.json`。
- 本地忽略：`artifacts/models/stage07/` 和 `artifacts/processed/stage07/`，含模型与逐封验证预测，不公开正文或完整词表。

## 问题与下一步

先恢复原数据与兼容环境，再运行真实对照，审核结果并锁定。第八步必须验证本步锁和独立验收后才能计算最终测试效果。预设 HTML token 比例仅是噪声代理，不能当作语义质量或效果提升的证明。

本阶段代码、测试和记录一同做本地提交。本阶段提交时 GitHub 连接账号对目标仓库只有读取权限，未上传；后续账号权限变化及应用 403 阻塞见 [最新交付复核](10a_交付复核与上传状态.md)。提交按实际时间记录，不补造历史时间或成员贡献。
