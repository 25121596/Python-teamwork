# 第七步 HTML 清洗对照与方案锁定

更新：2026-10-22。真实两臂对照与全部 18 项独立核验已通过，最终锁已保存，允许进入第八步。此前缺少语料时仅完成程序和 8 项合成检查，历史状态见前序 Git 提交。

## 目标与边界

固定本轮第六步的 LR C=2、train/validation ID、标签、分词及 TF-IDF 参数，仅比较 `text_clean` 与 `text_unstripped_html`。两臂分别在同一训练集学习词表和 IDF，验证集只转换。排序规则为验证 macro-F1 降序、误报率升序、spam 召回率降序，完全相同则选择去标签。

`read_records` 读取完整缓存字节来验证 SHA-256，但只返回指定训练/验证 ID；测试文本不用于拟合、选择或统计。第七步不打开测试 CSV/矩阵，不计算测试分类指标。原第六步历史与本轮 Linux 重建的差异见 [复现说明](06c_真实语料复现与差异说明.md)。

## 执行方法

```bash
python scripts/check_readiness.py
python scripts/compare_cleaning.py
python scripts/check_ablation.py
```

生成器核对上游来源，要求去标签臂逐项重现本轮第六步的验证预测、指标、词表和 IDF，之后才写 `final_lock.json`。验收器只在训练侧独立重训，核对两臂预测、词表、LR 系数、独立计算的指标和排序，并重放三种最终模型。

## 实测结果

| 输入 | 准确率 | spam 精确率 | spam 召回率 | macro-F1 | 正常误报率 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 去标签 | 98.8625% | 99.4057% | 97.5093% | 98.7753% | 0.3425% | 11 | 47 |
| 保留 HTML | 98.8429% | 99.4054% | 97.4563% | 98.7540% | 0.3425% | 11 | 48 |

去标签 macro-F1 高 0.0213 个百分点，验证预测只相差 1 封，去标签正确而保留 HTML 错误。两种缓存表示实际不同的训练样本为 156/23,792，验证为 36/5,099；所选正文标记 HTML 不必然表示实际含可剥离标签，因此不能把“has_html 比例”当作改动比例。预设 HTML 类词占训练 token 的比例由约 0.2209% 降至 0.0198%，只表示噪声代理，不能证明语义改善。

这是一项小幅、特定语料和固定模型下的优势，没有显著性或普遍改善的证据。按预设规则锁定去标签表示，不因差异小而修改选择规则。NB alpha=0.1、LR C=2 和多数类均冻结为训练侧模型；默认 predict，不调阈值，不合并训练与验证再拟合。

## 产物与验收

- `artifacts/metrics/cleaning_ablation.json`：两臂统计、来源范围、最终验证成绩与表示选择。
- `artifacts/metrics/stage07_validation.csv`：公开对照表。
- `artifacts/metrics/final_lock.json`：测试前模型、参数、来源和产物指纹。
- `artifacts/audit/ablation_validation.json`：独立重训与重算验收。
- `artifacts/models/stage07/`、`artifacts/processed/stage07/`：本地模型与逐封预测，保持 Git 忽略。

新增 Git 属性将所有公开 audit JSON、metrics JSON/CSV 固定 LF，使 Windows 检出不会改变最终锁引用的字节；开发样本仍保留原记录所需 CRLF。

本阶段独立提交，先保存通过的锁及验收，再执行最终测试。GitHub 应用写入仍为 403，未上传成功。不得删除锁或测试访问记录以重新挑选方案。

> 后续状态更新（2026-10-25）：写入权限已恢复，本分支已推送到远程并核验。
