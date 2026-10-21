"""Build an editable evidence-based report draft, never invent missing results.

Requires python-docx==1.2.0 in a separate document-authoring environment.
Run from any directory; source evidence is resolved relative to this script.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
FONT = "Noto Sans CJK SC"


def build():
    OUT.mkdir(exist_ok=True)
    eda = json.loads((ROOT / "artifacts/audit/eda_summary.json").read_text(encoding="utf-8"))
    with (ROOT / "artifacts/metrics/stage06_validation.csv").open(encoding="utf-8", newline="") as stream:
        experiments = list(csv.DictReader(stream))
    by_id = {row["candidate_id"]: row for row in experiments}
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin = sec.bottom_margin = Inches(.7)
    sec.left_margin = sec.right_margin = Inches(.78)
    sec.header_distance = sec.footer_distance = Inches(.3)
    for name, size in (("Normal", 11), ("Title", 24), ("Subtitle", 12),
                       ("Heading 1", 16), ("Heading 2", 12), ("Caption", 9)):
        style = doc.styles[name]
        style.font.name, style.font.size = FONT, Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        fonts = style.element.get_or_add_rPr().rFonts
        for key in list(fonts.attrib):
            if "theme" in key.lower(): del fonts.attrib[key]
        fonts.set(qn("w:eastAsia"), FONT)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.line_spacing = Pt(size * 1.45)
        snap = OxmlElement("w:snapToGrid"); snap.set(qn("w:val"), "0")
        style.element.get_or_add_pPr().append(snap)
    for style in doc.styles:
        for border in list(style.element.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    doc.styles["Subtitle"].font.italic = False
    doc.styles["Title"].font.bold = True
    doc.styles["Caption"].font.bold = False
    doc.styles["Normal"].paragraph_format.widow_control = True
    for name in ("Heading 1", "Heading 2"):
        doc.styles[name].font.bold = True
        doc.styles[name].paragraph_format.space_before = Pt(8)
    header = sec.header.paragraphs[0]
    header.text = "中文垃圾邮件分类  |  阶段报告草稿 · 仅含验证集成绩"
    header.style = doc.styles["Caption"]
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Python 数据分析编程 · 第六题    ").font.size = Pt(9)
    field = OxmlElement("w:fldSimple"); field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    md = []

    def p(text, style=None):
        doc.add_paragraph(text, style)
        md.append(text + "\n")

    def h(text, level=1):
        doc.add_heading(text, level)
        md.append("#" * (level + 1) + " " + text + "\n")

    def page():
        # Let text flow across sections; avoid an orphan paragraph followed by
        # a forced page break when fonts vary on another machine.
        pass

    def table(headers, rows, widths):
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for col, width in zip(t.columns, widths): col.width = Inches(width)
        for i, text in enumerate(headers): t.rows[0].cells[i].text = str(text)
        for row in rows:
            for cell, text in zip(t.add_row().cells, row): cell.text = str(text)
        repeat = OxmlElement("w:tblHeader"); t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for ri, row in enumerate(t.rows):
            trpr = row._tr.get_or_add_trPr()
            trpr.append(OxmlElement("w:cantSplit"))
            for ci, cell in enumerate(row.cells):
                cell.width = Inches(widths[ci]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                pr = cell._tc.get_or_add_tcPr()
                shade = OxmlElement("w:shd"); shade.set(qn("w:fill"), "243E55" if ri == 0 else "F1F5F8" if ri % 2 else "FFFFFF"); pr.append(shade)
                borders = OxmlElement("w:tcBorders")
                for edge in ("top", "left", "bottom", "right"):
                    node = OxmlElement("w:" + edge)
                    for key, val in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")): node.set(qn("w:" + key), val)
                    borders.append(node)
                pr.append(borders)
                margins = OxmlElement("w:tcMar")
                for edge, size in (("top", "85"), ("bottom", "85"), ("left", "105"), ("right", "105")):
                    node = OxmlElement("w:" + edge); node.set(qn("w:w"), size); node.set(qn("w:type"), "dxa"); margins.append(node)
                pr.append(margins)
                for para in cell.paragraphs:
                    para.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    para.paragraph_format.space_after = Pt(0); para.paragraph_format.line_spacing = Pt(14)
                    para.paragraph_format.keep_with_next = ri < len(t.rows) - 1
                    for run in para.runs:
                        run.font.size = Pt(10); run.bold = ri == 0
                        run.font.color.rgb = RGBColor.from_string("FFFFFF" if ri == 0 else "000000")
        doc.add_paragraph().paragraph_format.space_after = Pt(1)
        md.append("| " + " | ".join(headers) + " |\n|" + "---|" * len(headers))
        md.extend("| " + " | ".join(map(str, row)) + " |" for row in rows); md.append("")

    def figure(name, caption, width=6.75):
        para = doc.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.line_spacing = 1.0
        para.add_run().add_picture(str(ROOT / "artifacts/figures" / name), width=Inches(width))
        para.paragraph_format.space_after = Pt(2); para.paragraph_format.keep_with_next = True
        p(caption, "Caption")
        md.append(f"![{caption}](../artifacts/figures/{name})\n")

    p("中文垃圾邮件分类", "Title")
    p("基于 TREC06C、jieba 与 TF-IDF 的朴素贝叶斯和逻辑回归比较", "Subtitle")
    p("《Python 数据分析编程》小组项目 · 第六题\n证据基线：cd8e106；续作草稿日期：2026 年 10 月 22 日", "Caption")
    h("摘要")
    p("本项目以课程提供的 TREC 2006 中文邮件为对象，研究原始邮件解析、内容清洗、去重划分和文本分类。主索引包含 64,620 封邮件。清洗后全部技术可用，去除重复后保留 33,988 封代表，并冻结训练、验证和测试集合。特征采用 jieba 分词与仅在训练侧拟合的 50,000 维 TF-IDF；比较多数类、MultinomialNB 和 LogisticRegression。现有真实验证实验中，LR C=2 的准确率为 98.86%，macro-F1 为 98.78%，正常邮件误报 11 封、垃圾邮件漏报 47 封。该结果支持它进入后续清洗对照，不代表最终测试效果。")
    p("关键词：中文垃圾邮件；邮件解析；TF-IDF；朴素贝叶斯；逻辑回归")
    p("材料状态：原教师 Word 模板、成员信息及真实实验私有输入尚未取得。本稿先提供可核对的内容与可编辑排版；第七步实测、第八步最终测试与人工错分结论未完成，不能直接作为课程终稿提交。")
    h("1 引言")
    p("垃圾邮件识别是一个二分类问题。本项目以 spam=1、ham=0 编码标签，主输入为邮件主题与正文。误报是把正常邮件判为垃圾邮件，可能导致重要通知被拦截；漏报是把垃圾邮件当作正常邮件，影响过滤效果。因此除准确率外，同时报告垃圾邮件精确率、召回率、两类 F1、macro-F1 和正常邮件误报率。")
    p("工作目标包括可靠解析不同编码与 MIME 结构、公平比较两种模型，以及解释清洗和错误类型。采用小规模验证调参，保留最终测试集用于锁定方案后的评估。来源、版本、划分和验收记录随代码保存，使成绩能追溯到实际数据处理过程。")

    page(); h("2 数据介绍")
    table(["范围", "正常邮件", "垃圾邮件", "合计"],
          [["原始主索引", "21,766", "42,854", "64,620"], ["清洗后技术可用", "21,766", "42,854", "64,620"], ["去重保留代表", "21,411", "12,577", "33,988"]], [2.2,1.55,1.55,1.55])
    p("数据入口固定为 trec06c/full/index，每行提供标签和相对路径。delay/index 与主索引涉及相同邮件，不能视为新增数据或独立测试集。原始文件约 132.26 MiB，采用全量处理。开发集预先固定 ham、spam 各 200 封，仅用于调试解析器，不代表自然类别比例。")
    h("3 清洗与预处理")
    h("3.1 编码、MIME 与 HTML", 2)
    p("首先以二进制读取邮件并解析头部和 MIME 结构，先解 base64 或 quoted-printable 传输编码，再按声明字符集和明确候选回退解码。对 multipart/alternative 选取有效表示，避免把纯文本与 HTML 的相同内容重复拼接；对 related 保留根内容，跳过二进制附件和转发的整封子邮件。损坏边界、回退动作及质量警告均单独记录。")
    p("HTML 处理移除 script/style，解码实体并保留可读文本与锚文本；之后进行 Unicode NFKC、英文小写、控制字符清理、空白合并和常见联系字段归一化。主题与正文分别保留后组合。只有主题可用时仍保留样本；不可用样本应进入隔离账目。常见电话号码掩码不能等同于全面匿名化。")
    table(["互斥解析状态", "邮件数", "解释"], [["ok", "29,891", "常规处理可用"], ["recovered", "33,307", "记录回退后可用"], ["subject_only", "1,422", "仅主题可用"], ["隔离或错误", "0", "本批技术处理结果"]], [2.3,1.25,3.3])
    p("技术可用不等于语义完全正确。没有替换符也无法证明没有乱码，编码质量仍需结合原文复核。公开清洗示例为人工构造的机制说明，真实正文、解码文本和保留标签的对照文本只保存在本地。", "Caption")

    page(); h("3.2 内容去重与冻结划分", 2)
    p("对同标签且规范化主题与正文完全相同的邮件保留一个确定代表，共移除 30,632 封重复记录。相同原始字节、最终文本、有效正文或 HTML 对照文本建立关联分组，整组划分，防止精确内容跨集合。解析器开发样本触及的 372 个分组、1,738 封代表固定到训练侧。种子为 42，目标比例为 70%/15%/15%。")
    table(["集合", "正常邮件", "垃圾邮件", "合计"], [[name, f"{row['ham']:,}", f"{row['spam']:,}", f"{row['total']:,}"] for name, row in [("训练", eda['counts']['split_counts']['train']), ("验证", eda['counts']['split_counts']['validation']), ("测试", eda['counts']['split_counts']['test'])]], [2.2,1.55,1.55,1.55])
    h("4 探索分析")
    figure("stage05_category_counts.png", "图 1  类别规模与去重保留率。来源：既有第五阶段真实聚合产物。")
    p("垃圾邮件占比由原始约 66.32% 降为保留代表中的 37.00%；ham 保留率约 98.37%，spam 约 29.35%。这反映语料中垃圾邮件的精确重复更多，不能解释为清洗随机丢弃了垃圾邮件。评估对象因此是不同规范化文本，不能直接外推自然收件箱的邮件流比例。")
    p("训练侧正常与垃圾邮件的文本长度中位数分别为 302、499 字符，正文 HTML 率分别约 4.18%、33.88%。这些描述性差异有助于安排质量复核，但不作为读取测试表现后修改特征的依据。")

    page(); h("4.1 高频词与文本差异", 2)
    figure("stage05_training_word_coverage.png", "图 2  训练侧预设公开词表内的高覆盖词。每封邮件每词最多计一次。")
    p("正常邮件中的“工作”“问题”“时间”等词较常见；垃圾邮件中联系字段占位符和“公司”“服务”等词覆盖较高。图中词仅来自预设通用词表，并非完整词表的全局前列；高频也不等于能区分类别，须结合模型系数与实际错分判断。")
    h("5 特征与模型")
    p("jieba 使用精确模式、HMM=False，保留固定规则允许的中文单字、数字及联系字段占位符，使用 57 项固定停用词。TF-IDF 以训练邮件中的词频和文档频率形成向量：重复词的影响使用次线性缩放，常见词降低区分权重，再进行 L2 归一化。词表和 IDF 只在训练集拟合；验证、测试只能转换。")
    table(["组件", "固定配置或候选"], [["TF-IDF", "min_df=2；max_features=50,000；float64"], ["多数类", "由训练集确定多数标签"], ["MultinomialNB", "alpha=0.1、0.5、1.0"], ["LogisticRegression", "C=0.5、1、2；检查收敛"], ["选择顺序", "macro-F1 → 低误报率 → 高 spam 召回率"]], [2.15,4.7])
    p("NB 用平滑后的词分布进行分类，结构简单，适合稀疏文本基线；LR 联合学习每个词的线性权重，并用正则化约束模型。两者共用冻结划分和特征规则。样本 ID、标签、路径、路由头及现成垃圾评分均不作为特征。")

    page(); h("6 评估与可视化")
    h("6.1 现有真实验证结果", 2)
    table(["模型", "准确率", "spam 精确率", "spam 召回率", "macro-F1", "误报率"], [[{"majority":"多数类", "nb_alpha_1":"NB α=1", "lr_C_1":"LR C=1", "nb_alpha_0.1":"NB α=0.1", "nb_alpha_0.5":"NB α=0.5", "lr_C_0.5":"LR C=0.5", "lr_C_2":"LR C=2"}[r['candidate_id']], *[f"{100*float(r[k]):.2f}%" for k in ('accuracy','spam_precision','spam_recall','macro_f1','ham_false_positive_rate')]] for r in experiments], [1.25,1.03,1.22,1.22,1.1,1.03])
    p("表中全部为 5,099 封验证邮件的结果。来源：artifacts/metrics/stage06_validation.csv；不是本轮重新运行的真实语料实验。macro-F1 平均两类 F1，可避免只关注样本较多的正常邮件。多数类虽达到 62.99% 准确率，却漏掉全部 1,887 封垃圾邮件。")
    h("6.2 最优参数下的验证混淆计数", 2)
    table(["模型", "TN", "FP", "FN", "TP"], [[label, *[by_id[key][k] for k in ('tn','fp','fn','tp')]] for label,key in [("NB α=0.1","nb_alpha_0.1"),("LR C=2","lr_C_2")]], [2.05,1.2,1.2,1.2,1.2])
    p("TN 为正确识别正常邮件，FP 为误拦正常邮件，FN 为漏掉垃圾邮件，TP 为正确识别垃圾邮件。LR C=2 相比 NB alpha=0.1，将验证误报从 126 封降至 11 封，漏报从 84 封降至 47 封，macro-F1 高约 3.17 个百分点。因此按预设规则选为下一阶段候选。")
    p("这些比较支持当前语料与划分下的候选选择，尚未给出跨语料、时间外推或显著性结论。调参也使用了此验证集，最终泛化能力须由后续锁定方案的测试集评估。")
    h("6.3 清洗影响与最终评估状态", 2)
    p("第七步已实现固定 LR C=2 的去标签与保留 HTML 对照，各自在相同训练 ID 拟合 TF-IDF；去标签臂还必须重现既有第六步预测。真实对照未运行，不能宣称 HTML 清洗提升了准确率。第八步已实现锁定后预测、混淆矩阵、关联词和私有错分候选；真实最终测试未运行。")

    page(); h("6.4 错误分析方法与待执行内容", 2)
    p("最终评估后，NB、LR 各自按固定规则选取最多 15 个 FP 和 15 个 FN；不足则全部保留。候选包含真实正文，只保存在本地，不能上传仓库。人工阅读后应记录短文本、主题与正文不一致、残余编码问题或合法商业通知等实际证据；这些类别目前仅为审核方向，不是已经观察到的错因。")
    p("公开结果只写经过复核的安全概述及聚合数量。模型关联词用 LR 正负系数解释，限定在预设公开词表内；它反映条件关联，不是因果关系。全部错误的 HTML、短文本和编码标记计数也只作描述，不能用自动标记取代原文审核。")
    h("7 总结与结论")
    p("当前证据支持三点：第一，邮件解析必须记录编码和结构恢复过程，技术可读不保证语义无误。第二，精确去重显著改变本语料的类别构成，固定内容分组可降低重复泄漏。第三，相同训练特征下，LR C=2 在现有验证集上的 macro-F1 和误报控制均优于本轮 NB 候选，因此值得进入清洗对照。")
    p("主要局限包括语料年代较早、仍可能存在近重复模板、类别比例不能代表实际邮件流，以及验证选参带来的乐观偏差。真实清洗对照、最终测试和人工错分审核尚待完成。后续应先恢复原数据与兼容依赖，保持冻结划分和预先声明的选择规则，再完成实测与最终材料。")
    h("8 分工与个人承担")
    p("仓库暂未提供三名成员的姓名、学号、组号、组长及实际承担记录。本稿不将建议职责 A/B/C 当作实际贡献，也不把自动化辅助提交归为某位学生。终稿需由成员按实际完成的代码、复核与讲解填写，并与 GitHub 的真实提交记录对应。")
    p("可按“成员信息—负责模块—本人使用的方法—遇到的问题及处理—对应提交—复核人”整理实际工作。本次续作已完成逐文件审阅、第七八步程序和合成检查，按模块分别提交；真实语料复现、登记及线下课程提交没有被代办或确认为完成。")

    page(); h("9 附录：复现与证据")
    h("9.1 仓库与运行顺序", 2)
    p("仓库：https://github.com/25121596/Python-teamwork\n历史结果基线：cd8e106；用户最初指定的 5864154 是第二阶段历史版本，续作保留其后第三至第六阶段成果。")
    p("原实验采用 Windows、Python 3.12，精确依赖见 requirements.txt。先把课程原始语料放在仓库之外，通过 config.local.json 指定路径，依次执行 README 的索引、清洗、划分、EDA/特征和模型训练/独立核验。私有缓存不随 Git 分发，新电脑必须恢复或重新生成，不能仅凭旧成功 JSON 认为复现通过。")
    p("第七八步入口依次为 check_readiness.py、compare_cleaning.py、check_ablation.py、evaluate_final.py、check_final_results.py。前一步未通过则停止。第一次最终测试前保存最终锁；测试后不能删除访问记录再调参。")
    h("9.2 核心代码摘录", 2)
    p("以下为 scripts/evaluate_final.py 中的真实控制顺序（节选），用于解释为何先验证锁与验收，再读取测试集合：")
    code = 'lock = verify_lock(root)\nlock_sha = file_sha256(root / "artifacts/metrics/final_lock.json")\nrequire_ablation_validation(root, lock_sha)\n# 省略重复运行检查和测试访问记录写入\nrows = read_test_rows(root, lock)\nrecords = read_records(root, {"test": rows},\n                       lock["cleaning_cache_sha256"])["test"]'
    para = doc.add_paragraph(); para.paragraph_format.line_spacing = 1.12
    run=para.add_run(code); run.font.name="DejaVu Sans Mono"; run.font.size=Pt(8.5)
    md.append("```python\n" + code + "\n```\n")
    h("9.3 数据与资料来源", 2)
    p("课程要求与章节映射：项目计划.md；解析/去重/探索证据：artifacts/audit/ 下的 cleaning_summary.json、split_summary.json、eda_summary.json；模型证据：artifacts/metrics/model_selection.json 与 stage06_validation.csv；逐阶段验收：docs/stages/；逐文件用途：docs/仓库文件用途.md。")
    p("本报告数值取自上述仓库已有真实产物；合成测试只检验程序逻辑，128 项通过不等于真实实验复现。恢复原 Word 模板后需迁移本文内容、补齐成员与实测结果、更新目录和页码，再完成打印及提交检查。")
    path = OUT / "项目报告草稿.docx"
    doc.save(path)
    (OUT / "项目报告草稿.md").write_text("# 中文垃圾邮件分类：项目报告草稿\n\n" + "\n".join(md), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
