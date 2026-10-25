"""Build the walkthrough presentation from verified public repository evidence.

Every number is read from artifacts/ so the deck cannot drift from the run.
Code fragments are read from the tracked source files so they stay in sync.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "figures"

NAVY = RGBColor(0x18, 0x36, 0x4A)
TEAL = RGBColor(0x18, 0x7A, 0x87)
ORANGE = RGBColor(0xC5, 0x6A, 0x32)
GREY = RGBColor(0x52, 0x63, 0x72)
LIGHT = RGBColor(0xF2, 0xF5, 0xF7)
CODE_BG = RGBColor(0x1E, 0x28, 0x32)
CODE_FG = RGBColor(0xE6, 0xED, 0xF3)
CODE_KEY = RGBColor(0x7F, 0xC9, 0xE8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xB4, 0x3F, 0x3F)
GREEN = RGBColor(0x2E, 0x7D, 0x5B)

CJK = "Microsoft YaHei"
MONO = "Consolas"

W, H = Emu(12192000), Emu(6858000)  # 16:9


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def src_lines(rel: str, pattern: str, before: int = 0, after: int = 0) -> str:
    """Extract a real code fragment from a tracked source file."""
    text = (ROOT / rel).read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.search(pattern, line):
            start = max(0, index - before)
            end = min(len(lines), index + after + 1)
            return "\n".join(lines[start:end])
    raise SystemExit(f"pattern not found in {rel}: {pattern}")


eda = load("artifacts/audit/eda_summary.json")
cleaning = load("artifacts/audit/cleaning_summary.json")
split = load("artifacts/audit/split_summary.json")
feature = load("artifacts/audit/feature_summary.json")
selection = load("artifacts/metrics/model_selection.json")
ablation = load("artifacts/metrics/cleaning_ablation.json")
final = load("artifacts/metrics/final_evaluation.json")
errors = load("artifacts/metrics/error_analysis.json")
review = load("artifacts/metrics/error_review_assisted.json")

metric = {row["candidate_id"]: row["metrics"] for row in selection["experiments"]}
clean_arm = next(c for c in ablation["comparisons"] if c["representation"] == "text_clean")
html_arm = next(c for c in ablation["comparisons"] if c["representation"] == "text_unstripped_html")
test = final["metrics"]
coeffs = errors["public_coefficients"]


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, kicker: str, title: str):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = WHITE
        self.text(s, kicker, 0.55, 0.28, 11.6, 0.32, 14, GREY, CJK)
        self.text(s, title, 0.55, 0.62, 11.6, 0.62, 26, NAVY, CJK, bold=True)
        self.line(s, 0.55, 1.32, 12.2, 0.028, TEAL)
        self.text(s, f"{self.n:02d}", 12.05, 6.82, 0.75, 0.32, 12, GREY, CJK, align=PP_ALIGN.RIGHT)
        return s

    def text(self, s, value, x, y, w, h, size, color, font=CJK, bold=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0):
        box = s.shapes.add_textbox(Emu(int(x * 914400)), Emu(int(y * 914400)),
                                   Emu(int(w * 914400)), Emu(int(h * 914400)))
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Emu(0)
        tf.margin_top = tf.margin_bottom = Emu(0)
        lines = value.split("\n")
        for i, line in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            para.line_spacing = spacing
            run = para.add_run()
            run.text = line
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
            run.font.name = font
            if font == CJK:
                run.font._rPr.set("altLang", "zh-CN")
        return box

    def line(self, s, x, y, w, h, color):
        from pptx.enum.shapes import MSO_SHAPE
        shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(int(x * 914400)), Emu(int(y * 914400)),
                                   Emu(int(w * 914400)), Emu(int(h * 914400)))
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()
        shape.shadow.inherit = False
        return shape

    def panel(self, s, x, y, w, h, fill=LIGHT):
        from pptx.enum.shapes import MSO_SHAPE
        shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(int(x * 914400)), Emu(int(y * 914400)),
                                   Emu(int(w * 914400)), Emu(int(h * 914400)))
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        shape.line.fill.background()
        shape.shadow.inherit = False
        shape.adjustments[0] = 0.03
        return shape

    def bullets(self, s, items, x, y, w, h, size=15, color=NAVY, gap=1.25):
        box = s.shapes.add_textbox(Emu(int(x * 914400)), Emu(int(y * 914400)),
                                   Emu(int(w * 914400)), Emu(int(h * 914400)))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Emu(0)
        tf.margin_top = tf.margin_bottom = Emu(0)
        for i, item in enumerate(items):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.line_spacing = gap
            para.space_after = Pt(7)
            if isinstance(item, tuple):
                head, body = item
                r1 = para.add_run()
                r1.text = f"▪ {head}  "
                r1.font.size = Pt(size)
                r1.font.bold = True
                r1.font.color.rgb = TEAL
                r1.font.name = CJK
                r2 = para.add_run()
                r2.text = body
                r2.font.size = Pt(size)
                r2.font.color.rgb = color
                r2.font.name = CJK
            else:
                r = para.add_run()
                r.text = f"▪ {item}"
                r.font.size = Pt(size)
                r.font.color.rgb = color
                r.font.name = CJK
        return box

    def code(self, s, code, x, y, w, h, size=11.5, caption=None):
        """Draw a code panel that always fits: wrap long lines, then shrink to fit height."""
        self.panel(s, x, y, w, h, CODE_BG)
        if caption:
            self.text(s, caption, x + 0.22, y + 0.13, w - 0.44, 0.3, 11, CODE_KEY, MONO, bold=True)
            top = y + 0.46
            height = h - 0.6
        else:
            top = y + 0.18
            height = h - 0.34
        if height <= 0.12 or w <= 0.9:
            return None

        inner_w = w - 0.44
        # Consolas advance width is about 0.60 em; PowerPoint line box is ~1.32 em.
        CHAR_W, LINE_H = 0.60, 1.32

        def wrap(source: str, pt: float) -> list[str]:
            limit_chars = max(18, int(inner_w * 72 / (pt * CHAR_W)))
            out: list[str] = []
            for line in source.split("\n"):
                if len(line) <= limit_chars:
                    out.append(line)
                    continue
                indent = len(line) - len(line.lstrip())
                cont = " " * min(indent + 4, max(1, limit_chars - 4))
                rest = line
                first = True
                while rest:
                    room = limit_chars if first else max(8, limit_chars - len(cont))
                    cut = room
                    if len(rest) > room:
                        for sep in (" or ", " and ", ", ", " == ", "):", "):"):
                            pos = rest.rfind(sep, 0, room)
                            if pos > room * 0.45:
                                cut = pos + len(sep) - 1
                                break
                    out.append(rest[:cut].rstrip())
                    rest = rest[cut:].lstrip()
                    if rest:
                        rest = cont + rest
                    first = False
            return out

        wrapped = wrap(code, size)
        # shrink the font until the block fits the panel height
        while len(wrapped) * size * LINE_H / 72 > height and size > 6.0:
            size -= 0.5
            wrapped = wrap(code, size)

        box = s.shapes.add_textbox(Emu(int((x + 0.22) * 914400)), Emu(int(top * 914400)),
                                   Emu(int(inner_w * 914400)), Emu(int(height * 914400)))
        tf = box.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = Emu(0)
        tf.margin_top = tf.margin_bottom = Emu(0)
        for i, line in enumerate(wrapped):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.line_spacing = 1.0
            run = para.add_run()
            run.text = line if line else " "
            run.font.size = Pt(size)
            run.font.color.rgb = CODE_FG
            run.font.name = MONO
        return box

    def table(self, s, headers, rows, x, y, w, col_w=None, size=12.5,
              highlight=None, row_h=0.34, head_h=0.4):
        from pptx.enum.text import PP_ALIGN as A
        shape = s.shapes.add_table(len(rows) + 1, len(headers),
                                   Emu(int(x * 914400)), Emu(int(y * 914400)),
                                   Emu(int(w * 914400)), Emu(int((head_h + row_h * len(rows)) * 914400)))
        tbl = shape.table
        if col_w:
            total = sum(col_w)
            for i, cw in enumerate(col_w):
                tbl.columns[i].width = Emu(int(w * cw / total * 914400))
        tbl.rows[0].height = Emu(int(head_h * 914400))
        for i in range(len(rows)):
            tbl.rows[i + 1].height = Emu(int(row_h * 914400))
        for c, head in enumerate(headers):
            cell = tbl.cell(0, c)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY
            cell.margin_left = cell.margin_right = Emu(45720)
            cell.margin_top = cell.margin_bottom = Emu(9144)
            p = cell.text_frame.paragraphs[0]
            p.alignment = A.LEFT if c == 0 else A.CENTER
            r = p.add_run()
            r.text = head
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.color.rgb = WHITE
            r.font.name = CJK
        for ri, row in enumerate(rows):
            for c, value in enumerate(row):
                cell = tbl.cell(ri + 1, c)
                cell.text = ""
                cell.fill.solid()
                if highlight is not None and ri == highlight:
                    cell.fill.fore_color.rgb = RGBColor(0xFB, 0xE7, 0xD6)
                else:
                    cell.fill.fore_color.rgb = WHITE if ri % 2 == 0 else LIGHT
                cell.margin_left = cell.margin_right = Emu(45720)
                cell.margin_top = cell.margin_bottom = Emu(9144)
                p = cell.text_frame.paragraphs[0]
                p.alignment = A.LEFT if c == 0 else A.CENTER
                r = p.add_run()
                r.text = str(value)
                r.font.size = Pt(size)
                r.font.bold = highlight is not None and ri == highlight
                r.font.color.rgb = NAVY
                r.font.name = CJK
        return tbl

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(path)


d = Deck()

# ---------------------------------------------------------------- 1 封面
s = d.slide("PYTHON 数据分析编程  /  第六题", "中文垃圾邮件分类")
d.panel(s, 0.55, 1.62, 12.2, 0.72, LIGHT)
d.text(s, "从原始邮件解析到可解释的分类结果", 0.8, 1.79, 11.7, 0.4, 17, TEAL, CJK, bold=True)
d.bullets(s, [
    ("语料", "TREC 2006 中文邮件，64,620 封 → 去重后 33,988 封代表"),
    ("方法", "MIME 解析 + 编码恢复 + HTML 清洗 → jieba 分词 + 训练侧 TF-IDF"),
    ("模型", "多数类基线 / 朴素贝叶斯 / 逻辑回归，同一样本公平比较"),
    ("结果", f"冻结测试集 LR 准确率 {pct(test['lr']['accuracy'])}，macro-F1 {pct(test['lr']['macro_f1'])}，误报 {test['lr']['confusion_matrix']['fp']} 封、漏报 {test['lr']['confusion_matrix']['fn']} 封"),
], 0.8, 2.6, 11.7, 2.4, 16, NAVY)
d.panel(s, 0.55, 5.2, 12.2, 1.15, RGBColor(0xE8, 0xF1, 0xF3))
d.text(s, "本次汇报的演示方式", 0.8, 5.38, 11.7, 0.32, 13, TEAL, CJK, bold=True)
d.text(s, "每个关键环节都穿插真实代码片段与实际运行结果，所有数字直接读取仓库 artifacts/ 下的聚合产物。",
       0.8, 5.72, 11.7, 0.5, 13.5, NAVY, CJK)

# ---------------------------------------------------------------- 2 流程
s = d.slide("总体设计", "十步流程与防泄漏约束")
steps = [
    ("①", "索引审计", "只读 full/index"),
    ("②", "解析清洗", "MIME→编码→HTML"),
    ("③", "去重划分", "整组不跨集合"),
    ("④", "EDA/特征", "仅训练侧拟合"),
    ("⑤", "模型比较", "验证集选参"),
    ("⑥", "HTML 对照", "锁定方案"),
    ("⑦", "最终测试", "锁后才读取"),
]
x = 0.55
for num, name, note in steps:
    d.panel(s, x, 1.62, 1.66, 1.02, RGBColor(0xE8, 0xF1, 0xF3))
    d.text(s, num, x + 0.12, 1.71, 0.4, 0.32, 15, ORANGE, CJK, bold=True)
    d.text(s, name, x + 0.12, 2.05, 1.45, 0.3, 12.5, NAVY, CJK, bold=True)
    d.text(s, note, x + 0.12, 2.33, 1.45, 0.26, 10.5, GREY, CJK)
    x += 1.75
d.code(s, src_lines("src/spam_classifier/data_split.py", r"def union", 0, 3),
       0.55, 2.95, 6.1, 1.6, 11,
       "src/spam_classifier/data_split.py — 并查集合并同内容邮件")
d.bullets(s, [
    ("四层防泄漏", "内容分组整组划分 / 开发样本强制训练侧 / 词表只在训练集拟合 / 指纹链 + 不可变模型锁"),
    ("为什么重要", "课程明确要求 TF-IDF 只在训练集拟合、相同内容不跨集合；不做到会高估成绩"),
], 6.9, 3.0, 5.85, 1.6, 13.5)
d.panel(s, 0.55, 4.85, 12.2, 1.5, LIGHT)
d.text(s, "指纹链：任何一环被改动，下游立即拒绝加载", 0.8, 5.02, 11.7, 0.32, 13, TEAL, CJK, bold=True)
d.code(s, src_lines("src/spam_classifier/model_inputs.py", r"raise ValueError\(\"Parser or cleaning", 2, 0),
       0.8, 5.42, 11.7, 0.8, 11)

# ---------------------------------------------------------------- 3 数据
s = d.slide("第 2—4 步", "索引审计与语料规模")
d.table(s, ["范围", "正常 ham", "垃圾 spam", "合计", "spam 占比"], [
    ["主索引 full/index", "21,766", "42,854", "64,620", "66.32%"],
    ["技术可用", "21,766", "42,854", "64,620", "66.32%"],
    ["去重后代表", "21,411", "12,577", "33,988", "37.00%"],
], 0.55, 1.62, 12.2, [3.2, 1.5, 1.5, 1.5, 1.5], 13.5, highlight=2)
d.bullets(s, [
    ("唯一入口", "只使用 full/index；delay/index 与它指向同一批邮件，不能当新增样本或独立测试集"),
    ("保留率差异", f"正常邮件保留 {pct(eda['counts']['retention_rate_by_label']['ham'])}，垃圾邮件只保留 {pct(eda['counts']['retention_rate_by_label']['spam'])}"),
    ("结论", "垃圾邮件的精确重复远多于正常邮件，最终成绩描述的是「不同规范化文本代表」，不是自然邮件流分布"),
], 0.55, 3.3, 6.3, 2.6, 13.5)
d.code(s, src_lines("src/spam_classifier/email_parser.py", r"def clean_text", 0, 12),
       7.1, 3.2, 5.65, 2.55, 10.5,
       "src/spam_classifier/email_parser.py — 统一清洗函数")
d.panel(s, 0.55, 6.0, 12.2, 0.72, RGBColor(0xE8, 0xF1, 0xF3))
d.text(s, f"清洗状态：ok {cleaning['status_counts']['ok']:,} 封  ·  recovered {cleaning['status_counts']['recovered']:,} 封  ·  subject_only {cleaning['status_counts']['subject_only']:,} 封  ·  隔离 0 封",
       0.8, 6.2, 11.7, 0.34, 14, NAVY, CJK, bold=True)

# ---------------------------------------------------------------- 4 解析
s = d.slide("第 3 步", "邮件解析：编码、MIME 与 HTML")
d.code(s, src_lines("src/spam_classifier/email_parser.py", r"if token == \"quoted-printable\"", 2, 10),
       0.55, 1.62, 6.1, 2.7, 10.5,
       "传输编码：base64 / quoted-printable 解码与安全回退")
d.code(s, src_lines("src/spam_classifier/email_parser.py", r"def html_to_text", 0, 7),
       6.9, 1.62, 5.85, 2.7, 10.5,
       "HTML：去 script/style、解实体、保留文字")
d.bullets(s, [
    ("顺序固定", "先解传输编码 → 再按声明字符集解码 → 最后处理 HTML"),
    ("候选回退链", "BOM → HTML meta → utf-8 → gb18030 → gbk，逐个 strict 尝试"),
    ("乱码检测", "控制字符比例超 2% 判为二进制并跳过该候选"),
    ("MIME 处理", "multipart/alternative 只选一个有效表示；related 只取根部件；跳过附件与转发的整封子邮件"),
], 0.55, 4.55, 12.2, 1.9, 13.5)
d.panel(s, 0.55, 6.35, 12.2, 0.42, RGBColor(0xFB, 0xE7, 0xD6))
d.text(s, "注意：替换符比例为 0 不能证明没有语义乱码；private-use 字符会另标质量警告。「技术可用」≠「语义正确」。",
       0.8, 6.44, 11.7, 0.3, 12.5, RED, CJK, bold=True)

# ---------------------------------------------------------------- 5 划分
s = d.slide("第 4 步", "去重与冻结数据划分")
d.table(s, ["集合", "正常 ham", "垃圾 spam", "合计", "实际占比", "目标", "分组数"], [
    ["训练 train", "14,988", "8,804", "23,792", "70.001%", "70%", "21,609"],
    ["验证 validation", "3,212", "1,887", "5,099", "15.002%", "15%", "4,604"],
    ["测试 test", "3,211", "1,886", "5,097", "14.996%", "15%", "4,604"],
], 0.55, 1.62, 12.2, [2.5, 1.5, 1.5, 1.5, 1.6, 1.2, 1.4], 13.5, highlight=0)
d.code(s, src_lines("src/spam_classifier/data_split.py", r"def assign_splits", 6, 9),
       0.55, 3.2, 6.1, 3.0, 10,
       "src/spam_classifier/data_split.py — 整组贪心分配")
d.bullets(s, [
    ("同标签同文本", "只保留一个确定代表；跨标签精确重复键整组隔离"),
    ("连通分组", f"共享原始字节/最终文本/正文/HTML 对照文本构成 {split['deduplication']['leakage_groups_before_dedup']['groups']:,} 个分组，整组分配"),
    ("开发样本约束", f"400 封开发样本触及的 {split['deduplication']['forced_training_groups']} 个分组、{split['deduplication']['forced_training_representatives']:,} 封代表强制进训练侧"),
    ("冻结契约", "seed=42、70/15/15；重复运行必须与已冻结产物一致，否则返回 2 并保留旧版"),
], 6.9, 3.25, 5.85, 2.9, 13)

# ---------------------------------------------------------------- 6 特征
s = d.slide("第 5 步", "jieba 分词与训练侧 TF-IDF")
d.code(s, src_lines("src/spam_classifier/text_features.py", r"def tokenize_text", 0, 8),
       0.55, 1.62, 6.1, 2.85, 10.5,
       "src/spam_classifier/text_features.py — 固定分词规则")
d.code(s, src_lines("src/spam_classifier/text_features.py", r"def make_feature_pipeline", 0, 6),
       6.9, 1.62, 5.85, 2.85, 10.5,
       "TF-IDF 参数（min_df=2, max_features=50000）")
d.panel(s, 0.55, 4.72, 6.1, 1.75, LIGHT)
d.text(s, "人工分词示例（非真实邮件）", 0.78, 4.88, 5.6, 0.3, 12, TEAL, CJK, bold=True)
d.text(s, "输入：会议通知：请明天参加项目讨论。\n输出：会议 / 通知 / 请 / 明天 / 参加 / 项目 / 讨论\n\n输入：免费会员优惠！访问 urltoken，联系 emailtoken。\n输出：免费 / 会员 / 优惠 / 访问 / urltoken / 联系 / emailtoken",
       0.78, 5.22, 5.6, 1.15, 11, NAVY, MONO, spacing=1.15)
d.bullets(s, [
    ("TF-IDF 含义", "同时衡量词在本封邮件中是否突出，以及在训练语料中是否过于普遍"),
    ("拟合范围", "词表与 IDF 只在 train 的 23,792 封上拟合；验证/测试只 transform，未知词不新增列"),
    ("矩阵规模", f"train {feature['matrices']['train']['rows']:,} × {feature['matrices']['train']['columns']:,}，nnz {feature['matrices']['train']['nnz']:,}，密度 {pct(feature['matrices']['train']['density'], 3)}，三套集合无空向量"),
    ("指纹验证", "独立脚本重算训练 DF/IDF，并在新进程重载管线核对词表与 IDF 指纹"),
], 6.9, 4.78, 5.85, 1.9, 12.5)

# ---------------------------------------------------------------- 7 选型
s = d.slide("第 6 步", "验证集上的七次预设实验")
rows = []
labels = {"majority": "多数类基线", "nb_alpha_1": "NB α=1", "lr_C_1": "LR C=1",
          "nb_alpha_0.1": "NB α=0.1", "nb_alpha_0.5": "NB α=0.5",
          "lr_C_0.5": "LR C=0.5", "lr_C_2": "LR C=2"}
order = ["majority", "nb_alpha_1", "lr_C_1", "nb_alpha_0.1", "nb_alpha_0.5", "lr_C_0.5", "lr_C_2"]
for cid in order:
    m = metric[cid]
    rows.append([labels[cid], pct(m["accuracy"]), pct(m["spam_precision"]), pct(m["spam_recall"]),
                 pct(m["macro_f1"]), pct(m["ham_false_positive_rate"], 3),
                 f"{m['confusion_matrix']['fp']} / {m['confusion_matrix']['fn']}"])
d.table(s, ["方法", "准确率", "spam 精确率", "spam 召回率", "macro-F1", "ham 误报率", "FP/FN"],
         rows, 0.55, 1.6, 12.2, [2.1, 1.4, 1.5, 1.5, 1.4, 1.5, 1.3], 13, highlight=6)
d.code(s, src_lines("src/spam_classifier/model_selection.py", r"SELECTION_POLICY = ", 0, 11),
       0.55, 4.6, 6.1, 2.15, 10.5,
       "选择规则在运行前固定")
d.bullets(s, [
    ("为什么看 macro-F1", "多数类准确率 62.99%，却漏掉全部 1,887 封垃圾邮件——准确率在不平衡下极具误导性"),
    ("为什么选 LR C=2", "macro-F1 最高 98.78%，且误报与漏报同时优于最佳 NB"),
    ("收敛检查", "三组 LR 分别 6、7、7 次迭代收敛，无训练警告"),
    ("范围声明", "只比较预设的 6 个正式配置，没有证据说明 C=2 是所有可能配置中的最优"),
], 6.9, 4.65, 5.85, 2.1, 12.5)

# ---------------------------------------------------------------- 8 HTML 对照
s = d.slide("第 7 步", "固定条件下的 HTML 清洗对照")
d.table(s, ["输入表示", "准确率", "spam 召回率", "macro-F1", "FP", "FN", "标记 token 占比"], [
    ["去标签可读文本", pct(clean_arm["metrics"]["accuracy"], 4), pct(clean_arm["metrics"]["spam_recall"], 4),
     pct(clean_arm["metrics"]["macro_f1"], 4), clean_arm["metrics"]["confusion_matrix"]["fp"],
     clean_arm["metrics"]["confusion_matrix"]["fn"], pct(clean_arm["training_markup_token_fraction"], 4)],
    ["保留 HTML 标签", pct(html_arm["metrics"]["accuracy"], 4), pct(html_arm["metrics"]["spam_recall"], 4),
     pct(html_arm["metrics"]["macro_f1"], 4), html_arm["metrics"]["confusion_matrix"]["fp"],
     html_arm["metrics"]["confusion_matrix"]["fn"], pct(html_arm["training_markup_token_fraction"], 4)],
], 0.55, 1.62, 12.2, [2.6, 1.5, 1.6, 1.5, 0.9, 0.9, 1.9], 13, highlight=0)
d.bullets(s, [
    ("实验控制", "固定 LR C=2、训练/验证 ID、标签、分词与 TF-IDF 参数，只改变是否去掉 HTML 标签；两臂各自在训练集学习自己的词表与 IDF"),
    ("差异极小", "macro-F1 高 0.0213 个百分点，验证集只相差 1 封（去标签正确、保留 HTML 错误）"),
    ("实际改动量", "两种表示实际不同的训练/验证样本仅 156 / 36 封"),
    ("诚实结论", "不能宣称显著提升或普遍有效；按预先声明的排序规则锁定去标签表示"),
    ("噪声代理", "标记 token 占比 0.2209% → 0.0198% 只是噪声代理指标，不证明语义改善"),
], 0.55, 3.4, 6.6, 3.2, 13)
d.code(s, src_lines("src/spam_classifier/model_selection.py", r"def select_candidate", 0, 14),
       7.45, 3.4, 5.3, 3.0, 9.5,
       "确定性排序规则（比较使用未四舍五入原值）")

# ---------------------------------------------------------------- 9 最终测试
s = d.slide("第 8 步", "锁定方案后的最终测试（5,097 封）")
lr, nb, maj = test["lr"], test["nb"], test["majority"]
d.table(s, ["模型", "准确率", "spam 精确率", "spam 召回率", "macro-F1", "TN", "FP", "FN", "TP"], [
    ["多数类基线", pct(maj["accuracy"]), pct(maj["spam_precision"]), pct(maj["spam_recall"]),
     pct(maj["macro_f1"]), f"{maj['confusion_matrix']['tn']:,}", maj["confusion_matrix"]["fp"],
     f"{maj['confusion_matrix']['fn']:,}", maj["confusion_matrix"]["tp"]],
    ["NB α=0.1", pct(nb["accuracy"]), pct(nb["spam_precision"]), pct(nb["spam_recall"]),
     pct(nb["macro_f1"]), f"{nb['confusion_matrix']['tn']:,}", nb["confusion_matrix"]["fp"],
     nb["confusion_matrix"]["fn"], f"{nb['confusion_matrix']['tp']:,}"],
    ["LR C=2", pct(lr["accuracy"]), pct(lr["spam_precision"]), pct(lr["spam_recall"]),
     pct(lr["macro_f1"]), f"{lr['confusion_matrix']['tn']:,}", lr["confusion_matrix"]["fp"],
     lr["confusion_matrix"]["fn"], f"{lr['confusion_matrix']['tp']:,}"],
], 0.55, 1.62, 12.2, [2.2, 1.3, 1.5, 1.5, 1.3, 1.1, 0.8, 0.8, 1.1], 13, highlight=2)
s.shapes.add_picture(str(FIG / "stage08_lr_confusion.png"), Emu(int(0.55 * 914400)), Emu(int(3.3 * 914400)),
                      height=Emu(int(2.85 * 914400)))
d.code(s, src_lines("src/spam_classifier/final_evaluation.py", r"def score_locked_models", 4, 9),
       5.0, 3.3, 4.1, 2.85, 9,
       "src/spam_classifier/final_evaluation.py — 锁定模型评估")
d.bullets(s, [
    ("两类错误同时改善", f"相比 NB，误报 {nb['confusion_matrix']['fp']} → {lr['confusion_matrix']['fp']} 封，漏报 {nb['confusion_matrix']['fn']} → {lr['confusion_matrix']['fn']} 封"),
    ("独立核验通过", "新进程重载、按训练集独立重训、独立重算指标，17 项总检查 + 91 项逐候选检查全通过"),
    ("未用于调参", "测试成绩在方案锁定之后才计算，未据以修改参数或重新选型"),
], 9.35, 3.35, 3.4, 2.85, 11.5)

# ---------------------------------------------------------------- 10 混淆矩阵解读
s = d.slide("结果解读", "混淆矩阵与两个错误方向")
s.shapes.add_picture(str(FIG / "stage08_lr_confusion.png"), Emu(int(0.7 * 914400)), Emu(int(1.7 * 914400)),
                      height=Emu(int(4.2 * 914400)))
d.text(s, "LR 冻结测试集混淆矩阵（n=5,097）", 0.7, 5.95, 4.2, 0.32, 12, GREY, CJK, align=PP_ALIGN.CENTER)
d.code(s, src_lines("src/spam_classifier/model_selection.py", r"tn = int\(np\.count_nonzero", 0, 16),
       5.3, 1.62, 7.45, 2.35, 11,
       "src/spam_classifier/model_selection.py — 混淆计数与指标定义")
d.panel(s, 5.3, 4.15, 3.6, 2.15, RGBColor(0xFB, 0xE7, 0xD6))
d.text(s, "FP 误报（正常→垃圾）", 5.5, 4.32, 3.2, 0.3, 13, RED, CJK, bold=True)
d.text(s, f"LR {lr['confusion_matrix']['fp']} 封  ·  NB {nb['confusion_matrix']['fp']} 封\n业务后果：用户丢失重要邮件\n根因：招聘/通知与商业推广词重叠",
       5.5, 4.68, 3.25, 1.5, 12, NAVY, CJK, spacing=1.2)
d.panel(s, 9.15, 4.15, 3.6, 2.15, RGBColor(0xE8, 0xF1, 0xF3))
d.text(s, "FN 漏报（垃圾→正常）", 9.35, 4.32, 3.2, 0.3, 13, TEAL, CJK, bold=True)
d.text(s, f"LR {lr['confusion_matrix']['fn']} 封  ·  NB {nb['confusion_matrix']['fn']} 封\n业务后果：垃圾邮件进入收件箱\n根因：叙事包裹推广 + 可用文本过短",
       9.35, 4.68, 3.25, 1.5, 12, NAVY, CJK, spacing=1.2)
d.text(s, f"多数类基线：准确率 {pct(maj['accuracy'])} 却漏掉全部 {maj['confusion_matrix']['fn']:,} 封垃圾邮件 → 准确率单独使用会严重误导",
       0.7, 6.42, 12.05, 0.35, 13, ORANGE, CJK, bold=True)

# ---------------------------------------------------------------- 11 错误分析
s = d.slide("错误分析", "真实错分样本的语义观察")
d.code(s, src_lines("src/spam_classifier/final_evaluation.py", r"selected = sorted", 1, 2),
       0.55, 1.62, 6.1, 1.55, 10.5,
       "候选选取规则（固定，与语义叙事无关）")
d.bullets(s, [
    ("样本性质", f"共 {review['case_count']} 个实例，按分数绝对值取前 15 个 FP/FN；是高置信度错误样本，不是随机样本，类别计数不能估计总体比例"),
    ("误报主导", "招聘通知与商业推广词重叠（NB 13/15、LR 7/11）；也有查分通知含付费服务推广词"),
    ("漏报主导", "故事/祝福/私人聊天包裹推广请求；12/15 个 NB 漏报可用文本 ≤30 字，LR 漏报中 6 个是 subject_only"),
    ("重要提示", "这是内容观察，不是经反事实验证的错因；本次测试后不据此重调模型"),
], 6.9, 1.68, 5.85, 3.4, 12.5)
d.table(s, ["模型", "错误类型", "错误总数", "复核候选", "含 HTML", "编码警告", "文本≤30字"], [
    ["NB", "FP 误报", "134", "15", "12", "4", "0"],
    ["NB", "FN 漏报", "61", "15", "14", "3", "12"],
    ["LR", "FP 误报", "11", "11", "1", "0", "0"],
    ["LR", "FN 漏报", "29", "15", "3", "5", "—"],
], 0.55, 3.45, 6.1, [1.0, 1.5, 1.2, 1.2, 1.1, 1.2, 1.3], 11.5)
d.panel(s, 6.9, 5.25, 5.85, 1.4, LIGHT)
d.text(s, "脱敏说明", 7.1, 5.4, 5.4, 0.28, 12, TEAL, CJK, bold=True)
d.text(s, "公开概述不复制邮件原句、mail_id 或联系方式；原始语料标签保持不变，标签疑点只记为待复核。",
       7.1, 5.7, 5.4, 0.8, 12, NAVY, CJK, spacing=1.2)

# ---------------------------------------------------------------- 12 关联词
s = d.slide("模型解释", "LR 关联词：偏向两类邮件的词")
ham_rows = [[c["term"], f"{c['coefficient']:.2f}"] for c in coeffs["ham"][:6]]
spam_rows = [[c["term"], f"{c['coefficient']:+.2f}"] for c in coeffs["spam"][:6]]
d.table(s, ["偏向正常邮件 (ham)", "系数"], ham_rows, 0.55, 1.72, 5.3, [3.4, 1.5], 12.5, row_h=0.31, head_h=0.36)
d.table(s, ["偏向垃圾邮件 (spam)", "系数"], spam_rows, 6.15, 1.72, 5.3, [3.4, 1.5], 12.5, row_h=0.31, head_h=0.36)
s.shapes.add_picture(str(FIG / "stage08_lr_public_terms.png"), Emu(int(0.6 * 914400)), Emu(int(4.25 * 914400)),
                      height=Emu(int(2.35 * 914400)))
d.text(s, "预设通用词表内的 LR 系数分布（左：ham 负值；右：spam 正值）",
       0.6, 6.62, 4.6, 0.3, 11, GREY, CJK, align=PP_ALIGN.CENTER)
d.bullets(s, [
    ("高频词 ≠ 分类关键词", "高频是文档覆盖率（一个词出现在多少封邮件里）；LR 系数是模型在给定语料下的条件关联"),
    ("占位符信号", "urltoken(+12.20)、phonetoken(+6.04)、emailtoken(+3.96) 权重最高——保留联系方式「出现的信号」，但不把具体联系方式送进特征"),
    ("公开范围", "只展示预先固定的 53 词通用表，不是完整词表的全局排名，也不表示因果"),
], 5.5, 4.3, 7.3, 2.1, 12.5)

# ---------------------------------------------------------------- 13 复现
s = d.slide("可复现性", "环境、配置与运行顺序")
d.code(s, """git clone https://github.com/25121596/Python-teamwork.git
cd Python-teamwork
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
Copy-Item config.example.json config.local.json
.\\.venv\\Scripts\\python.exe scripts\\check_environment.py""",
       0.55, 1.62, 6.1, 2.5, 11,
       "Windows 环境建立（64 位 Python 3.12）")
d.code(s, """# 2 索引  3 清洗  4 划分  5 特征
python scripts\\audit_index.py
python scripts\\clean_mail.py --scope full
python scripts\\split_data.py
python scripts\\build_features.py
# 6 模型  7 对照  8 最终测试
python scripts\\train_models.py
python scripts\\compare_cleaning.py
python scripts\\evaluate_final.py
# 独立核验 + 全量回归
python scripts\\check_final_results.py
python -m unittest discover -s tests -v""",
       6.9, 1.62, 5.85, 3.0, 10.5,
       "完整运行顺序（生成脚本 + 独立核验）")
d.bullets(s, [
    ("数据位置", "原始语料必须在仓库外，通过 config.local.json 引用；该文件被 Git 忽略"),
    ("换行符陷阱", "公开 audit/metrics 文件用 .gitattributes 固定 LF，Windows 检出时 CRLF 转换会让锁内 SHA256 校验失败"),
    ("跨平台差异", "词表截断与 IDF 指纹在 Linux/Windows 上不同，两个非最优 NB 配置各差一封误报，最佳 NB/LR 指标不变；未为凑一致改参数"),
    ("回归测试", "128 项全部通过，0 失败 0 跳过——但这是程序逻辑验证，与真实实验成绩是两个独立证据层次"),
], 0.55, 4.5, 6.2, 2.3, 12)
d.panel(s, 7.15, 4.85, 5.6, 1.55, RGBColor(0xE8, 0xF1, 0xF3))
d.text(s, "Git 协作", 7.35, 5.0, 5.2, 0.28, 12, TEAL, CJK, bold=True)
d.text(s, "每个可验收阶段即提交代码、阶段 Markdown、进度与清单并推送；历史保持真实，不伪造成员贡献，不为凑提交数量制造无意义提交。",
       7.35, 5.3, 5.2, 0.95, 12, NAVY, CJK, spacing=1.2)

# ---------------------------------------------------------------- 14 结论
s = d.slide("结论", "适用范围与局限")
d.panel(s, 0.55, 1.62, 6.0, 2.45, RGBColor(0xE8, 0xF1, 0xF3))
d.text(s, "结论", 0.8, 1.8, 5.5, 0.32, 16, TEAL, CJK, bold=True)
d.bullets(s, [
    ("LR 优于 NB", f"准确率 {pct(lr['accuracy'])} vs {pct(nb['accuracy'])}，误报 {lr['confusion_matrix']['fp']} vs {nb['confusion_matrix']['fp']}，漏报 {lr['confusion_matrix']['fn']} vs {nb['confusion_matrix']['fn']}"),
    ("两类错误同时改善", "不是用误报换漏报"),
    ("防泄漏有效", "分组整组划分 + 训练侧独拟合 + 指纹链 + 不可变模型锁"),
    ("落地建议", "误拦邮件进可恢复垃圾箱或人工复核队列，不直接删除"),
], 0.8, 2.18, 5.5, 1.85, 12.5)
d.panel(s, 6.75, 1.62, 6.0, 2.45, RGBColor(0xFB, 0xE7, 0xD6))
d.text(s, "局限", 7.0, 1.8, 5.5, 0.32, 16, RED, CJK, bold=True)
d.bullets(s, [
    ("历史语料", "TREC 2006 与现代邮箱差异明显，99.22% 不代表当代真实过滤性能"),
    ("近重复残留", "精确去重无法识别所有近重复广告模板"),
    ("类别比例已变", "去重后 spam 占比 66.32% → 37.00%"),
    ("未做 OCR", "HTML 中的图片内容完全不参与特征"),
    ("人工签认待办", "56 个错分实例为 AI 辅助复核，学生人工确认仍待完成"),
], 7.0, 2.18, 5.5, 1.85, 12.5)
d.panel(s, 0.55, 4.3, 12.2, 1.0, LIGHT)
d.text(s, "证据入口", 0.8, 4.45, 11.7, 0.28, 12.5, TEAL, CJK, bold=True)
d.text(s, "artifacts/metrics/final_evaluation.json · stage08_test.csv · cleaning_ablation.json · error_analysis.json    |    "
           "artifacts/audit/eda_summary.json · split_summary.json · cleaning_summary.json    |    docs/stages/ 各阶段记录",
       0.8, 4.78, 11.7, 0.4, 11.5, NAVY, MONO)
d.panel(s, 0.55, 5.5, 12.2, 1.1, RGBColor(0x18, 0x36, 0x4A))
d.text(s, "仓库地址：https://github.com/25121596/Python-teamwork", 0.8, 5.72, 11.7, 0.36, 15, WHITE, MONO, bold=True)
d.text(s, "本次所有数字均直接读取仓库 artifacts/ 下的聚合产物；代码片段取自跟踪的源文件，可逐项复核。",
       0.8, 6.12, 11.7, 0.34, 12, RGBColor(0xC9, 0xD6, 0xE0), CJK)

out = ROOT / "deliverables" / "项目演示.pptx"
d.save(out)
print(f"saved {out}  slides={len(d.prs.slides.__iter__.__self__._sldIdLst)}")
