/** Editable 8-slide final-experiment presentation from verified public evidence.
 * Run a copied builder in a scratch directory containing a node_modules link
 * to CODEX_PRIMARY_RUNTIME_NODE_MODULES; requires bundled Artifact Tool.
 * node build_presentation.mjs ABS_REPO_PATH ABS_BUILD_PATH
 * Build paths and private QA are deliberately outside the repository.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const root = path.resolve(process.argv[2]);
const buildDir = path.resolve(process.argv[3]);
const skill = '/root/.codex/skills/builtins/presentations';
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const FONT = 'Noto Sans CJK SC';
const navy = '#18364A', teal = '#187A87', orange = '#C56A32', grey = '#526372';
const presentation = Presentation.create({slideSize:{width:1280,height:720}});
const eda = JSON.parse(await fs.readFile(path.join(root,'artifacts/audit/eda_summary.json'),'utf8'));
const selection = JSON.parse(await fs.readFile(path.join(root,'artifacts/metrics/model_selection.json'),'utf8'));
const metric = id => selection.experiments.find(x=>x.candidate_id===id).metrics;
const ablation=JSON.parse(await fs.readFile(path.join(root,'artifacts/metrics/cleaning_ablation.json'),'utf8'));
const final=JSON.parse(await fs.readFile(path.join(root,'artifacts/metrics/final_evaluation.json'),'utf8'));
const audit=JSON.parse(await fs.readFile(path.join(root,'artifacts/audit/final_evaluation_validation.json'),'utf8'));
const review=JSON.parse(await fs.readFile(path.join(root,'artifacts/metrics/error_review_assisted.json'),'utf8'));
if(!audit.passed || !final.passed || review.ai_assisted_review_status!=='complete') throw new Error('Real verified evidence required');
const test=final.metrics, lr=test.lr, nb=test.nb, cm=lr.confusion_matrix;
const clean=ablation.comparisons.find(x=>x.representation==='text_clean');
const html=ablation.comparisons.find(x=>x.representation==='text_unstripped_html');
const selected=final.representation==='text_clean'?'去标签可读文本':'保留 HTML 文本';
const pct = n => (n*100).toFixed(2)+'%';
const content = [
 {title:'中文垃圾邮件分类',time:'0:00—0:20',source:'artifacts/metrics/final_evaluation.json',
 speech:`大家好，我们的项目是中文垃圾邮件分类。我们从原始邮件解析开始，比较朴素贝叶斯和逻辑回归，重点关注误拦正常邮件和漏掉垃圾邮件。锁定方案后的测试结果表明，逻辑回归准确率为${pct(lr.accuracy)}，但仍存在错误。下面说明数据、实验过程和这些数字的适用范围。`},
 {title:'去重改变了语料的类别构成',time:'0:20—0:55',source:'artifacts/audit/eda_summary.json',
 speech:'课程语料共有六万四千六百二十封邮件，垃圾邮件约占百分之六十六点三。主索引和延迟索引涉及相同邮件，因此只使用主索引。清洗后全部邮件技术可用，按规范化文本精确去重后剩三万三千九百八十八封代表。正常邮件保留约百分之九十八点四，垃圾邮件只保留百分之二十九点三，说明垃圾邮件的重复更多。去重后垃圾邮件占百分之三十七，所以最终成绩描述不同文本代表，不能直接代表自然收件箱。'},
 {title:'解析质量与数据划分',time:'0:55—1:35',source:'artifacts/audit/cleaning_summary.json；artifacts/audit/split_summary.json',
 speech:'解析先解传输编码，再按字符集解码并记录回退。MIME 多部分只选有效正文，避免纯文本和 HTML 重复拼接。HTML 去脚本和标签，保留文字，常见联系字段再归一化。约三万封常规可读，三万三千多封需要恢复，一千四百二十二封只有主题可用。技术可用不证明语义完全正确。相同正文或对照文本关联的样本整组划分，开发样本及关联组只进入训练。训练约二万三千八百封，验证和测试各约五千一百封。'},
 {title:'训练特征与验证模型比较',time:'1:35—2:15',source:'artifacts/metrics/model_selection.json；artifacts/audit/feature_summary.json',
 speech:'中文使用 jieba 精确分词和固定停用词。TF-IDF 兼顾词在本封邮件中是否突出，以及在训练语料中是否过于普遍。五万维词表和 IDF 只从训练集学习，验证与测试只能转换。朴素贝叶斯用类别词分布分类，逻辑回归联合学习词权重。两者各比较三个参数，按验证 macro-F1、误报率和召回率选择。NB 最佳 alpha 为零点一，LR 最佳 C 为二。多数类准确率接近百分之六十三，却漏掉全部垃圾邮件，说明准确率不能单独使用。'},
 {title:'固定条件下的 HTML 清洗对照',time:'2:15—2:45',source:'artifacts/metrics/cleaning_ablation.json；artifacts/metrics/final_lock.json',
 speech:`清洗对照固定 LR C 等于二、相同训练验证样本与其他配置，只改变是否去掉 HTML 标签。两臂分别只在训练集学习 TF-IDF，去标签臂还必须重现上一步预测。去标签的验证 macro-F1 为${pct(clean.metrics.macro_f1)}，保留 HTML 为${pct(html.metrics.macro_f1)}。按预先规则选取${selected}，把模型、参数和文件指纹锁定后，再打开最终测试。实际改变的训练和验证输入仅一百五十六封和三十六封，净少漏报一封，不能宣称普遍或显著提升。`},
 {title:'锁定方案后的最终测试',time:'2:45—3:45',source:'artifacts/metrics/stage08_test.csv；artifacts/audit/final_evaluation_validation.json',
 speech:`这里是五千零九十七封最终测试邮件，含三千二百一十一封正常邮件和一千八百八十六封垃圾邮件。LR 准确率为${pct(lr.accuracy)}，macro-F1 为${pct(lr.macro_f1)}，垃圾邮件精确率为${pct(lr.spam_precision)}，召回率为${pct(lr.spam_recall)}。NB 误拦${nb.confusion_matrix.fp}封正常邮件，漏掉${nb.confusion_matrix.fn}封垃圾邮件；LR 分别为${cm.fp}和${cm.fn}封。混淆矩阵的行是真实标签，列是预测。LR 正常邮件误报率为${pct(lr.ham_false_positive_rate)}。独立重算指标和重载模型预测均通过核验。测试结果没有被用于改参数或重新选择方案。`},
 {title:'真实错分与关键词解释',time:'3:45—4:30',source:'docs/真实错分分析.md；artifacts/metrics/error_analysis.json',
 speech:`我们为 NB 和 LR 各按分数绝对值选择最多十五个误报和十五个漏报，共${review.case_count}个模型错误实例。已经完成 AI 辅助语义核查，长邮件检查首中尾选段，仍需成员人工确认。屏幕上展示脱敏后的语义观察，正文和联系方式不公开。这是高置信度错误样本，不能用类别数量估算全部错误的比例。高频词表示经常出现，LR 系数表示模型中的分类关联，两者不同。公开词图只覆盖预设通用词表，权重也不代表因果。`},
 {title:'结论与适用范围',time:'4:30—5:00',source:'artifacts/metrics/final_evaluation.json；README.md；docs/成员分工.md',
 speech:`在本语料和冻结划分下，LR C 等于二的最终表现优于本轮 NB 基线，但仍误拦${cm.fp}封正常邮件。因此更适合先进入可恢复垃圾箱或复核队列。局限包括旧语料、可能残留的近重复和编码问题，以及去重后类别比例变化。未来改进需要新验证方案和新测试集。本项目按真实阶段保留 Git 提交，成员分工需要与本人实际工作相符。感谢大家。`}
];

function text(slide, value, x,y,w,h,size=28,color=navy,bold=false,align='left') {
  const shape=slide.shapes.add({name:'text-'+slide.shapes.items.length,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0,lineSpacing:1.12};
  return shape;
}
function slideAt(i) {
  const slide=presentation.slides.add(); slide.background.fill='#FFFFFF';
  text(slide,'PYTHON 数据分析编程  /  第六题',64,30,950,28,18,grey);
  if(i>0) text(slide,content[i].title,64,91,1152,78,43,navy,true);
  text(slide,'TREC06C  实验结果与误差分析',64,668,1000,26,18,grey);
  text(slide,`${String(i+1).padStart(2,'0')} / 08`,1080,668,136,26,18,grey,false,'right');
  slide.speakerNotes.text=`建议时段：${content[i].time}\n\n${content[i].speech}\n\n来源：${content[i].source}\n仓库：https://github.com/25121596/Python-teamwork\n事实范围：真实验证、清洗对照与锁定后的最终测试；AI 辅助错分观察待成员确认。`;
  return slide;
}
function table(slide, values, x,y,widths,rowHeight=62,fontSize=24) {
  const t=slide.tables.add({rows:values.length,columns:widths.length,left:x,top:y,width:widths.reduce((a,b)=>a+b,0),height:values.length*rowHeight,columnWidths:widths,values});
  t.borders.assign({style:'solid',fill:'#D9E2E7',width:1});
  for(let r=0;r<values.length;r++) {
    t.rows[r].height=rowHeight;
    t.cells.block({row:r,column:0,rowCount:1,columnCount:widths.length}).assign({fill:r===0?navy:r%2?'#EFF4F6':'#FFFFFF',textStyle:{typeface:FONT,fontSize,color:r===0?'#FFFFFF':navy,bold:r===0,alignment:'center',verticalAlignment:'middle'},margins:{top:10,bottom:10,left:14,right:14},anchor:'center'});
  }
  return t;
}

{
  const s=slideAt(0);
  text(s,'中文垃圾邮件分类',64,188,1152,100,64,navy,true);
  text(s,'原始邮件解析与模型比较',68,310,1100,64,34,teal);
  text(s,'TREC06C  ·  jieba / TF-IDF  ·  Naive Bayes / Logistic Regression',68,411,1100,80,25,grey);
  text(s,'研究问题：如何同时减少误报与漏报？',68,536,1120,52,31,navy);
}
{
  const s=slideAt(1),c=eda.counts;
  const chart=s.charts.add('bar',{position:{left:48,top:201,width:810,height:407},title:'原始与去重代表数量（封）',categories:['原始语料','去重代表'],series:[{name:'正常邮件',values:[c.raw_indexed.ham,c.deduplicated_representatives.ham],fill:teal},{name:'垃圾邮件',values:[c.raw_indexed.spam,c.deduplicated_representatives.spam],fill:orange}],hasLegend:true,barOptions:{direction:'column',grouping:'clustered',gapWidth:110},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:21}},yAxis:{min:0,max:50000,majorUnit:10000,numberFormatCode:'#,##0',textStyle:{fontSize:20},majorGridlines:{style:'solid',fill:'#DCE4E8',width:1}},xAxis:{textStyle:{fontSize:24}},legend:{position:'bottom',textStyle:{fontSize:22}},titleTextStyle:{fontSize:24}});
  applyPresentationChartFont(chart,{fontFamily:FONT});
  text(s,'64,620',907,218,300,77,56,navy,true);
  text(s,'原始邮件',910,297,280,45,26,grey);
  text(s,'33,988',907,378,300,77,56,teal,true);
  text(s,'不同规范化文本代表',910,458,300,70,26,grey);
  text(s,'spam 占比：66.32% → 37.00%',64,611,1152,40,26,navy);
}
{
  const s=slideAt(2);
  text(s,'先解传输编码，再解字符编码；MIME 避免重复拼接。',64,187,1152,48,28);
  table(s,[['解析状态','邮件数','含义'],['ok','29,891','常规处理可用'],['recovered','33,307','回退或结构恢复后可用'],['subject_only','1,422','仅保留有效主题'] ],64,265,[310,220,622],65,27);
  text(s,'训练 23,792  /  验证 5,099  /  最终测试 5,097',64,559,1152,52,27,teal);
  text(s,'相同内容整组划分，开发样本关联组只进入训练侧。',64,615,1152,40,25,grey);
}
{
  const s=slideAt(3);
  text(s,'jieba 分词，训练侧 50,000 维 TF-IDF',64,191,1152,54,33,teal,true);
  text(s,'验证与测试只转换；相同训练样本与特征规则。',64,260,1152,48,28);
  table(s,[['验证模型','准确率','macro-F1','正常误报'],...['majority','nb_alpha_0.1','lr_C_2'].map((id,i)=>{let m=metric(id);return[['多数类','NB α=0.1','LR C=2'][i],pct(m.accuracy),pct(m.macro_f1),String(m.confusion_matrix.fp)];})],64,335,[360,264,264,264],63,27);
  text(s,'验证 n = 5,099；选参先看 macro-F1，再看误报率。',64,612,1152,41,26,grey);
}
{
  const s=slideAt(4);
  text(s,'只改变 HTML 表示，固定 LR C=2 和训练验证 ID',64,192,1152,58,31,teal,true);
  table(s,[['输入表示','验证准确率','验证 macro-F1','误报率'],...[[clean,'去标签'],[html,'保留 HTML']].map(([x,name])=>[name,pct(x.metrics.accuracy),pct(x.metrics.macro_f1),pct(x.metrics.ham_false_positive_rate)])],64,289,[360,264,264,264],73,27);
  text(s,`最终锁定：${selected}`,64,551,1152,53,32,teal,true);
  text(s,'独立重训核验通过；锁定之后才计算最终测试成绩。',64,614,1152,40,26,grey);
}
{
  const s=slideAt(5);
  table(s,[['测试模型','准确率','macro-F1','正常误报','垃圾漏报'],...['majority','nb','lr'].map((id,i)=>{let m=test[id];return[['多数类','NB α=0.1','LR C=2'][i],pct(m.accuracy),pct(m.macro_f1),String(m.confusion_matrix.fp),String(m.confusion_matrix.fn)];})],64,191,[290,215,215,216,216],59,26);
  text(s,'LR 测试混淆矩阵',64,452,535,42,27,teal,true);
  table(s,[['真实 / 预测','正常','垃圾'],['正常',String(cm.tn),String(cm.fp)],['垃圾',String(cm.fn),String(cm.tp)]],64,505,[225,155,155],43,23);
  text(s,pct(lr.ham_false_positive_rate),713,466,450,77,51,teal,true);
  text(s,'LR 正常邮件误报率',715,550,460,46,27,grey);
  text(s,'最终测试 n = 5,097；没有重新选参',715,604,490,40,23,grey);
}
{
  const s=slideAt(6);
  table(s,[['真实实例','语义观察'],...review.slide_examples.map(x=>[x.case_id,x.short_observation])],64,196,[260,892],64,26);
  text(s,`共 ${review.case_count} 个实例完成 AI 辅助阅读，待成员人工确认。`,64,535,1152,55,28,teal);
  text(s,'置信度排序样本不能代表全部错误；词频与模型系数含义不同。',64,607,1152,48,25,grey);
}
{
  const s=slideAt(7);
  text(s,'实验结论',64,199,1152,51,33,teal,true);
  text(s,`LR C=2 测试 macro-F1 ${pct(lr.macro_f1)}。\n仍有 ${cm.fp} 封正常邮件被误拦，建议可恢复垃圾箱或复核。`,64,267,1152,112,31);
  text(s,'适用范围',64,414,1152,48,33,teal,true);
  text(s,'历史语料、去重后的文本代表，不能直接外推实际邮件流。\n未来改模型或阈值，需要新验证方案与未见测试集。',64,477,1152,109,29);
  text(s,'github.com/25121596/Python-teamwork',64,611,1152,40,26,grey);
}

await fs.mkdir(buildDir,{recursive:true});
const candidatePath=path.join(buildDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(buildDir,'output','中文垃圾邮件分类_汇报.pptx');
await fs.mkdir(path.dirname(finalPath),{recursive:true});
await finalizePresentation({workspaceDir:buildDir,candidatePath,finalPath,pythonExecutable:process.env.CODEX_PRIMARY_RUNTIME_PYTHON,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,4,5,6,7].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:8,requiredNativeTableOwnerSlides:[3,4,5,6,7],requiredNativeChartOwnerSlides:[2],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(buildDir,'validation.json')});
for(let i=0;i<8;i++) {
  const blob=await presentation.export({slide:presentation.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(buildDir,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
await fs.copyFile(finalPath,path.join(root,'deliverables','汇报.pptx'));
const md=['# 5 分钟汇报讲稿','',
  '对应《汇报.pptx》8 页，已补入真实清洗对照、最终测试和 AI 辅助错分观察。计划时长合计 5 分钟，实际语速需彩排计时；成员人工复核、姓名与实际分工确认后加入开场和结尾，再录制正式视频。',''];
content.forEach((row,i)=>md.push(`## 第 ${i+1} 页 · ${row.time} · ${row.title}`,'',row.speech,'',`依据：${row.source}` ,''));
await fs.writeFile(path.join(root,'deliverables','五分钟汇报讲稿.md'),md.join('\n'));
console.log(JSON.stringify({pptx:path.join(root,'deliverables','汇报.pptx'),slides:8,validation:path.join(buildDir,'validation.json')}));
