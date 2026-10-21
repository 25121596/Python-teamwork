/** Editable 8-slide progress presentation from existing public evidence.
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
const pct = n => (n*100).toFixed(2)+'%';
const content = [
  {title:'中文垃圾邮件分类', time:'0:00—0:25', source:'项目计划.md；artifacts/metrics/model_selection.json',
   speech:'大家好，我们的项目是中文垃圾邮件分类。目标是把原始邮件整理成可靠的文本特征，比较朴素贝叶斯和逻辑回归，同时关注误拦正常邮件和漏掉垃圾邮件的代价。本次材料展示已有的数据处理与验证实验；清洗对照和最终测试尚待真实数据恢复后执行，下面所有模型成绩都属于验证集。'},
  {title:'去重改变了语料的类别构成', time:'0:25—1:05', source:'artifacts/audit/eda_summary.json；artifacts/audit/index_summary.json',
   speech:'课程提供的 TREC06C 主索引有六万四千六百二十封邮件，其中垃圾邮件约占百分之六十六点三。我们只使用 full/index，delay/index 涉及相同邮件，不能拿来当独立测试集。清洗后全部邮件技术可用，按相同内容去重后剩三万三千九百八十八封代表。正常邮件保留约百分之九十八点四，垃圾邮件只保留百分之二十九点三，说明垃圾邮件的精确重复更多。去重后垃圾邮件占百分之三十七，因此这里评估的是不同文本代表，而不是实际收件箱的自然邮件流。'},
  {title:'解析与清洗保留完整质量记录', time:'1:05—1:35', source:'docs/stages/03_邮件解析与全量清洗.md；artifacts/audit/cleaning_summary.json',
   speech:'邮件处理先解传输编码，再按字符集解码，并记录回退。MIME 多部分只选合适的正文，避免纯文本和 HTML 重复拼接，也跳过附件。HTML 去掉脚本和样式，保留可读文字，再归一化常见联系字段。最终约三万封常规解析成功，三万三千多封经过恢复，一千四百二十二封只有主题可用。这里的可用只是技术可读，不能证明语义完全正确，也不能把没有替换符当作没有乱码。'},
  {title:'按内容分组冻结训练、验证与测试', time:'1:35—1:55', source:'artifacts/audit/split_summary.json；docs/stages/04_去重与冻结数据划分.md',
   speech:'相同正文、原始字节或对照文本关联的邮件整组划分，防止同一内容同时出现在训练和测试中。调试解析器接触过的开发样本及其关联组固定到训练侧。训练约二万三千八百封，验证和测试各约五千一百封。训练学特征与模型，验证选方案，测试只用于最后评估。'},
  {title:'训练侧 TF-IDF 与两种分类模型', time:'1:55—2:35', source:'docs/分词与特征说明.md；src/spam_classifier/text_features.py；artifacts/metrics/model_selection.json',
   speech:'中文先用 jieba 精确分词，关闭 HMM，使用固定停用词。TF-IDF 同时考虑某个词在这封邮件里的出现情况，以及它在训练语料中是否过于常见；再把向量归一化。五万维词表和 IDF 只能在训练集拟合，验证和测试只能转换。朴素贝叶斯用词的类别分布判断邮件，逻辑回归联合学习词权重。我们给两者相同的样本和特征规则，各比较三个参数，并加入多数类作为最低参照。选参优先看 macro-F1，再看正常邮件误报率和垃圾邮件召回率。'},
  {title:'LR C=2 在现有验证集上表现最好', time:'2:35—3:40', source:'artifacts/metrics/stage06_validation.csv；artifacts/audit/model_validation.json',
   speech:'这里是五千零九十九封验证邮件的真实历史结果。多数类准确率接近百分之六十三，但它把全部垃圾邮件都漏掉，所以不能只看准确率。朴素贝叶斯最佳 alpha 为零点一，macro-F1 约百分之九十五点六；逻辑回归最佳 C 为二，macro-F1 约百分之九十八点七八。右下的混淆计数显示，逻辑回归把十一封正常邮件误拦为垃圾邮件，漏掉四十七封垃圾邮件；对应正常邮件误报率约百分之零点三四。与朴素贝叶斯的一百二十六次误报相比，它更好地保护了正常邮件。这些结果支持把 LR C 等于二作为下一步候选，但参数是在验证集上选择的，不能把这组数当作最终测试成绩。'},
  {title:'清洗效果与错因需要独立证据', time:'3:40—4:25', source:'docs/stages/07_HTML清洗对照与方案锁定.md；docs/stages/08_最终测试与错误分析流程.md；artifacts/figures/stage05_training_word_coverage.png',
   speech:'下一步清洗对照固定 LR C 等于二、相同训练验证样本及其他配置，只改变是否去除 HTML 标签。真实对照还没运行，因此暂时不能宣称去标签一定提升效果。最终方案锁定后才计算测试成绩，再分别检查误报和漏报。程序已能为 NB 和 LR 各选最多十五个误报和十五个漏报候选，但真实文本必须私下阅读，公开时只能用审核后的安全概述。另外，高频词只表示经常出现，LR 系数才反映模型里的分类关联；公开词图只覆盖预设通用词表，不能当成完整词表的排名。'},
  {title:'结论、局限与复现入口', time:'4:25—5:00', source:'README.md；进度记录.md；docs/成员分工.md',
   speech:'目前可以确认，编码和结构恢复需要质量记录，去重会明显改变类别构成，而 LR C 等于二在现有验证集上优于本轮 NB 候选。局限包括旧语料、可能残留的近重复模板，以及验证选参带来的乐观偏差。后续将恢复真实数据，完成清洗对照、最终测试和人工错分分析，再补齐成员实际分工与正式报告。代码、聚合证据和各阶段记录均由 Git 管理，仓库链接见屏幕。感谢大家。'},
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
  text(slide,'阶段汇报草稿 · 当前成绩均为验证集结果',64,668,1000,26,18,grey);
  text(slide,`${String(i+1).padStart(2,'0')} / 08`,1080,668,136,26,18,grey,false,'right');
  slide.speakerNotes.text=`建议时段：${content[i].time}\n\n${content[i].speech}\n\n来源：${content[i].source}\n仓库：https://github.com/25121596/Python-teamwork\n事实范围：既有阶段六真实验证结果；阶段七、八实测尚未执行。`;
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
  text(s,'从原始邮件到可核验的模型比较',68,310,1100,64,34,teal);
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
  text(s,'HTML 去脚本与标签，保留可读正文；常见联系字段归一化。',64,559,1152,52,27,teal);
  text(s,'技术可用 ≠ 语义完全正确',64,615,1152,40,25,grey);
}
{
  const s=slideAt(3);
  table(s,[['集合','正常邮件','垃圾邮件','合计'],...['train','validation','test'].map((name,i)=>{const r=eda.counts.split_counts[name];return[['训练','验证','测试'][i],r.ham.toLocaleString('en-US'),r.spam.toLocaleString('en-US'),r.total.toLocaleString('en-US')];})],64,206,[312,280,280,280],70,28);
  text(s,'相同内容整组划分，开发样本关联组只进入训练侧。',64,532,1152,52,29,teal);
  text(s,'训练学特征与模型  ·  验证选方案  ·  测试作最终评估',64,599,1152,47,27,grey);
}
{
  const s=slideAt(4);
  text(s,'词表与 IDF 只在训练集学习',64,195,1152,57,34,teal,true);
  text(s,'jieba 精确分词，HMM=False；固定停用词。\n50,000 维稀疏 TF-IDF；min_df=2；L2 归一化。\n验证与测试只转换，不参与拟合。',64,272,1152,132,29);
  table(s,[['方法','作用 / 参数'],['多数类','提供最低参照'],['MultinomialNB','alpha = 0.1 / 0.5 / 1'],['LogisticRegression','C = 0.5 / 1 / 2'] ],64,420,[430,722],50,25);
}
{
  const s=slideAt(5),ids=['majority','nb_alpha_0.1','lr_C_2'];
  const values=[['验证模型','准确率','macro-F1','正常误报','垃圾漏报'],...ids.map((id,i)=>{let m=metric(id);return[['多数类','NB α=0.1','LR C=2'][i],pct(m.accuracy),pct(m.macro_f1),String(m.confusion_matrix.fp),String(m.confusion_matrix.fn)];})];
  table(s,values,64,191,[290,215,215,216,216],59,26);
  text(s,'LR 验证混淆矩阵',64,452,535,42,27,teal,true);
  table(s,[['真实 / 预测','正常','垃圾'],['正常','3,201','11'],['垃圾','47','1,840']],64,505,[225,155,155],43,23);
  text(s,'0.34%',713,466,450,77,51,teal,true);
  text(s,'LR 正常邮件误报率',715,550,460,46,27,grey);
  text(s,'验证 n = 5,099；最终测试尚未运行',715,604,490,40,23,grey);
}
{
  const s=slideAt(6);
  table(s,[['要回答的问题','预定验证方式'],['HTML 清洗是否有效？','固定 LR C=2，只改变 HTML 处理'],['为何误报或漏报？','各类最多 15 个候选，私下逐例复核'],['哪些词与分类有关？','区分训练词覆盖率与 LR 系数'] ],64,199,[450,702],77,27);
  text(s,'真实对照与最终测试尚待执行，当前没有对应实测结论。',64,547,1152,63,29,teal);
  text(s,'高频不等于有判别力；公开图仅展示预设通用词表。',64,613,1152,42,25,grey);
}
{
  const s=slideAt(7);
  text(s,'已有证据',64,199,1152,51,33,teal,true);
  text(s,'质量恢复需记录；去重改变类别比例。\nLR C=2 在本轮验证集上优于 NB 候选。',64,267,1152,112,31);
  text(s,'适用范围与下一步',64,414,1152,48,33,teal,true);
  text(s,'旧语料与近重复仍有限制。\n恢复真实输入，完成对照、测试与人工错分复核。',64,477,1152,109,29);
  text(s,'github.com/25121596/Python-teamwork',64,611,1152,40,26,grey);
}

await fs.mkdir(buildDir,{recursive:true});
const candidatePath=path.join(buildDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const finalPath=path.join(buildDir,'output','中文垃圾邮件分类_汇报草稿.pptx');
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
await fs.copyFile(finalPath,path.join(root,'deliverables','汇报草稿.pptx'));
const md=['# 5 分钟汇报讲稿（阶段草稿）','',
  '对应《汇报草稿.pptx》8 页，总计划时长 5 分钟；实际语速需彩排计时。当前材料只含真实历史验证结果，必须完成真实清洗对照、最终测试和错分复核后替换相应内容，再录制正式课程视频。成员姓名与实际分工由小组确认后加入开场和结尾。',''];
content.forEach((row,i)=>md.push(`## 第 ${i+1} 页 · ${row.time} · ${row.title}`,'',row.speech,'',`依据：${row.source}` ,''));
await fs.writeFile(path.join(root,'deliverables','五分钟汇报讲稿.md'),md.join('\n'));
console.log(JSON.stringify({pptx:path.join(root,'deliverables','汇报草稿.pptx'),slides:8,validation:path.join(buildDir,'validation.json')}));
