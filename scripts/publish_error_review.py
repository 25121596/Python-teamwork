"""Export reviewed semantic paraphrases, leaving locked evaluation files intact.

The private notes explicitly identify AI assistance. This script never promotes
that work to human confirmation and never publishes mail IDs or original text.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from spam_classifier.dataset import file_sha256
from spam_classifier.experiment_io import read_json, verify_lock, write_json


def publish(root: Path):
    verify_lock(root)
    final=read_json(root/'artifacts/metrics/final_evaluation.json')
    audit=read_json(root/'artifacts/audit/final_evaluation_validation.json')
    if final.get('passed') is not True or audit.get('passed') is not True:
        raise ValueError('Final evaluation must pass first.')
    if audit.get('final_evaluation_sha256') != file_sha256(root/'artifacts/metrics/final_evaluation.json'):
        raise ValueError('Final audit does not match the evaluation.')
    for path,sha in final['output_sha256'].items():
        if file_sha256(root/path)!=sha: raise ValueError('Locked prediction evidence changed.')
    original_path=root/'artifacts/processed/stage08/error_review.json'
    notes_path=root/'artifacts/processed/stage08/assisted_notes.json'
    original=read_json(original_path); notes=read_json(notes_path)
    if (notes['source_review_sha256']!=file_sha256(original_path)
        or original['lock_sha256']!=final['lock_sha256']
        or notes['lock_sha256']!=final['lock_sha256']
        or notes['reviewer_kind']!='ai_assisted'):
        raise ValueError('Notes do not match the private review source.')
    expected=[]
    with np.load(root/'artifacts/processed/stage08/test_predictions.npz',allow_pickle=False) as arr:
        ids={v:i for i,v in enumerate(arr['mail_ids'].tolist())}
        names=arr['model_names'].tolist()
        for model in ('nb','lr'):
            for kind in ('FP','FN'):
                for number,case in enumerate(original['cases'][model][kind],1):
                    i=ids[case['mail_id']]; true=int(arr['y_true'][i]); pred=int(arr['predictions'][i,names.index(model)])
                    if ((true,pred)!=(case['true_label'],case['prediction'])
                        or (true,pred)!=({'FP':(0,1),'FN':(1,0)}[kind])):
                        raise ValueError('Case differs from locked predictions.')
                    expected.append((f'{model.upper()}-{kind}-{number:02}',model,kind))
    if [x['case_id'] for x in notes['cases']]!=[x[0] for x in expected]:
        raise ValueError('Each selected case must be reviewed exactly once in order.')
    cases=[]
    for note,(identity,model,kind) in zip(notes['cases'],expected):
        text=note['safe_paraphrase']; code=note['reason_code']
        if (note.get('assisted_reviewed') is not True or code not in notes['reason_labels']
            or not isinstance(text,str) or not 8<=len(text)<=180
            or re.search(r'https?://|www\.|[\w.+-]+@[\w.-]+|\d{7,}',text,re.I)):
            raise ValueError('Missing review or unsafe public paraphrase.')
        if note.get('reading_scope') not in ('subject_and_body','subject_and_selected_beginning_middle_end'):
            raise ValueError('Reading scope must be explicit.')
        cases.append({'case_id':identity,'model':model,'error_type':kind,'reason_code':code,'safe_paraphrase':text,'reading_scope':note['reading_scope']})
    chosen=notes['report_example_ids']
    if len(chosen)!=len(set(chosen)) or not set(chosen)<={x['case_id'] for x in cases}:
        raise ValueError('Report examples must refer to reviewed cases.')
    by_id={x['case_id']:x for x in cases}
    result={'schema_version':1,'lock_sha256':final['lock_sha256'],
        'ai_assisted_review_status':'complete','human_confirmation_status':'pending',
        'reviewer_kind':'AI-assisted semantic reading; not student sign-off',
        'selection':original['selection'],'case_count':len(cases),
        'sample_caveat':'Highest absolute model scores, not a random sample; category counts cannot estimate all-error prevalence. Observations are hypotheses, not proven causes or relabeling.',
        'source_review_sha256':file_sha256(original_path),'private_notes_sha256':file_sha256(notes_path),
        'reason_labels':notes['reason_labels'],
        'counts':{model:{kind:dict(Counter(x['reason_code'] for x in cases if x['model']==model and x['error_type']==kind)) for kind in ('FP','FN')} for model in ('nb','lr')},
        'cases':cases,'report_examples':[by_id[x] for x in chosen],
        'interpretation':notes['interpretation'], 'reading_scope':notes['reading_scope'], 'slide_examples':notes['slide_examples']}
    for item in result['slide_examples']:
        if item['case_id'] not in by_id or not 8<=len(item['short_observation'])<=55:
            raise ValueError('Invalid slide excerpt.')
        if re.search(r'https?://|www\.|[\w.+-]+@[\w.-]+|\d{7,}',item['short_observation'],re.I):
            raise ValueError('Unsafe slide excerpt.')
    write_json(root/'artifacts/metrics/error_review_assisted.json',result)
    lines=['# 真实错分语义复核','',f'共 {len(cases)} 个模型错误实例；AI 辅助语义复核完成，学生人工确认待完成。','',result['reading_scope'],'','候选按模型分数绝对值选取，并非随机样本。类别计数不能用于估计全部错分的比例；以下观察不是已证实的错误原因，也不用于修改原标签。','','| 实例 | 观察类别 | 安全语义概述 |','|---|---|---|']
    lines.extend(f'| {x["case_id"]} | {notes["reason_labels"][x["reason_code"]]} | {x["safe_paraphrase"]} |' for x in cases)
    lines.extend(['',notes['interpretation'],'','原始邮件、mail_id 对应关系和分数只在私有缓存中。公开概述不复制邮件原句或联系信息。原始语料标签保持不变，标签疑点只记为待复核。',''])
    (root/'docs/真实错分分析.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'published_cases':len(cases),'human_confirmation_status':'pending'},ensure_ascii=False))


if __name__=='__main__': publish(ROOT)
