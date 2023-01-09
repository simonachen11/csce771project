import numpy as np
from rouge import Rouge
import os, sys
import jieba
jieba.initialize()

sys.setrecursionlimit(1000000)

rouge = Rouge()

metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

def compute_rouge(source, target, unit='char'):

    if unit == 'word': 
        source = jieba.cut(source, HMM=False)
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    print(source)
    print(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }

def compute_metrics(source, target, unit='char'):
   
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics

with open("summary_result2.txt") as f:
    content = " ".join([l.rstrip() for l in f])
#print(content)
source = content
with open("person_summary_result2.txt") as f:
    tmp = " ".join([l.rstrip() for l in f])
target = tmp
metrics = compute_metrics(source,target)
print("score=", metrics['main'])
metrics = compute_metrics(source,target,'word')
print("score=", metrics['main'])