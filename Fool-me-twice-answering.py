dir_qa='../fact-check-summarization/gen-qas/'
dir_claim='../fool-me-twice/dataset/'

import pandas as pd
import json
import re

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

claim_train = dir_claim +'train.jsonl'
claim_dev = dir_claim +'dev.jsonl'
claim_test = dir_claim +'test.jsonl'

# print(claim_train)
df_train = pd.read_json(claim_train, lines=True)
df_dev = pd.read_json(claim_dev, lines=True)
df_test = pd.read_json(claim_test, lines=True)

d_frames = [df_train, df_dev, df_test]
df_claims = pd.concat(d_frames, ignore_index = True)

def pre_process(row):
    
# Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', row['text'])
    return text

df_claims['text'] = df_claims.apply(pre_process, axis=1)

def evi_concat(row):
    text = ' '.join([e['text'] for e in row['gold_evidence']])
    text = re.sub(r'\s+', ' ', text)
    return text

df_claims['gold_evi_concat'] = df_claims.apply(evi_concat, axis=1)


qas_file = dir_qa+'train.target.hypo.beam60.qas_filtered'
df_qa = pd.read_json(qas_file, lines=True)

quals_score = pd.read_json(dir_qa + 'train.source.source_eval_noprepend.quals', lines=True)

df_claims['quals_score'] = quals_score['eval_ns-ns'].tolist()

df_qa = df_qa[0]

df_claims['qas'] = df_qa


from farm.infer import Inferencer

# nlp = Inferencer.load('deepset/bert-large-uncased-whole-word-masking-squad2', task_type='question_answering', batch_size=2048, gpu=True)
# nlp = Inferencer.load('deepset/roberta-base-squad2', task_type='question_answering', batch_size=2048, gpu=True)
nlp = Inferencer.load('deepset/electra-base-squad2', task_type='question_answering', batch_size=2048, gpu=True)


QA_input_gold = []
for index, row in df_claims.iterrows():
    q = {}
    q['questions'] = []
    for qa in row['qas']['qas']:
        q['questions'].append(qa['q'])
    if len(row['gold_evi_concat']) > 10:
        q['text'] = row['gold_evi_concat']
    else: 
        q['text'] = ' '.join(row['sentences'])
    QA_input_gold.append(q)

answer_gold = nlp.inference_from_dicts(dicts=QA_input_gold)

with open('fm2-gold-answer-electra.json', 'w') as f:
    json.dump(answer_gold, f)
