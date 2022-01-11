import pandas as pd
import numpy as np
import json
import re

dir_qa='../fact-check-summarization/gen-qas/'
dir_claim='../fool-me-twice/dataset/'

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

df_answer_gold = pd.read_json('fm2-gold-answer-electra.json')

def relabel(row):
    if row['label'] == "SUPPORTS" :
        return 1
    if row['label'] == 'REFUTES' :
        return 0
    
df_claims['label'] = df_claims.apply(relabel, axis=1)

docidsx = []
qids = []
ids = []
claims = []
evidences = []
questions = []
answers = []
answers_gold = []
labels = []
g = 0
for index, row in df_claims.iterrows():
    
    for qa in row['qas']['qas']:
        docidsx.append(index)
        ids.append(row['id'])
        claims.append(row['text'])
        evidences.append(row['gold_evi_concat'])
        questions.append(qa['q'])
        answers.append(qa['a'])
        qids.append(df_answer_gold['predictions'][g][0]['id'].split('-')[1])
        answers_gold.append(df_answer_gold['predictions'][g][0]['answers'][0]['answer'])
        labels.append(row['label'])
        g+=1
        
dict_prepare = {

    'docid':docidsx,
    'qid':qids,
    'id':ids,
    'text':claims,
    'gold_evi_concat':evidences,
    'question':questions,
    'answer_text':answers,
    'answer_gold':answers_gold,
    'label':labels,

}

df_input = pd.DataFrame(dict_prepare, columns = ['docid','qid', 'id', 'text', 'gold_evi_concat', 'question','answer_text', 'answer_gold', 'label'])

grouped_sort = df_input.groupby('docid')

df_aggre_again = grouped_sort.aggregate(lambda x: tuple(x))

cid = []
docid = []
claim = []
gold_evidence = []
q_concat = []
ac_concat = []
ag_concat = []
label = []
for index,row in df_aggre_again.iterrows():
    docid.append(index)
    cid.append(row['id'][0])
    claim.append(row['text'][0])
    gold_evidence.append(row['gold_evi_concat'][0])
    q_concat.append(' '.join(row['question']))
    ac_concat.append('\n'.join(row['answer_text']))
    ag_concat.append('\n'.join(row['answer_gold']))
    label.append(row['label'][0])
            
dict_final = {

    'docid':docid,
    'id':cid,
    'text':claim,
    'gold_evi_concat':gold_evidence,
    'question':q_concat,
    'answer_text':ac_concat,
    'answer_gold':ag_concat,
    'label':label,

}

df_input_cat = pd.DataFrame(dict_final, columns = ['docid','id','text','question', 'gold_evi_concat',
                                                     'answer_text','answer_gold',
                                                     'label'])

df_train_set = df_input_cat[df_input_cat['id'].isin(set(df_train['id']))]
df_dev_set = df_input_cat[df_input_cat['id'].isin(set(df_dev['id']))]
df_test_set = df_input_cat[df_input_cat['id'].isin(set(df_test['id']))]

df_train_set.to_pickle('qa-fool-me-twice-train-oria-electra.pkl')
df_dev_set.to_pickle('qa-fool-me-twice-dev-oria-electra.pkl')
df_test_set.to_pickle('qa-fool-me-twice-test-oria-electra.pkl')