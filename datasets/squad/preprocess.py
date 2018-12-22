#!/usr/bin/python3
import json
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
import numpy as np

data=pd.read_csv("./data/SQuAD-v1.1.csv",encoding = "ISO-8859-1")
data['cid']=data.index
context_dict=dict()
contexts=pd.DataFrame()

index=0
qa_table=pd.DataFrame()

def get_qa(row):
    global index
    qa_set=row['QuestionAnswerSets'][1:-1]
    for i in qa_set.split("<|"):
        sets=i.split("|>")[0]
        sets=sets.split(',')
        try:
            ans=sets[1].split('->')[1]
            que=sets[0].split('->')[1]
            qa_table.loc[index,"question"]=que.replace('"','').strip().lower()
            qa_table.loc[index,"answer"]=ans[2:-2].replace('"','').strip().lower()
            qa_table.loc[index,"contextid"]=row['cid']
            index+=1
        except:
            continue


qa_table=qa_table.dropna()
context=pd.DataFrame({'id':data.index,'context':data['Context'].tolist()})

def data_prep(row):
    para=context[context.id==row['contextid']]['context'].tolist()[0]
    for ind,sent in enumerate(TextBlob(para).sentences):
        if str(row['answer']) in sent.string.lower():
            row['target']=ind
            row['start']=sent.string.lower().find(row['answer'])
            row['end']=row['start']+len(row['answer'])
            return row


tqdm.pandas()
data.progress_apply(get_qa,axis=1)
qa_table['start']=np.nan
qa_table['end']=np.nan
qa_table['target']=np.nan
qa_table=qa_table.progress_apply(data_prep,axis=1)
qa_table.to_csv("./data/qa_table.csv",index=False)
context.to_csv('./data/context.csv',index=False)
print(qa_table.head(15))
