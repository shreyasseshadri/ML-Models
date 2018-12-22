#!/usr/bin/python3
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from sklearn.metrics.pairwise import cosine_similarity

qa_table=pd.read_csv("./data/qa_table.csv").dropna()
context=pd.read_csv("./data/context.csv").dropna()

if not os.path.exists('./data/d2v.model'):
    sentance_corpus=[]
    count=0
    def make_corpus(row):
        global count
        for sent in TextBlob(row.replace("'",'').replace('"','')).sentences:
            sentance_corpus.append(TaggedDocument(words=sent.words.lower(),tags=str(count)))
            count+=1

    tqdm.pandas()
    context['context'].progress_apply(make_corpus)
    qa_table['question'].progress_apply(make_corpus)

    max_epochs = 50
    vec_size = 100
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)
    
    model.build_vocab(sentance_corpus)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(sentance_corpus,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("./data/d2v.model")
    print("Model Saved")
else:
    model=Doc2Vec.load('./data/d2v.model')
    
    pred=[]
    def predict(row):
        para=context[context.id==row['contextid']]['context'].tolist()[0].replace("'",'').replace('"','')
        question=TextBlob(row['question']).words
        max=-2
        maxi=-1
        for ind,sent in enumerate(TextBlob(para).sentences):
            sim_count=len(set(question).intersection(sent.words.lower()))
            if sim_count>max:
                max=sim_count
                maxi=ind
        pred.append(maxi)

    # def predict(row):
    #     para=context[context.id==row['contextid']]['context'].tolist()[0].replace("'",'').replace('"','')
    #     question_vec=np.array([])
    #     for word in TextBlob(row['question']).words:
    #         question_vec=np.concatenate([question_vec,model.wv.word_vec(word)])

    #     max=-2
    #     maxi=-1
    #     for ind,sent in enumerate(TextBlob(para).sentences):
    #         sentance=sent.words.lower()
    #         sentance_vec=np.array([])
    #         for word in sentance:
    #             sentance_vec=np.concatenate([sentance_vec,model.wv.word_vec(word)])
    #         sim=cosine_similarity(np.array([sentance_vec]),np.array([question_vec]))
    #         print(sim)
    #         if sim[0][0]>max:
    #             max=sim
    #             maxi=ind
    #     pred.append(maxi)
        
    tqdm.pandas()
    qa_table.progress_apply(predict,axis=1)
    target=np.array(qa_table['target'].tolist())
    pred=np.array(pred)
    print(np.mean(pred==target))

