# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:39:21 2018

@author: dhanu
"""

import pandas as pd
import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

def main():
    data = pd.read_csv('train.csv')
    data['inappropriate'] = 0
    
    
    for index in data.index:
        data.loc[index,'inappropriate'] = data.loc[index,'toxic'] | data.loc[index,'severe_toxic'] | data.loc[index,'obscene'] | data.loc[index,'threat'] | data.loc[index,'insult'] | data.loc[index,'identity_hate']
    
    del data['id']
    del data['toxic']
    del data['severe_toxic']
    del data['obscene']
    del data['threat']
    del data['insult']
    del data['identity_hate']
    data['comment_text'] = data['comment_text'].map(lambda com : clean_text(com))
    data.to_csv('data.csv', index = False)
    
if __name__ == "__main__":
    main()