# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:48:16 2018

@author: dhanushsr
"""

from __future__ import unicode_literals, print_function, division
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import bcolz
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from preprocess import clean_text
plt.switch_backend('agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================================
# EMBEDDING
# =============================================================================



def get_weights_matrix(target_vocab):
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
    
    return torch.from_numpy(weights_matrix)

# =============================================================================



MAX_LENGTH  = 4948

TRAINING = True

UNK_TOKEN = 0

# =============================================================================
# EnglishGUAGE CLASS
# =============================================================================

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"UNK": 0}
        self.word2count = {}
        self.index2word = {0: "UNK"}
        self.n_words = 1  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# =============================================================================

    



# =============================================================================
# CREATING EMBEDDING LAYER
# =============================================================================
def create_emb_layer(weights_matrix, trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if trainable:
        emb_layer.weight.requires_grad = True

    return emb_layer, num_embeddings, embedding_dim

# =============================================================================

#Encoder Code
class EncoderRNN(nn.Module):
    def __init__(self, weights_matrix, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True) 
        self.gru = nn.GRU(embedding_dim, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#FullyConnectedNetwork
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedNN, self).__init__()
        self.dp1 = nn.Dropout()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh1 = nn.Tanh()
        self.dp2 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.dp3 = nn.Dropout()
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input_tensor):
        input_tensor = input_tensor.view(-1)
        o1 = self.dp1(input_tensor)
        a1 = self.fc1(o1)
        h1 = self.tanh1(a1)
        o2 = self.dp2(h1)
        a2 = self.fc2(o2)
        h2 = self.tanh2(a2)
        o3 = self.dp3(h2)
        a3 = self.out(o3)

        return a3


def indexesFromSentence(lang, sentence):
    output = []
    for word in sentence.split(' '):
        try:
            output.append(lang.word2index[word])
        except:
            output.append(UNK_TOKEN)
    return output


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(Lan, pair):
    input_tensor = tensorFromSentence(Lan, pair[0])
    target_class = torch.tensor(pair[1], dtype = torch.long, device=device).view(-1,1)
    return (input_tensor, target_class)





def evaluate(encoder, fcn, sentence, Lan, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(Lan, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        output = fcn(encoder_hidden)

        return output







def main():
    global hidden_size, data, English,vectors, words, word2idx, glove
    
    hidden_size = 100    
    
    vectors = bcolz.open(f'Embedding/6B.300d.dat')[:]
    words = pickle.load(open(f'Embedding/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'Embedding/6B.300_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    
    print("Imported Embedding Data")
    
    fullData = pd.read_csv("dataset.csv")
    
    print("Imported Full Data")
    
    data = []
    
    English = Lang('English')
    print("Counting words...")
    for article, toxicity in zip(fullData['article'], fullData['toxicity']):
        English.addSentence(article)
        data.append(list([article,toxicity]))
    print("Counted Words:")
    print(English.name, English.n_words)
    
    train, test = train_test_split(data, random_state=42, test_size=0.4, shuffle=True)
    
    print("Data Split")
    encoder1 = torch.load('encoder.network')
    fnn1 = torch.load('fcn.network')
    sentence = "hello"
    sentence = clean_text(sentence)
    output = evaluate(encoder1, fnn1, sentence, English)
    output = F.sigmoid(output)
    if output >= 0.7:
        output = 1
    else:
        output = 0
    return output

if __name__ == "__main__":
    main()
