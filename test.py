# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 00:21:55 2018

@author: dhanu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:48:16 2018

@author: dhanu
"""

from __future__ import unicode_literals, print_function, division
import time
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.time_utils import timeSince
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

vectors = bcolz.open(f'Embedding/6B.300d.dat')[:]
words = pickle.load(open(f'Embedding/6B.300_words.pkl', 'rb'))
word2idx = pickle.load(open(f'Embedding/6B.300_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}


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


plt.switch_backend('agg')
MAX_LENGTH  = 4948
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING = True

UNK_TOKEN = 0

# =============================================================================
# LANGUAGE CLASS
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

def prepareData():
    finalData = pd.read_csv("dataset.csv")
    data = []
    Lan = Lang('English')
    print("Counting words...")
    for article, toxicity in zip(finalData['article'], finalData['toxicity']):
        Lan.addSentence(article)
        data.append(list([article,toxicity]))
    print("Counted Words:")
    print(Lan.name, Lan.n_words)
    return data, Lan



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



def train(input_tensor, target_class, encoder, fcn, encoder_optimizer, fcn_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    fcn_optimizer.zero_grad()

    input_length = input_tensor.size(0)
#    target_length = target_tensor.size(0)sss

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]


    fcn_output = fcn(encoder_hidden).view(-1)
    target_class = target_class.view(-1)
    loss = criterion(fcn_output, torch.tensor(target_class, dtype=torch.float32, device = device))
    loss.backward()
    encoder_optimizer.step()
    fcn_optimizer.step()

    return loss.item()


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(Lan, pair):
    input_tensor = tensorFromSentence(Lan, pair[0])
    target_class = torch.tensor(pair[1], dtype = torch.long, device=device).view(-1,1)
    return (input_tensor, target_class)


def trainIters(encoder, fcn ,Lan, data,print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
#    data , Lan = prepareData()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001, betas=(0.9, 0.999))
    fcn_optimizer = optim.Adam(fcn.parameters(), lr=0.001, betas=(0.9, 0.999))
#    training_pairs = [tensorsFromPair(Lan, random.choice(data))
#                      for i in range(n_iters)]
    criterion = nn.BCEWithLogitsLoss()
    n_iters = len(data)
    epoch = 0
    iter = 1
    while iter<=30001:
        training_pair = tensorsFromPair(Lan, data[(iter-1)%len(data)])
        input_tensor = training_pair[0]
        target_class = training_pair[1]

        loss = train(input_tensor, target_class, encoder,
                     fcn, encoder_optimizer, fcn_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            torch.save(encoder, 'encoder.network')
            torch.save(fcn, 'fcn.network')
            print('iter = %d , epoch = %d' % (iter, epoch))
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        iter += 1
        if((iter-1)%len(data) == 0):
            epoch += 1
            showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

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
    error = 0
    for iter in range(len(test)):
        input_sentence = test[iter][0]
        target = test[iter][1]
        output = evaluate(encoder1, fnn1, input_sentence, English)
        if F.sigmoid(output)[0].item()  >= 0.7:
            output = 1
        else:
            output = 0
        error += (output - target) ** 2
        
        error_rms = np.sqrt(error/len(data))
        print(error_rms)
        accuracy = (1- error_rms)*100
        print('Accuracy = %.4f%%' %accuracy)
        
        
if __name__ == "__main__":
    main()