# -*- coding: utf-8 -*-
"""

MLD Final project - 5 sentences generation 5-gram models

Babandeep Singh and Blake Weston


"""

import numpy as np
import re
from nltk.util import ngrams
import string
import random

corpus = open("Harry Potter and Sorcer's Stone.txt", "r").read().lower()

clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', corpus)
clean_text = re.sub(' +', ' ', clean_text)
clean_text = re.sub(' \n+',' ',clean_text)

clean_text[0:60]

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os
import time



def generateNGrams(sequence, n):
    ngrams = []
    for i in range(len(sequence)-n+1):
        ngrams.append(' '.join(tuple(sequence[i:i+n])))
    #print(str(n)+"-Grams Generated\n")
    #print(ngrams[:5])
    return ngrams

def prep_data(corpus_text):
  word_counts = Counter(corpus_text)
  sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

# Converting int to words 
  int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}

# words to int 
  vocab_to_int = {w: k for k, w in int_to_vocab.items()}
  n_vocab = len(int_to_vocab)

  print('Vocabulary size', n_vocab)

  int_text = [vocab_to_int[w] for w in corpus_text]
  num_batches = int(len(int_text) / (seq_size * batch_size))

  in_text = int_text[:num_batches * batch_size * seq_size]

  out_text = np.zeros_like(in_text)
  out_text[:-1] = in_text[1:]
  out_text[-1] = in_text[0]
  in_text = np.reshape(in_text, (batch_size, -1))
  out_text = np.reshape(out_text, (batch_size, -1))
  return in_text,out_text,n_vocab,int_to_vocab,vocab_to_int

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

def train(model, iterator, optimizer, criterion):
  loss_values= []
  batches = get_batches(in_text, out_text, batch_size, seq_size)
  state_h, state_c = model.zero_state(batch_size)

  # Transfer data to GPU
  state_h = state_h.to(device)
  state_c = state_c.to(device)
  for x, y in batches:
    #iteration += 1
    # Tell it we are in training mode
    model.train()
    # Reset all gradients
    optimizer.zero_grad()
    # Transfer data to GPU
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    logits, (state_h, state_c) = model(x, (state_h, state_c))
    loss = criterion(logits.transpose(1, 2), y)
    loss_value = loss.item()
    loss.backward()
    state_h = state_h.detach()
    state_c = state_c.detach()
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(),5)
    optimizer.step()
    loss_values.append(loss_value)
  return np.mean(loss_values)

seq_size=20
batch_size=64
embedding_size=300
lstm_size=64
n_epochs = 500
checkpoint_path='checkpoint'
ngrams=5
learning_rate = 1

corpus_text = generateNGrams(clean_text.split(),ngrams)
in_text,out_text,n_vocab, int_to_vocab_tri,vocab_to_int_tri= prep_data(corpus_text) # changed here 
batches = get_batches(in_text, out_text, batch_size, seq_size)

# vocab_to_int_tri
# int to vocab tri
import pickle

with open('vocab_to_int.pickle', 'wb') as handle:
    pickle.dump(vocab_to_int_tri, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('int_to_vocab.pickle', 'wb') as handle:
    pickle.dump(int_to_vocab_tri, handle, protocol=pickle.HIGHEST_PROTOCOL)

device = 'cpu'

"""Only when training is needed. So far trained on 500 epochs with 0.89 loss

"""

# model_trigram = RNNModule(n_vocab, seq_size,embedding_size, lstm_size)
# model_trigram = model_trigram.to(device)
# optimizer = torch.optim.SGD(model_trigram.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
# sentences = []
# for e in range(n_epochs):
#   start_time = time.time() 
#   loss_values = train(model_trigram, batches, optimizer, criterion)
#   print('\nEpoch: {}/{}'.format(e, n_epochs),
#         'Loss: {}'.format(np.mean(loss_values)))
#   time_taken = round(time.time() - start_time,2)
#   print("Time Taken to run epoch--- %s seconds ---" % (round(time_taken,2)))

# print(str(ngrams)+'-LSTM model')

# # saving the model 
# # torch.save(model_trigram,'/content/drive/Shareddrives/MLD OPS /')

# # torch.save(model_trigram,'RNN_model')

# model.state_dict()

# torch.save(model.state_dict(), '/content/drive/Shareddrives/MLD OPS /model_dict')



"""# LOading the model """

model = RNNModule(n_vocab, seq_size, embedding_size, lstm_size)
model.load_state_dict(torch.load('/content/drive/Shareddrives/MLD OPS /model_dict'))

# generating tri grams 

ix = random.randint(0,500)
query = corpus_text[ix]
# query = 'noticed the first sign of'

# model = torch.load('/content/drive/Shareddrives/MLD OPS /RNN_model')
model = model.to(device)
 
gram = generateNGrams(query.split(),ngrams)
model.eval()
sh,sc = model.zero_state(1)
sh = sh.to(device)
sc = sc.to(device)
for w in gram:
  ix = torch.tensor([[vocab_to_int_tri[w]]]).to(device)
  output, (state_h, state_c) = model(ix, (sh, sc))
  
_, top_ix = torch.topk(output[0], k=20)

choices = top_ix.tolist()

choice = np.random.choice(choices[0])
gram.append(int_to_vocab_tri[choice])

for _ in range(6):
  ix = torch.tensor([[choice]]).to(device)
  output, (state_h, state_c) = model(ix, (state_h, state_c))
  
  _, top_ix = torch.topk(output[0], k=10)
  choices = top_ix.tolist()
  choice = np.random.choice(choices[0])

  gram.append(int_to_vocab_tri[choice])


sentence = ' '.join(gram)

sentence_20 = sentence.split()[0:100]

print(' '.join(sentence_20))









