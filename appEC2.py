import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import torch 
import torch.nn as nn
import torch.nn.functional as F
import re
from nltk.util import ngrams
import string
import random
from collections import Counter
import os


app = Flask(__name__)

class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,lstm_size,batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

seq_size=20
batch_size=64
embedding_size=300
lstm_size=64
ngrams=5
n_vocab = 81977

model = RNNModule(n_vocab, seq_size, embedding_size, lstm_size)
model.load_state_dict(torch.load('model_dict', map_location=torch.device('cpu')))

def generateNGrams(sequence, n):
    ngrams = []
    for i in range(len(sequence)-n+1):
        ngrams.append(' '.join(tuple(sequence[i:i+n])))
    return ngrams

with open(r"vocab_to_int.pickle", "rb") as input_file1:
    vocab_to_int_tri = pickle.load(input_file1)

with open(r"int_to_vocab.pickle", "rb") as input_file2:
    int_to_vocab_tri = pickle.load(input_file2)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate',methods=['POST'])
def generate():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    # output it gets ['and mrs dursley woke up']
    query = int_features[0]
    
    gram_gen = generateNGrams(query.split(),5)

    model.eval()

    sh,sc = model.zero_state(1)
    for w in gram_gen:
        ix = torch.tensor([[vocab_to_int_tri[w]]]).long()
        output, (state_h, state_c) = model(ix, (sh, sc))
        
        
    _, top_ix = torch.topk(output[0], k=10)
    
    choices = top_ix.tolist()

    choice = np.random.choice(choices[0])
    gram_gen.append(int_to_vocab_tri[choice])
    
    for _ in range(20):
        ix = torch.tensor([[choice]]).long()
        output, (state_h, state_c) = model(ix, (state_h, state_c))
        
        _, top_ix = torch.topk(output[0], k=10)
        choices = top_ix.tolist()
        
        choice = np.random.choice(choices[0])
        
        gram_gen.append(int_to_vocab_tri[choice])
       
    
    sentence = ' '.join(gram_gen)

    sentence_10 = sentence.split()[0:200]

    return render_template('home.html', prediction_text=' '.join(sentence_10))


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port= 8080)
