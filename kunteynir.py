import sys, random, math
from collections import Counter
import numpy as np
import re
import random
import nltk
import pandas as pd

#RUSSIAN TOXIC COMMENTS:
# ds = pd.read_csv('./dataframes/labeled.csv')
# raw_toxic = ds.comment.to_list()[:200]

#OXXXYneuron:
f = open('./oxxxy.txt')
raw_oxxxy = f.readlines()

raw = raw_oxxxy
raw = list(set(raw))
tokens = list()
for line in raw:
    for el in re.split(r'[\.\!\?]', line):
        token = (re.sub(r'[.\d\n\t\?\,\-\*]','', el).split(' '))
        while '' in token: token.remove('')
        tokens.append(token)
tokens = [el for el in tokens if len(el) > 2]
#print('simple tokens len = {}, len = {}'.format(len(tokens), tokens))

n_tokens = list()
for line in raw:
    sent = nltk.sent_tokenize(line, language='russian')
    for el in sent:
        n_token = (re.sub(r'[\d\n\t\?\,\.\)\(\«\»\"\'a-zA-Z]','', el).split(' '))
        while '' in n_token: n_token.remove('')
        n_tokens.append(n_token)
n_tokens = [el for el in n_tokens if len(el) > 2]
#print('nltk tokens len = {}, len = {}'.format(len(n_tokens), n_tokens))



vocab = set()
for sent in n_tokens:
    for word in sent:
        if len(word)>0:
            vocab.add(word)
vocab = list(vocab)
# vocab.remove('')
# print(vocab)
#print('vocab:{}, length = {}'.format(vocab[:10], len(vocab)))

word2index ={}
for i, word in enumerate(vocab):
    word2index[word]=i

#print(word2index)

def words2indicies(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=0)

np.random.seed(1)
embed_size = 10 # Размер векторного представления
#Матрица векторных преставлений
embed = (np.random.rand(len(vocab), embed_size)-0.5)*0.1
recurrent = np.eye(embed_size)#Реккурентная матрица
start = np.zeros(embed_size)#Векторное представление для пустого предложения
#Выходные веса
decoder = (np.random.rand(embed_size, len(vocab))-0.5)*0.1
one_hot = np.eye(len(vocab))#Матрица поиска выходных весов для ф. потерь

def predict(sent):
    layers = list()
    layer = {}
    layer['hidden'] = start
    layers.append(layer)
    loss = 0
    for target_i in range(len(sent)):
        layer ={}
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))

        loss-= np.log(layer['pred'][sent[target_i]])
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent)+embed[sent[target_i]]
        layers.append(layer)
    return layers, loss

import tqdm
with tqdm.trange(2000, desc= 'progress') as t:
    for it in t:
        alpha =0.002
        sent = words2indicies(n_tokens[it%len(n_tokens)][1:])
        layers, loss = predict(sent)

        for layer_idx in reversed(range(len(layers))):
            layer = layers[layer_idx]
            target = sent[layer_idx-1]
            if layer_idx>0:
                layer['output_delta'] = layer['pred']-one_hot[target]
                new_hidden_delta = layer['output_delta'].dot(decoder.T)
                if layer_idx==len(layers)-1:
                    layer['hidden_delta'] = new_hidden_delta
                else:
                    layer['hidden_delta'] = new_hidden_delta+\
                    layers[layer_idx+1]['hidden_delta'].dot(recurrent.T)
            else:
                layer['hidden_delta'] = layers[layer_idx+1]['hidden_delta'].dot(recurrent.T)
        start-=layers[0]['hidden_delta']*alpha/float(len(sent))
        for layer_idx, layer in enumerate(layers[1:]):
            #outer -  внешнее произведение двух векторов
            decoder -=np.outer(layers[layer_idx]['hidden'],
                               layer['output_delta'])*alpha/float(len(sent))
            embed_idx = sent[layer_idx]
            embed[embed_idx] -= layers[layer_idx]['hidden_delta']*alpha/float(len(sent))
            recurrent-=np.outer(layers[layer_idx]['hidden'], layer['hidden_delta'])*alpha/float(len(sent))

        if it%100==0:
        #t.set_description(str(np.exp(loss/len(sent))))
            print('Perplexity: '+str(np.exp(loss/len(sent))))
punchs= []
punch_pairs = []
for idx in np.random.choice(range(len(n_tokens)), 500):
    # print('idx = {}'.format(idx))
    l, _ = predict(words2indicies(n_tokens[idx]))
    list_indicies = []
    line_true = []
    line_pred = []
    for i, each_layer in enumerate(l[1:-1]):
        input = n_tokens[idx][i]
        true = n_tokens[idx][i + 1]
        pred = vocab[each_layer['pred'].argmax()]
        for el in line_pred*10:
            if el ==pred:
        # while pred in line_pred:
                # print('here!')
                EL = each_layer['pred'].copy()
                EL = np.delete(EL, int(EL.argmax()))
                pred = vocab[EL.argmax()]
        line_true.append(true)
        line_pred.append(pred)

    punch = line_pred[1].title()+" " +' '.join([el.lower() for el in line_pred[2:]])
    punchs.append(punch)
    # print(punch)


while punchs:
    el = punchs.pop(0)

    for rifm in punchs:
        st = random.randint(3, 4)
        split1 = el.split(' ')
        split2 = rifm.split(' ')
        if split1[-1][-st:] == split2[-1][-st:] \
                and split1[-1]!=split2[-1] \
                and el not in punch_pairs and rifm not in punch_pairs:
            punch_pairs.append(el)
            punch_pairs.append(rifm)
            break
punch_pairs = [el + ['.', ",", '!', '?'][random.randint(0, 3)]
                for el in punch_pairs]
print(*punch_pairs, sep='\n')
