# Input dataset and calculate word weights
# -*- coding: utf-8 -*-
# calculate cos sim between label and each word
# followed by 2SynRepSentiScore
import numpy as np
import os, sys
import csv
import nltk
from afinn import Afinn
af = Afinn()

# inputs #
loadpath = "./SA0726/SAtrain5.csv"
############
# outputs #
outputpath = 'SSTweights250'
######################
maxLength = 305
#################
#sentiwordNet#######
sentiwordnet = np.load('sentiwordnet.npy',allow_pickle=True).item()
########################

## senticnet
from senticnet.senticnet import SenticNet  # 不全面，中性词没有分值
sn = SenticNet()
# polarity_label = sn.polarity_label(ww)
# polarity_value = sn.polarity_value(ww)

csvFile = open(loadpath, "r")
reader = csv.reader(csvFile)
text = []
label = []
for item in reader:
    # print(item)
    sen = item[0].lower()
    sen = nltk.word_tokenize(sen)
    sen = ' '.join(sen)
    text.append(sen)
    label.append(item[1])

print('text:',len(text),text)
print('label:',len(label),label)


total = len(text)
cs = [] # all cosine similarity
stopwords = [] # create its own stop words
print('total sentences:',total)
for i in range(total):
    sen = text[i] # current sentence
    sen_split = sen.split()
    temp = []
    for token in sen_split:
        try:
            # afinn lexicon ##
            weight = abs(af.score(token))
            ## sentiwordnet#####
            # weight = abs(sentiwordnet[token]) * 4
            #
            # ## sentinet
            # weight = abs(float(sn.polarity_value(token))) *4

        except:
            weight = 0.0
        temp.append(weight)
    temp = np.array(temp)
    cs.append(temp)
print(cs)
cs = np.array(cs)
print('cs',cs.shape)
np.save(outputpath, cs)
print('load path:',loadpath)

