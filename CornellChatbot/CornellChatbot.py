import pandas as pd #pandas datareader
import numpy as np
import tensorflow as tf
import re
import time
tf.__version__

#loads cornell data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

#sentences to train
lines[:10]
conv_lines[:10]

#dict to map line id with text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
#list of converstions line ids
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
convs[:10]

#sort into questions(input) or answers (target/output)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

#debugtest
limit = 0
for i in range(limit, limit+5):
    print(questions[i])
    print(answers[i])
    print()

#compare question/answer langth
print(f"Question Length:",len(questions))
print(f"Answer Length:",len(answers))
