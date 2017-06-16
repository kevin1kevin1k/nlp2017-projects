from bs4 import BeautifulSoup as BS
import numpy as np
import arrow
import os

submissions = []
with open('../Submissions - NTU NLP 2017 - Term Project 2 _ Kaggle in Class.htm') as f:
    html = f.read()
    soup = BS(html, 'lxml')
    for i, tr in enumerate(reversed(soup('tr'))):
        div = tr('div')
        center = tr.find_all(class_='center')
        file_ = tr.find_all(class_='file')
        if div and center and file_:
            desc = div[0].text.strip()
            acc = float(center[0].text)
            filename = file_[0].text
            submissions.append( (desc, acc, filename) )

submissions.sort(key=lambda x: x[ACC], reverse=True)

int2relation = ['Temporal', 'Contingency', 'Comparison', 'Expansion']
relation2int = {r:i for i, r in enumerate(int2relation)}
DESC, ACC, FILENAME = range(3)
PREDICT_PATH = '../predictions'
TEST_N = 1000
TOP_N = 23
BLENDING = 'uniform'
# 'uniform' for equal vote
# 'linear' for voting weighted by Kaggle accuracy

Y = np.zeros([TEST_N, 4])
ids = [''] * TEST_N
for sub in submissions[:TOP_N]:
    filepath = os.path.join(PREDICT_PATH, sub[FILENAME])
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            
            id_, relation = line.strip().split(',')
            ids[i - 1] = id_
            if BLENDING == 'uniform':
                Y[i - 1, relation2int[relation]] += 1
            elif BLENDING == 'linear':
                Y[i - 1, relation2int[relation]] += sub[ACC]

y = Y.argmax(axis=1)
time = arrow.now('Asia/Taipei').format('YYYYMMDD_HH:mm:ss')
filepath = os.path.join(PREDICT_PATH, 'ensemble_%s_top%d_%s.csv' % (BLENDING, TOP_N, time))
with open(filepath, 'w') as f:
    f.write('Id,Relation\n')
    for id_, r in zip(ids, y):
        f.write('%s,%s\n' % (id_, int2relation[r]))
