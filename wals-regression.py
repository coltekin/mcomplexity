#!/usr/bin/env python3 

import argparse
import csv
import numpy as np
from sklearn.linear_model import RidgeCV

ap = argparse.ArgumentParser()
ap.add_argument('--scores', '-s', default="complexity-scores.txt",
        help='Tab-separated file containing scores for each treebank.')
ap.add_argument('--wals-features', '-w',
        default=("22A,26A,27A,28A,29A,30A,33A,34A,37A,38A,49A,51A,"
                "57A,59A,65A,66A,67A,69A,70A,73A,74A,75A,78A,94A"
                "101A,102A,111A,112A"),
        help='Comma separated list of WALS features')
ap.add_argument('--wals-db', '-W', default="wals-values.csv",
        help='CSV file containing WALS values')
args = ap.parse_args()

feats = set(args.wals_features.split(','))

lang, measures = [], []
with open(args.scores,'rt') as f:
    csvr = csv.DictReader(f, delimiter="\t")
    for row in csvr:
        lcode = row['WALS_code']
        lang.append(lcode)
        m = dict()
        for col, val in row.items():
            if col in {'treebank', 'WALS_code'}: continue
            if col.endswith('_sd'): continue
            m[col] = float(val)
        measures.append(m)

wals = {k:dict() for k in set(lang)}
fvset = set()
with open(args.wals_db, 'rt') as f:
    csvr = csv.DictReader(f)
    for i, row in enumerate(csvr):
        f, l = row['Parameter_ID'], row['Language_ID']
        if f in feats and l in wals:
            v = row['Value']
            wals[l][f] = v
            fvset.add("-".join((f,v)))

fvset = sorted(fvset)
x = np.zeros((len(lang), len(fvset)))
for i, l in enumerate(lang):
    for j, fv in enumerate(fvset):
        f, v = fv.split('-')
        if wals[l].get(f, None) == v:
            x[i,j] = 1
        elif f in wals[l]:
            x[i,j] = -1
        else:
            x[i,j] = 0

mset = sorted(set(measures[0]))
#for m in mset:
print("measure\terror")
for m in "ttr msp ws wh lh is mfh -ia".split(): # for consisten ordering
    y = np.array([measures[i][m] for i in range(len(lang))])
    y = (y - y.mean()) / y.std()
    errors = []
    for i in range(x.shape[0]):
        trn_x = np.vstack((x[:i], x[i+1:]))
        trn_y = np.hstack((y[:i], y[i+1:]))
        model = RidgeCV(normalize=True).fit(trn_x, trn_y)

        tst_x = x[i,np.newaxis]
        gld_x = y[i]
        y_pred = model.predict(tst_x)[0]
        errors.append(abs(y_pred - gld_x))
    print("{}\t{}".format(m, np.mean(errors)))
