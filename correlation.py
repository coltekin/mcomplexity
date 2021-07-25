#!/usr/bin/env python3
"""Plot the table with tikz with some optional settings.
   (some stuff is easier to do in a real programming language).
"""

import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument('input')
ap.add_argument('-n', '--normalize', action='store_true')
ap.add_argument('-m', '--measures', default='all',
                            help='comma separated measures, or all')
ap.add_argument('-T', '--tbinfo', default='ud_sample.tsv')
ap.add_argument('-o', '--output', default='test-tikz.tex')
args = ap.parse_args()



tbinfo = dict()
with open(args.tbinfo, 'rt') as fp:
    _ = next(fp)
    for line in fp:
        id_, lang, tb, size, pec, anc, nomor, fam = line.split('\t')
        lang = lang.replace(' ', '_')
        tbinfo['-'.join((lang,tb))] = {
            'id': id_,
            'iso': id_.split('_')[0],
            'lang': lang,
            'size': size, 
            'peculiar': bool(len(pec.strip())),
            'nomor': bool(len(nomor.strip())),
            'ancinet': bool(len(anc.strip())),
            'fam': 'IE' if not len(fam.strip()) else fam.strip().replace('-','')
        }

scores = dict()
with open(args.input) as fp:
    head = next(fp).strip().split('\t')
    for line in fp:
        row = line.strip().split('\t')
        tb = row[0]
        scores[tb] = dict()
        for i,val in enumerate(row[1:]):
            scores[tb][head[i+1]] = val

exclude = {'Chinese-GSD', 'Japanese-GSD', 'Korean-GSD', 'Korean-Kaist', 'Latin-ITTB'}
tblist = [x for x in scores.keys() if x not in exclude]
mlist = [s for s in scores[tblist[0]].keys() if '_' not in s]
measures = dict()
for m in mlist:
    measures[m] = np.array([float(scores[tb][m]) for tb in tblist])
    sd = m + "_sd"
    measures[sd] = np.array([float(scores[tb][sd]) for tb in tblist])

corr = np.ones(shape=(len(mlist),len(mlist)))
corrp = np.zeros(shape=(len(mlist),len(mlist)))
for i, m1 in enumerate(mlist):
    for j, m2 in enumerate(mlist[:i+1]):
        if m1 == m2: continue
        corr[i,j], corrp[i,j] = pearsonr(measures[m1], measures[m2])
        corr[j,i], corrp[j,i] = spearmanr(measures[m1], measures[m2])

print(r"\setlength{\tabcolsep}{1pt}")
print(r"\renewcommand{\arraystretch}{0}")
print(r"\begin{tabular}{l*{" + str(len(mlist)) +
        r"}{r}}")
print((len(mlist)*"& {{{}}} ").format(*mlist), r"\\")
print(r"\midrule")
for i, m1 in enumerate(mlist):
    print("{}".format(m1))
    for j, m2 in enumerate(mlist):
        if corr[i,j] < 0:
            col = "red"
        else:
            col = "blue"
        if m1 == m2:
            print(r"& \phantom{|}", end="")
        else:
            if corrp[i,j] > 0.05:
                mark = r"^{\text{\tiny*}}"
            else:
                mark = r"^{\text{\tiny\phantom{*}}}"
            print("& \\tikz{{\\node[align=right,text width=9mm,"
              "minimum height=4ex,anchor=base,fill={}!{}]"
              " {{${:0.2f}{}$}};}}".format(col, 75*abs(corr[i,j]),
              corr[i,j], mark), end="")
    print(r"\\")
print(r"\end{tabular}")
print(r"\setlength{\tabcolsep}{6pt}")
print(r"\renewcommand{\arraystretch}{1}")
