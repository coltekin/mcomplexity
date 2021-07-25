#!/usr/bin/env python3
"""Plot the table with tikz with some optional settings.
   (some stuff is easier to do in a real programming language).
"""

import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


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

d = np.zeros(shape=(len(tblist),len(mlist)))
for j, m in enumerate(mlist):
    for i, tb in enumerate(tblist):
        d[i,j] = measures[m][i]

sc = MinMaxScaler()
d_mm = sc.fit_transform(d)

pca = PCA()

dt = pca.fit_transform(d)
print("% ", ["{:0.6f}".format(x) for x in pca.explained_variance_ratio_])
print("% ", sum(pca.explained_variance_ratio_[:1]))
print("% ", sum(pca.explained_variance_ratio_[:2]))
print("% ", sum(pca.explained_variance_ratio_[:3]))
print("% ", sum(pca.explained_variance_ratio_[:4]))

print(r'\begin{tikzpicture}[framed,x=5mm,y=20mm,')
print(r'  dot/.style={inner sep=0pt,minimum width=2pt,fill,circle},')
print(r'  ]')

pc1min, pc1max = dt[:,0].min(), dt[:,0].max()
pc2min, pc2max = dt[:,1].min(), dt[:,1].max()
pc1breaks = np.linspace(pc1min,pc1max,num=5)
pc2breaks = np.linspace(pc2min,pc2max,num=5)

print("% 1: {} {} 2: {} {}".format(pc1min, pc1max, pc2min, pc2max))
print("\\draw[->] ({},{}) -- ({},{});".format(
    pc1min - 0.5, pc2min - 0.1, pc1max + 0.5, pc2min - 0.1))
print("\\draw[->] ({},{}) -- ({},{});".format(
    pc1min - 0.5, pc2min - 0.1, pc1min - 0.5, pc2max + 0.1))
for i,x in enumerate(pc1breaks):
    print("\\node[font=\\tiny,anchor=north] at ({},{}) {{{:0.1f}}};".format(
        x, pc2min - 0.1, pc1breaks[i]))
    print("\\draw ({}, {}) -- ++(0,-2pt);".format(pc1breaks[i], pc2min
        - 0.1))
for i,y in enumerate(pc2breaks):
    print("\\node[font=\\tiny,anchor=east] at ({},{}) {{{:0.1f}}};".format(
        pc1min - 0.5, y, pc2breaks[i]))
    print("\\draw ({}, {}) -- ++(-2pt,0);".format(pc1min - 0.5, pc2breaks[i]))
for i,tb in enumerate(tblist):
    print("\\node[font=\\tiny,anchor=south west,inner sep=1pt]"
          "at ({},{}) {{{}}};".format(
        dt[i,0], dt[i,1], tbinfo[tb]['id'].replace('_', '.')))
    print("\\node[circle,fill,inner sep=0,minimum width=2pt] at ({},{}) {{}};"
            .format( dt[i,0], dt[i,1]))
print(r'\end{tikzpicture}')

with open('pca-dim.txt', 'wt') as fp:
    print("treebank", end="", file=fp)
    for j in range(dt.shape[1]):
        print("\tPC{}\tPC{}_sd".format(j+1, j+1), end="", file=fp)
    print(file=fp)
    for i,tb in enumerate(tblist):
        print("{}".format(tb), end="", file=fp)
        for j in range(dt.shape[1]):
            print("\t{:0.12f}\t0.0".format(dt[i,j]), end="", file=fp)
        print(file=fp)


## print(r"\setlength{\tabcolsep}{1pt}")
## print(r"\renewcommand{\arraystretch}{0}")
## print(r"\begin{tabular}{l*{" + str(len(mlist)) +
##         r"}{r}}")
## print((len(mlist)*"& {{{}}} ").format(*mlist), r"\\")
## print(r"\midrule")
## for i, m1 in enumerate(mlist):
##     print("{}".format(m1))
##     for j, m2 in enumerate(mlist):
##         if corr[i,j] < 0:
##             col = "red"
##         else:
##             col = "blue"
##         if m1 == m2:
##             print(r"& \phantom{|}", end="")
##         else:
##             if corrp[i,j] > 0.05:
##                 mark = r"^{\tiny*}"
##             else:
##                 mark = r"^{\tiny\phantom{*}}"
##             print("& \\tikz{{\\node[align=right,text width=9mm,"
##               "minimum height=4ex,anchor=base,fill={}!{}]"
##               " {{${:0.2f}{}$}};}}".format(col, 100*abs(corr[i,j]),
##               corr[i,j], mark), end="")
##     print(r"\\")
## print(r"\end{tabular}")
## print(r"\setlength{\tabcolsep}{6pt}")
## print(r"\renewcommand{\arraystretch}{1}")
