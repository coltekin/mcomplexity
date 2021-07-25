#!/usr/bin/env python3
"""Plot the table with tikz with some optional settings.
   (some stuff is easier to do in a real programming language).
"""

import argparse
import numpy as np

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

tblist = list(scores.keys())
mlist = [s for s in scores[tblist[0]].keys() if '_' not in s]
measures = dict()
for m in mlist:
    measures[m] = np.array([float(scores[tb][m]) for tb in tblist])
    sd = m + "_sd"
    measures[sd] = np.array([float(scores[tb][sd]) for tb in tblist])

# TODO: parametrize the dimensions
print(r'\begin{tikzpicture}[framed,x=6.77mm,y=2.4mm,')
print(r'  dot/.style={inner sep=0pt,minimum width=2pt,fill,circle},')
print(r'  lnode/.style={inner sep=1pt,anchor=south,font=\tiny},')
print(r'  Afroasiatic/.style={lnode,black},')
print(r'  Austroasiatic/.style={lnode,black},')
print(r'  Austronesian/.style={lnode,black},')
print(r'  Basque/.style={lnode,black},')
print(r'  Japonic/.style={lnode,black},')
print(r'  Koreanic/.style={lnode,black},')
print(r'  SinoTibetan/.style={lnode,black},')
print(r'  Turkic/.style={lnode,black},')
print(r'  Uralic/.style={lnode,black},')
print(r'  IE/.style={lnode,black},')
print(r']')
for i,m in enumerate(mlist):
    ii = i + 1
    mnorm = (measures[m] - measures[m].mean()) / measures[m].std()
    print("% {}".format((measures[m] - measures[m].mean()).tolist()))
    print("% {}".format(mnorm.tolist()))
    sdnorm = measures[m+"_sd"] / measures[m].std()
    print("% sd: {}".format(measures[m+"_sd"].tolist()))
    print("% sdnorm: {}".format(sdnorm.tolist()))
    minm, maxm = mnorm.min(), mnorm.max()
    scale = max(abs(minm),abs(maxm))
    print("% {}: min = {}, max = {}, mean = {}".format(m,measures[m].min(), measures[m].max(), measures[m].mean()))
    print("% {}: min = {}, max = {}, scale = {}".format(m, minm, maxm, scale))
    shift = 0.1
    xmid =  (i + 1) * shift + 2 * i + 1 
    print(" \\draw[gray] ({}, 0) -- ({}, {});".format(
        xmid, xmid, len(tblist)))
    print(" \\path[fill=blue,opacity=0.2] ({}, -0.3) rectangle ({}, {});".format(
        xmid - 1, xmid +1, len(tblist) + 0.1))
    print(" \\node[anchor=base] at ({}, {}) {{{}}};".format(
        xmid, len(tblist) + 0.5, m))
    print(" \\draw ({},-0.5) -- ++(0, -3pt) -- ++(0, 3pt) --"
          "({}, -0.5) -- ++(0, -3pt) -- ++(0, 3pt) --"
          "({}, -0.5) -- ++(0,-3pt);".format(xmid - 1, xmid, xmid + 1))
    print(" \\node[inner sep=1pt,font=\\tiny,anchor=north west] at ({}, -0.5)"
            "{{{:0.2f}}};".format( xmid - 1, measures[m].min()))
    print(" \\node[inner sep=1pt,font=\\tiny,anchor=north east] at ({}, -0.5)"
            "{{{:0.2f}}};".format( xmid + 1, measures[m].max()))
    for j,(sc, scsd, tb) in enumerate(sorted(zip(mnorm, sdnorm, tblist))):
        xpos = xmid + sc / scale
        anchor = "south east" if sc > 0 else "south west"
        if sc / scale < 0:
            xshift = -min(sc / scale + 1, 0.3)
        else:
            xshift = min(1 - sc / scale, 0.3)
        xshift = xshift * 6.9
        print("% {}/{} shift: {}, pos: {}".format(m, tb, xshift, sc / scale))
        print("  \\node[{}, anchor={},xshift={}mm] at ({},{}) {{{}}};".format(
            tbinfo[tb]['fam'], anchor, xshift, xpos, j,
            tbinfo[tb]['id'].replace('_',r'.')))
        print("  \\node[dot] at ({},{}) {{}};".format( xpos, j))
#        print("  \\draw ({},{}) -- ({}, {});".format(
#            xpos - scsd/scale, j, xpos + scsd/scale, j))

print(r'\end{tikzpicture}')
