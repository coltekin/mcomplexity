#!/usr/bin/env python3
"""
"""

import sys
import os.path
from collections import Counter
import argparse
import numpy as np
from multiprocessing import Pool
import zlib
import random
import re
import glob

from conllu import conllu_sentences






def score_file(fname, ctype='UD'):
    nodes = []
    if ctype == 'UD':
        for sent in conllu_sentences(fname):
            nodes.extend(sent.nodes[1:])
    else: # PBC
        nodes = read_pbc(fname)

    ttr = []
    msp = []
    pos_ent = []
    pos_count = []
    feat_ent = []
    feat_count = []
    cent_form_feat = []
    cent_feat_form = []

    if ctype != 'UD':
        msp = pe = pc = fe = fc = pos_ent = pos_count =\
        feat_ent = feat_count = form_feat = feat_form =\
        cent_form_feat = cent_feat_form = [0.0] * len(ttr)

    return fname, (ttr, msp, pe, pc, fe, fc, pos_ent, pos_count, feat_ent,
            feat_count, form_feat, feat_form, cent_form_feat, cent_feat_form)

# 
# fmt = "{}" + "{}{{}}".format(opt.separator) *16
# print("# sample_size = {}, samples = {}".format(
#     opt.sample_size, opt.samples))
# print(fmt.format('fname', 'ttr', 'ttr_sd', 'msp', 'msp_sd',
#     'pos_ent', 'pos_ent_sd', 'pos_types', 'pos_types_sd',
#     'feat_ent', 'feat_ent_sd', 'feat_types', 'feat_types_sd',
#     'cent_form_feat', 'cent_form_feat_sd',
#     'cent_feat_form', 'cent_feat_form_sd'), flush=True)
# 
# for fname, (ttr, msp, pe, pc, fe, fc, pos_ent, pos_count, feat_ent, feat_count, form_feat, feat_form, cent_form_feat, cent_feat_form) in res:
#     print(fmt.format(os.path.basename(fname).replace('.conllu', ''),
#                      np.mean(ttr), np.std(ttr),
#                      np.mean(msp), np.std(msp),
#                      np.mean(pos_ent), np.std(pos_ent),
#                      np.mean(pos_count), np.std(pos_count),
#                      np.mean(feat_ent), np.std(feat_ent),
#                      np.mean(feat_count), np.std(feat_count),
#                      np.mean(cent_form_feat), np.std(cent_form_feat), 
#                      np.mean(cent_feat_form), np.std(cent_feat_form)),
#                      flush=True)
# 
# nodes = []
# for sent in conllu_sentences(opt.files[0]):
#     nodes.extend(sent.nodes[1:])
# smpl = sample(nodes, opt.sample_size)
# juola_complexity(nodes)

def read_treebank(tbdir):
    sentences = []
    for tbf in glob.glob(tbdir + '/*.conllu'):
        sentences.extend(conllu_sentences(tbf))
    return sentences


def sample_nodes(sentences, sample_size=1000, random_sample=True,
        filter_num=True, filter_pos={'X', 'PUNCT'}):
    """ Filter/sample sentences from given treebank sentences.

    Arguments:
    sample_size:    The size of the samples in number of nodes. If None (or
                    anything that evaluates to False, the whole
                    corpus is used.
    random_sample:  Sample formed by chosing sentences randomly
                    with replacement.  The order within the sentences
                    are preserved. If False, the order is not reandomized.
    filter_pos:     Set of POS tags to skip while creating the
                    node list.
    filter_num:     Skip the numbers (if written as arabic numerals).

    """
    nodes = []
    i = -1
    while not sample_size or len(nodes) < sample_size:
        if random_sample:
            i = random.randrange(len(sentences))
        else:
            i = (i + 1) % len(sentences)
        for n in sentences[i].nodes[1:]:
            if filter_pos and n.upos in filter_pos:
                continue
            elif filter_num and n.upos == 'NUM' and not n.form.isalpha():
                continue
            elif n.form is None or n.lemma is None: # error in some treebanks
                continue
            nodes.append(n)
        if not sample_size and i == len(sentences):
            break
    if sample_size:
        nodes = nodes[:sample_size]
    return nodes

def get_ttr(sentences, sample_size=1000, random_sample=True,
        lowercase=True, filter_num=True, filter_pos={'X', 'PUNCT'},
        **kwargs):
    """ Calculate the type/token ratio on a sample of the given treebank.

    Arguments:
    lowercase:      Convert the words to lowercase.

    Other arguments are as defined in sample_nodes().

    Return value is the type/token ratio over the sample.
    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)


    if lowercase:
        words = [n.form.lower() for n in nodes]
    else:
        words = [n.form for n in nodes]
    return len(set(words)) / len(words)

def get_msp(sentences, sample_size=1000, random_sample=True,
        lowercase=True, filter_num=True, filter_pos={'X', 'PUNCT'},
        **kwargs):
    """ Calculate the 'mean size of paradigm' on a sample of the given treebank.

    Arguments:
    lowercase:      Convert the words to lowercase.

    Other arguments are as defined in sample_nodes().

    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    if lowercase:
        nlemmas = len(set((x.lemma.lower() for x in nodes)))
        nwords = len(set((x.form.lower() for x in nodes)))
    else:
        nlemmas = len(set((x.lemma for x in nodes)))
        nwords = len(set((x.form for x in nodes)))
    return (nwords / nlemmas)

def get_wh_lh(sentences, sample_size=1000, random_sample=True,
        lowercase=True, filter_num=True, filter_pos={'X', 'PUNCT'},
        smooth=None, **kwargs):
    """ Calculate the unigram words and lemma entropy.

    Arguments:
    lowercase:  Convert the words to lowercase.
    smooth:     Apply smoothing. A numeric value indicates 'add alpha'
                smoothing, 'GT' means absolute discouting based
                on Good-Tring. [these are currently not
                (re)implemented here as they are not used in the
                paper.]

    Other arguments are as defined in sample_nodes().

    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    if lowercase:
        clemmas = Counter((x.lemma.lower() for x in nodes))
        cwords = Counter((x.form.lower() for x in nodes))
    else:
        clemmas = Counter((x.lemma for x in nodes))
        cwords = Counter((x.form for x in nodes))
    nlemmas = sum(clemmas.values())
    nwords = sum(cwords.values())
    wh, lh = 0, 0
    for w in cwords:
        p = cwords[w] / nwords
        wh -= p * np.log2(p)
    for l in clemmas:
        p = clemmas[l] / nlemmas
        lh -= p * np.log2(p)
    return wh, lh

def get_mfh(sentences, sample_size=1000, random_sample=True,
        filter_num=True, filter_pos={'X', 'PUNCT'},
        smooth=None, **kwargs):
    """ Calculate the morphological feature (and POS) entropy.
    POS en

    Arguments:
    smooth:     Apply smoothing. A numeric value indicates 'add alpha'
                smoothing, 'GT' means absolute discouting based
                on Good-Tring. [these are currently not
                (re)implemented here as they are not used in the
                paper.]

    Other arguments are as defined in sample_nodes().
    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    cfeat = Counter()
    cpos = Counter()
    npos, nfeat = 0, 0
    for node in nodes:
        if node.feats:
            feats = node.feats.split('|')
            cfeat.update(feats)
        cpos.update([node.upos])
    npos = sum(cpos.values())
    nfeat = sum(cfeat.values())
    ph, mfh = 0, 0
    for pos in cpos:
        p = cpos[pos] / npos
        ph -= p * np.log2(p)
    for feat in cfeat:
        p = cfeat[feat] / nfeat
        mfh -= p * np.log2(p)
    return mfh, ph


def random_words(words, uniform=False):
    alphabet = {str(i):i for i in range(10)}
    if uniform:
        chcount = Counter(set((ch for w in words for ch in w)))
    else:
        chcount = Counter((ch for w in words for ch in w ))
    if len(chcount) > 256:
        # Non-alphabetic scripts are not comparable,
        # we calculate a value for the sake of robustness
        print("Warning: more than 255 characters", file=sys.stderr)
        chcount = Counter(dict(chcount.most_common(255)))
    n = sum(chcount.values())
    p = [chcount[i]/n for i in chcount]
    worddict = set(words)
    rdict = {w:''.join(np.random.choice(list(chcount), size=len(w),
                    replace=True, p=p)) for w in worddict}
    return [rdict[w] for w in words]

def get_ws(sentences, sample_size=1000, random_sample=True,
        lowercase=True, filter_num=True, filter_pos={'X', 'PUNCT'},
        **kwargs):
    """Calculate the information loss when word-internal structure is destroyed.

    Arguments:
    lowercase:  Convert the words to lowercase.

    Other arguments are as defined in sample_nodes().

    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    if lowercase:
        words = [n.form.lower() for n in nodes]
    else:
        words = [n.form for n in nodes]
    rwords = random_words(words)

    alphabet = {' ': 0}
    for w in rwords:
        for ch in w:
            if ch not in alphabet:
                alphabet[ch] = len(alphabet) 

    text = ' '.join(words)      # original text
    rtext = ' '.join(rwords)    # randomized `cooked' text
    # We binarize them to remove the effects of Unicode encoding
    bintext =  bytearray([alphabet.get(ch, 0) for ch in text])
    comptext = zlib.compress(bintext, level=9)
    cr = len(bintext)/len(comptext)
    rbintext =  bytearray([alphabet.get(ch, 1) for ch in rtext])
    rcomptext = zlib.compress(rbintext, level=9)
    rcr = len(rbintext)/len(rcomptext)
    return cr - rcr

def get_is(sentences, sample_size=1000, random_sample=True, 
        **kwargs): #ignore unsused arguments
    """Calculate maximum number of inflectional markers per verb.

    All arguments passed to sample_nodes().

    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample)
    fset = set()    # set of features
    fvset = set()   # set of feature-value pairs
    featcount = []  # number of features marked on each verb
    for node in nodes:
        if node.upos == 'VERB' and node.feats:
            fvlist = node.feats.split('|')
            fvset.update(fvlist)
            feats = (fv.split('=')[0] for fv in fvlist)
            fset.update(feats)
            featcount.append(len(fvlist))
    avg = 0
    if featcount:
        avg = sum(featcount)/len(featcount)
#    return len(fset), len(fvset), avg
    return len(fset)

def get_wh(*args, **kwargs):
    return get_wh_lh(*args, **kwargs)[0]

def get_lh(*args, **kwargs):
    return get_wh_lh(*args, **kwargs)[1]


measures = {
    'ttr':  ('Type/token ratio', get_ttr), 
    'msp':  ('Means size of paradigm', get_msp), 
    'ws':   ('Word structure information', get_ws), 
    'wh':   ('Word entropy (unigram)', get_wh), 
    'lh':   ('Lemma entropy', get_lh), 
    'is':   ('Inflectional synthesis', get_is), 
    'mfh':  ('Morphological feature entropy', get_mfh),
}


ap = argparse.ArgumentParser()
ap.add_argument('treebanks', nargs='+')
ap.add_argument('-j', '--nproc', default=1, type=int,
                    help='number of processes')
ap.add_argument('-s', '--samples', default=10, type=int,
                    help='number of samples')
ap.add_argument('-S', '--sample-size', default=1000, type=int)
ap.add_argument('--separator', default='\t')
ap.add_argument('-n', '--normalize', action='store_true')
ap.add_argument('-m', '--measures', default='all',
                    help='comma separated measures, or all')
ap.add_argument('-o', '--output', default='measures.txt')
args = ap.parse_args()

if args.measures == 'all':
    mlist = tuple(measures.keys())
else:
    mlist = args.measures.split(',')

def get_score(jobdesc):
    func = measures[jobdesc[0]][1]
    tb = read_treebank(jobdesc[1])
    kwargs = jobdesc[2]
    return jobdesc, func(tb,**kwargs)

kwargs = {'sample_size': args.sample_size}
joblist = []
for m in mlist:
    for tbdir in args.treebanks:
        for _ in range(args.samples):
            joblist.append((m, tbdir, kwargs))

pool = Pool(processes=args.nproc)
res = pool.map(get_score, joblist)

scores = dict()
for (m, tb, _), sc in res:
    tb = os.path.basename(tb.rstrip('/')).replace('UD_','')
    if (m, tb) not in scores:
        scores[(m, tb)] = []
    scores[(m, tb)].append(sc)

tblist = [os.path.basename(tb.rstrip('/')).replace('UD_','') \
            for tb in args.treebanks]

fmt = "\t{}" * (2*len(mlist))
head = [x for pair in zip(mlist, (m + "_sd" for m in mlist)) for x in pair]
with open(args.output, 'wt') as fp:
    print("treebank", fmt.format(*head), file=fp)
    for tb in tblist:
        print(tb, end="", file=fp)
        sclist = []
        for m in mlist:
            sc = np.array(scores[(m,tb)])
            sclist.extend((sc.mean(), sc.std()))
        print(fmt.format(*sclist), file=fp)
