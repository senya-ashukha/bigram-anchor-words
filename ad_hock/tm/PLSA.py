# -*- coding: utf-8 -*-
import sys

import numpy as np
import random as rnd
import math

from itertools import izip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def read(f_name):
    doc_count, wrd_count, str_documents = open(f_name).read().split('\n', 2)

    documents = []
    for _, words, counts in grouper(str_documents.split('\n')[:-1], 3):
        words, counts = map(int, words.split()), map(int, counts.split())
        documents.append(np.array(zip(words, counts)))
    
    return int(doc_count), int(wrd_count), np.array(documents)

_, _, documents_train = read('NIPS-collection/NIPSOld_t.txt')
_, _, documents_test = read('NIPS-collection/NIPSOld_c.txt')

def norn_mtx(x, y, axis):
    mtx = np.random.random((x, y))
    for row in mtx.T if axis == 'x' else mtx: 
        row /= row.sum()
    return mtx

def preplexity(F, T, documents):
    doc_count, wrd_count = len(T[0]), len(F[0])
    lh = 0
    for d in xrange(doc_count):
        for w, ndw in documents[d]:
            lh += ndw * math.log(np.dot(F[w, :], T[:, d]))
    return math.e ** (-lh / sum([sum([ndw for _, ndw in documents[d]]) for d in xrange(doc_count)]))

def half_random(documents_test):
    estimat_doc, control_doc = [], []
    for d in documents_test:
        est_d, con_d = [], []
        for w, wrdc in d:
            if wrdc > 1:
                est_d.append((w, wrdc/2))
                con_d.append((w, wrdc/2))
            else:
                if rnd.randint(0, 100) % 2:
                    est_d.append((w, wrdc/2))
                else:
                    con_d.append((w, wrdc/2))
        estimat_doc.append(np.array(est_d))
        control_doc.append(np.array(con_d))
        
    return np.array(estimat_doc), np.array(control_doc)

def estimate_teta(F, documents_test, num_topics=100):
    doc_count = len(documents_test)
    T = np.zeros((num_topics, doc_count))
    
    estimate_doc, control_doc = half_random(documents_test)

    for d in range(doc_count):
        for w, wrd_count in estimate_doc[d]:
            T.T[d] += (F[w] * wrd_count)
            
    for row in T.T: 
        row /= row.sum()
            
    return T, control_doc

def plsa_em(documents_train, documents_test=None, num_topics=100, num_itter=10, metric=preplexity):
    doc_count, wrd_count = len(documents_train), len(set([wrd for doc in documents_train for wrd, _ in doc]))
    F, T = norn_mtx(wrd_count, num_topics, axis='x'), norn_mtx(num_topics, doc_count, axis='y')
    
    for itter in xrange(num_itter):
        Nwt, Ntd, Nt, Nd = np.zeros((wrd_count, num_topics)), np.zeros((num_topics, doc_count)), np.zeros(num_topics), np.zeros(doc_count)
        
        for d in xrange(doc_count):
            for w, ndw in documents_train[d]:
                ndwt = F[w, :] * T[:, d]
                ndwt *= ndw * (1.0/ndwt.sum())
                
                Nwt[w] += ndwt
                Ntd[:, d] += ndwt
                Nt += ndwt
                Nd[d] += ndwt.sum()

                
        for w in xrange(wrd_count):
            F[w] = Nwt[w] / Nt
            
        for t in range(num_topics):
            T[t] = Ntd[t] / Nd
            
        Ti, docs = estimate_teta(F, documents_test)
            
        print 'itter %s prepl: train = %.2f, test = %.2f' % (itter, metric(F, T, documents_train), metric(F, Ti, docs))
        sys.stdout.flush()
        
    return F, T

F, T = plsa_em(documents_train, documents_test, num_itter=30)