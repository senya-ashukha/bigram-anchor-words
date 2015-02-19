
# coding: utf-8

# In[1]:

import sys
import scipy.io
import scipy
import numpy as np
import scipy.sparse
from copy import copy

import time
import math
from math import *
from numpy.random import RandomState
import itertools
import multiprocessing as mp
import itertools
from scipy.optimize import fmin_slsqp
from itertools import izip_longest, izip
from multiprocessing import Pool

#=================================================================================================
def prepr():
    infile = open('docword.nips.txt')
    num_docs, num_words, nnz = int(infile.readline()), int(infile.readline()), int(infile.readline())
    K = 100

    M = scipy.sparse.lil_matrix((num_words, num_docs))

    for l in infile:
        d, w, v = [int(x) for x in l.split()]
        M[w - 1, d - 1] = v

    full_vocab = 'vocab.nips.txt'
    output_matrix, output_vocab = 'M.trunc', 'V.trunc'

    cutoff = 50

    table = dict()
    numwords = 0
    with open(full_vocab, 'r') as file:
        for line in file:
            table[line.rstrip()] = numwords
            numwords += 1

    remove_word = [False]*numwords

    # Read in the stopwords
    with open('stopwords.txt', 'r') as file:
        for line in file:
            if line.rstrip() in table:
                remove_word[table[line.rstrip()]] = True

    if M.shape[0] != numwords:
        print 'Error: vocabulary file has different number of words', M.shape, numwords
        sys.exit()
        
    print 'Number of words is ', numwords
    print 'Number of documents is ', M.shape[1]

    M = M.tocsr()

    new_indptr = np.zeros(M.indptr.shape[0], dtype=np.int32)
    new_indices = np.zeros(M.indices.shape[0], dtype=np.int32)
    new_data = np.zeros(M.data.shape[0], dtype=np.float64)

    indptr_counter = 1
    data_counter = 0

    for i in xrange(M.indptr.size - 1):

        # if this is not a stopword
        if not remove_word[i]:

            # start and end indices for row i
            start = M.indptr[i]
            end = M.indptr[i + 1]
            
            # if number of distinct documents that this word appears in is >= cutoff
            if (end - start) >= cutoff:
                new_indptr[indptr_counter] = new_indptr[indptr_counter-1] + end - start
                new_data[new_indptr[indptr_counter-1]:new_indptr[indptr_counter]] = M.data[start:end]
                new_indices[new_indptr[indptr_counter-1]:new_indptr[indptr_counter]] = M.indices[start:end]
                indptr_counter += 1
            else:
                remove_word[i] = True

    new_indptr = new_indptr[0:indptr_counter]
    new_indices = new_indices[0:new_indptr[indptr_counter-1]]
    new_data = new_data[0:new_indptr[indptr_counter-1]]

    M = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr))
    M = M.tocsc()

    print 'New number of words is ', M.shape[0]
    print 'New number of documents is ', M.shape[1]

    # Output the new vocabulary
    output = open(output_vocab, 'w')
    row = 0
    with open(full_vocab, 'r') as f:
        for line in f:
            if not remove_word[row]:
                output.write(line)
            row += 1
    output.close()


    # In[4]:

    print "identifying candidate anchors"
    candidate_anchors = []

    for i in xrange(M.shape[0]):
        if len(np.nonzero(M[i, :])[1]) > 100:
            candidate_anchors.append(i)
    print len(candidate_anchors), "candidates"


    # In[5]:

    print output_vocab
    vocab = open(output_vocab)
    vocab = vocab.read()
    vocab = vocab.strip()
    vocab = vocab.split()

    return M, candidate_anchors, vocab

#=================================================================================================

def generate_Q_matrix(M):
    vocabSize, numdocs = M.shape[0], M.shape[1]
    M, diag_M = np.array(M.todense()),  np.zeros(vocabSize)

    for column in M.T:
        denom = column.sum() * (column.sum()-1)
        diag_M += column * 1.0 / denom
        column  /= sqrt(denom)
    
    Q = (np.dot(M, M.T) - np.diag(diag_M)) / numdocs
    
    return Q

def random_projection(M, new_dim=1000):
    old_dim = M.shape[0]
    R = np.searchsorted(np.cumsum([1./6, 2./3, 1./6]), RandomState(100).random_sample(new_dim*old_dim)) - 1
    R = np.reshape(math.sqrt(3)*R, (new_dim, old_dim))
    return np.dot(R, M)

def gramm_shmit_step(M, basis, j, candidates, dist): 
    max_dist_idx = candidates[np.argmax([dist(M[i]) for i in candidates])]
    if j >= 0: basis[j] = M[max_dist_idx]/np.sqrt(dist(M[max_dist_idx]))
    
    return M[max_dist_idx], max_dist_idx

def Projection_Find(M_orig, r, candidates, dist=lambda x: np.dot(x, x)):
    dim, M = M_orig.shape[1], M_orig.copy()    
    anchor_words, anchor_indices, basis = np.zeros((r, dim)), np.zeros(r, dtype=np.int), np.zeros((r-1, dim))
    
    for j in range(-1, r - 1):
        if j >= 0:
            for i in candidates: M[i] -= anchor_words[0] if j == 0 else np.dot(M[i], basis[j-1]) * basis[j-1]    
        anchor_words[j+1], anchor_indices[j+1] = gramm_shmit_step(M, basis, j, candidates, dist)
        
    return (anchor_words, list(anchor_indices))

def findAnchors(Q, K, candidates):
    row_sums = Q.sum(1)    
    Q = (Q.transpose() / row_sums).transpose()

    Q_red = random_projection(Q.T).T
    anchors, anchor_indices = Projection_Find(Q_red, K, candidates)

    Q = (Q.transpose() * row_sums).transpose()

    return anchor_indices

# sum(Ci) = 1
def constrS(x):
    return 1 - x.sum()

# Ci >= 0 i in [0, len(Ci)]
def constrG(x):
    return x.sum() - np.abs(x).sum()

c0 = np.random.random(100); c0 /= c0.sum(); 

def RecoverL2((Qw, Qanchors)):
    def L2(c): return sqrt(((Qw - np.dot(c.T, Qanchors))**2).sum())
    c  = fmin_slsqp(L2, c0, f_eqcons=constrS, f_ieqcons=constrG, iter=10, iprint=-1)
    print L2(c0), L2(c)
    return c

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def RecoverA(Q, anchors, n_jobs=4):
    print 'RecoverA'
    V, K = Q.shape[0], len(anchors) 
    P_w = np.matrix(np.diag(np.dot(Q, np.ones(V))))
    Q = (Q.transpose() / Q.T.sum(1)).transpose()

    A, pool= [], mp.Pool(n_jobs)

    def t():
        for w in xrange(V): yield (Q[w], Q[anchors])

    for args in grouper(t(), n_jobs*5):
        A += pool.map(RecoverL2, args)

    A = P_w * np.matrix(A)
     
    return np.array(A / A.sum(0))

if __name__ == "__main__":
    M, candidate_anchors, vocab = prepr()
    Q = generate_Q_matrix(M)
    anchors = findAnchors(Q, 100, candidate_anchors)
    A = RecoverA(Q, anchors)

    for k in xrange(len(anchors)):
        topwords = np.argsort(A[:, k])[-10:][::-1]
        print vocab[anchors[k]], ':',
        for w in topwords:
            print vocab[w],
        print ""