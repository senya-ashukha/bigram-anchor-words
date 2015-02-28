from math import *
import multiprocessing as mp

import numpy as np
from numpy.random import RandomState
from scipy.optimize import fmin_slsqp
from tmtk.utils.math import norm_vec

def topic_cov_matrix(m_mtx):
    vocab_size, num_docs = m_mtx.shape[0], m_mtx.shape[1]
    m_mtx, m_mtx_di = np.array(m_mtx.todense()), np.zeros(vocab_size)

    for column in m_mtx.T:
        denom = column.sum() * (column.sum() - 1)
        m_mtx_di += column * 1.0 / denom
        column /= sqrt(denom)

    q_mtx = (np.dot(m_mtx, m_mtx.T) - np.diag(m_mtx_di)) / num_docs
    return q_mtx

def random_projection(mtx, new_dim=1000):
    old_dim = mtx.shape[0]
    r_mtx = np.searchsorted(np.cumsum([1.0/6, 2.0/3, 1.0/6]), RandomState(100).random_sample(new_dim * old_dim)) - 1
    r_mtx = np.reshape(math.sqrt(3) * r_mtx, (new_dim, old_dim))
    return np.dot(r_mtx, mtx)

def gram_shmidt_step(m_mtx, basis, j, candidates, dist):
    max_dist_idx = candidates[np.argmax([dist(m_mtx[i]) for i in candidates])]
    if j >= 0: basis[j] = m_mtx[max_dist_idx]/math.sqrt(dist(m_mtx[max_dist_idx]))

    return max_dist_idx

def projection_find(m_mtx, r, candidates, dist=lambda x: np.dot(x, x)):
    dim, m_mtx = m_mtx.shape[1], m_mtx.copy()
    anchor_indices, basis = np.zeros(r, dtype=np.int), np.zeros((r-1, dim))

    for j in range(-1, r - 1):
        if j >= 0:
            for i in candidates:
                m_mtx[i] -= m_mtx[anchor_indices[0]] if j == 0 else np.dot(m_mtx[i], basis[j-1]) * basis[j-1]
        anchor_indices[j+1] = gram_shmidt_step(m_mtx, basis, j, candidates, dist)

    return list(anchor_indices)

def find_anchors(cov_mtx, num_topics, candidates):
    cov_mtx = (cov_mtx.transpose() / cov_mtx.sum(1)).transpose()
    cov_mtx_red = random_projection(cov_mtx.T).T
    anchors, anchor_indices = projection_find(cov_mtx_red, num_topics, candidates)

    return anchor_indices

# sum(Ci) = 1
def constrS(x): return 1 - x.sum()

# Ci >= 0 i in [0, len(Ci)]
def constrG(x): return x.sum() - np.abs(x).sum()

def recover_l2(cov_i, iter=40):
    def fast_l2(c): return cov_norm - 2 * np.dot(c, cov_anc) + np.dot(c.T, np.dot(cov_cov_mtx_anc, c))
    cov_norm, cov_anc, c_start = sqrt((cov_i ** 2).sum()), np.dot(cov_mtx_anc, cov_i.T), norm_vec(cov_mtx_anc.shape[0])
    return fmin_slsqp(fast_l2, c_start, f_eqcons=constrS, f_ieqcons=constrG, iter=iter, iprint=-1)

def recover_word_topic(cov_mtx, anchors, n_jobs=4):
    voc_size, num_topics = cov_mtx.shape[0], len(anchors)
    p_mtx = np.matrix(np.diag(np.dot(cov_mtx, np.ones(voc_size))))
    cov_mtx = (cov_mtx.transpose() / cov_mtx.T.sum(1)).transpose()

    global cov_mtx_anc, cov_cov_mtx_anc
    cov_mtx_anc, cov_cov_mtx_anc = cov_mtx[anchors], np.dot(cov_mtx[anchors], cov_mtx[anchors].T)

    word_topic = p_mtx * np.matrix(mp.Pool(n_jobs).map(recover_l2, cov_mtx))
    return np.array(word_topic / word_topic.sum(0))

def anchor_model(documents_train, documents_test, num_topics=100, metric=None, verbose=False):

    candidate_anchors = find_candidate(documents_train)
    cov_matrix = topic_cov_matrix(documents_train)
    anchors = find_anchors(cov_matrix, num_topics, candidate_anchors)
    word_topic = recover_word_topic(cov_matrix, anchors)

    return word_topic
