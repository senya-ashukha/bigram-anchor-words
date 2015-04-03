import math
import numpy as np

from copy import copy
from scipy import sparse

from tmtk.collection.collection import bag_of_words

def m_matrix(documents, wrd_count):
    doc_count = len(documents)
    m_mtx = sparse.lil_matrix((wrd_count, doc_count-1), dtype=np.float64)

    for num_doc, doc in enumerate(documents):
        for wrd, count in doc:
            m_mtx[wrd, num_doc-1] = count

    M = m_mtx.tocsr()

    new_indptr = np.zeros(M.indptr.shape[0], dtype=np.int32)
    new_indices = np.zeros(M.indices.shape[0], dtype=np.int32)
    new_data = np.zeros(M.data.shape[0], dtype=np.float64)

    indptr_counter = 1

    for i in xrange(M.indptr.size - 1):
        start, end = M.indptr[i], M.indptr[i + 1]
        new_indptr[indptr_counter] = new_indptr[indptr_counter-1] + end - start
        new_data[new_indptr[indptr_counter-1]:new_indptr[indptr_counter]] = M.data[start:end]
        new_indices[new_indptr[indptr_counter-1]:new_indptr[indptr_counter]] = M.indices[start:end]
        indptr_counter += 1

    new_indptr = new_indptr[0:indptr_counter]
    new_indices = new_indices[0:new_indptr[indptr_counter-1]]
    new_data = new_data[0:new_indptr[indptr_counter-1]]

    M = sparse.csr_matrix((new_data, new_indices, new_indptr))
    M = M.tocsc()
    return M

def topic_cov_mtx(m_mtx):
    m_mtx = copy(m_mtx)
    vocabSize, numdocs = m_mtx.shape[0], m_mtx.shape[1]

    diag_M = np.zeros(vocabSize)

    for j in xrange(m_mtx.indptr.size - 1):
        start, end = m_mtx.indptr[j], m_mtx.indptr[j + 1]

        wpd = np.sum(m_mtx.data[start:end])

        row_indices = m_mtx.indices[start:end]

        diag_M[row_indices] += m_mtx.data[start:end]/(wpd*(wpd-1))
        m_mtx.data[start:end] = m_mtx.data[start:end]/math.sqrt(wpd*(wpd-1))

    Q = np.array((m_mtx*m_mtx.transpose()/numdocs).todense()) - np.diag(diag_M/numdocs)

    return Q

def random_projection(mtx, new_dim=1100):
    old_dim = mtx.shape[0]
    r_mtx = np.searchsorted(
        np.cumsum([1.0/6, 2.0/3, 1.0/6]), np.random.random_sample(new_dim * old_dim)) - 1
    r_mtx = np.reshape(math.sqrt(3) * r_mtx, (new_dim, old_dim))
    return np.dot(r_mtx, mtx)

def gram_shmidt_step(m_mtx, basis, j, candidates, dist):
    max_dist_idx = candidates[np.argmax(
        [dist(m_mtx[i]) if not np.isnan(dist(m_mtx[i])) else 0 for i in candidates])]
    if j >= 0: basis[j] = m_mtx[max_dist_idx]/math.sqrt(dist(m_mtx[max_dist_idx]))

    return max_dist_idx

def projection_find(m_mtx, r, candidates, dist=lambda x: np.dot(x, x)):
    dim, m_mtx = m_mtx.shape[1], np.array(m_mtx.copy())
    anchor_indices, basis = np.zeros(r, dtype=np.int), np.zeros((r-1, dim))

    for j in range(-1, r - 1):
        if j >= 0:
            for i in candidates:
                m_mtx[i] -= m_mtx[anchor_indices[0]] if j == 0 else np.dot(m_mtx[i], basis[j-1]) * basis[j-1]
        anchor_indices[j+1] = gram_shmidt_step(m_mtx, basis, j, candidates, dist)

    return list(anchor_indices)

def find_anchors(cov_mtx, candidates, num_topics=100):
    cov_mtx = row_normolized(copy(cov_mtx))
    cov_mtx_red = random_projection(cov_mtx.T).T
    anchor_indices = projection_find(cov_mtx_red, num_topics, candidates)

    return anchor_indices

def logsum_exp(y):
    m = y.max()
    return m + np.log((np.exp(y - m)).sum())

def row_normolized(mtx):
    for row in mtx:
        denom = row.sum()
        row /= denom if denom else 1.0
    return mtx

def col_normolized(mtx):
    for row in mtx.T:
        denom = row.sum()
        row /= denom if denom else 1.0
    return mtx

def RecoverL2(y):
    global x, XX

    eps = 10e-7
    c1 = 10**(-4)
    c2 = 0.75
    XY = np.dot(x, y)
    YY = float(np.dot(y, y))

    (K,n) = x.shape
    alpha = np.ones(K)/K

    log_alpha = np.log(alpha)

    it = 1
    aXX = np.dot(alpha, XX)
    aXY = float(np.dot(alpha, XY))
    aXXa = float(np.dot(aXX, alpha.transpose()))

    grad = 2*(aXX-XY)
    new_obj = aXXa - 2*aXY + YY

    stepsize = 1
    decreased = False
    itter = 0;
    while 1:
        itter += 1
        eta = stepsize
        old_alpha = copy(alpha)
        old_log_alpha = copy(log_alpha)
        if new_obj == 0:
            break
        if stepsize == 0:
            break

        it += 1
        log_alpha -= eta*grad
        log_alpha -= logsum_exp(log_alpha)
        alpha = np.exp(log_alpha)

        aXX = np.dot(alpha, XX)
        aXY = float(np.dot(alpha, XY))
        aXXa = float(np.dot(aXX, alpha.transpose()))

        old_obj = new_obj
        new_obj = aXXa - 2*aXY + YY
        if not new_obj <= old_obj + c1*stepsize*np.dot(grad, alpha - old_alpha):
            stepsize /= 2.0 #reduce stepsize
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            decreased = True
            continue

        old_grad = copy(grad)
        grad = 2*(aXX-XY)

        if (not np.dot(grad, alpha - old_alpha) >= c2*np.dot(old_grad, alpha-old_alpha)) and (not decreased):
            stepsize *= 2.0
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            continue

        decreased = False

        lam = copy(grad)
        lam -= lam.min()

        gap = np.dot(alpha, lam)
        convergence = gap
        if (convergence < eps or itter > 200):
            break

    return alpha

def recover_word_topic(cov_mtx, anchors):
    V, K = cov_mtx.shape[0], len(anchors)
    P_w = np.dot(cov_mtx, np.ones(V))
    row_normolized(cov_mtx)

    global x, XX
    x, XX = cov_mtx[anchors], np.dot(cov_mtx[anchors], cov_mtx[anchors].T)

    A = np.matrix(np.diag(P_w)) * np.matrix(map(RecoverL2, cov_mtx))
    return np.array(A / A.sum(0))

def find_candidate(m_mtx, k=10):
    candidate_anchors = []

    for i in xrange(m_mtx.shape[0]):
        if len(np.nonzero(m_mtx[i, :])[1]) > k:
            candidate_anchors.append(i)

    return candidate_anchors

def anchor_model(train, test, wrd_count, num_topics=100, metrics=None, verbose=False):
    bw_train, bw_test = bag_of_words(train), bag_of_words(test)

    m_mtx = m_matrix(bw_train, wrd_count)
    cov_matrix = topic_cov_mtx(m_mtx)
    candidate_anchors = find_candidate(m_mtx)
    anchors = find_anchors(cov_matrix, candidate_anchors, num_topics)
    word_topic = recover_word_topic(cov_matrix, anchors)
    if metrics:
        metric_val = [metric(word_topic, train, test) for metric in metrics]
        print 'end: %s' % ' '.join(metric_val)

    return word_topic, anchors

def print_topics(F, id_to_wrd, anch, top=8):
    for k in xrange(len(anch)):
        topwords = np.argsort(F[:, k])[-top:][::-1]
        print id_to_wrd[anch[k]], ':',
        print ' '.join([id_to_wrd[w] for w in topwords])