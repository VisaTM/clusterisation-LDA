#!/usr/bin/env python3
# coding: utf-8
import pandas as pd
from scipy.stats import entropy
import gensim
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Jensen–Shannon divergence : JSD(P,Q) = 1/2D(P,M)+ 1/2D(Q,M) M=1/2(P+Q) D=entropy
def jensen_shannon(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, M) + entropy(Q, M))


def js_acp(distributions):

    dist_matrix = squareform(pdist(distributions, metric=jensen_shannon))
    return acp(dist_matrix)


def acp(pair_dists, n_components=2):
    pair_dists = np.asarray(pair_dists, np.float64)
    # Decomposition en valeurs singulieres
    n = pair_dists.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = - H.dot(pair_dists ** 2).dot(H) / 2
    eigvals, eigvecs = np.linalg.eig(B)
    ix = eigvals.argsort()[::-1][:n_components]
    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]
    eigvals[np.isclose(eigvals, 0)] = 0
    if np.any(eigvals < 0):
        ix_neg = eigvals < 0
        eigvals[ix_neg] = np.zeros(eigvals[ix_neg].shape)
        eigvecs[:, ix_neg] = np.zeros(eigvecs[:, ix_neg].shape)

    return np.sqrt(eigvals) * eigvecs


def _df_with_names(data, index_name, columns_name):
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _topic_coordinates(topic_term_dists, topic_proportion):
    K = topic_term_dists.shape[0]
    mds_res = js_acp(topic_term_dists)
    assert mds_res.shape == (K, 2)
    mds_df = pd.DataFrame({'x': mds_res[:, 0], 'y': mds_res[:, 1], 'topics': range(1, K + 1), 'Frequence': topic_proportion * 100})
    return mds_df


def _series_with_name(data, name):
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)


def acp_lda(ldamodel, mycorpus, mydict):
    lda = ldamodel
    if not gensim.matutils.ismatrix(mycorpus):
       corpus_csc = gensim.matutils.corpus2csc(mycorpus, num_terms=len(mydict))
    else:
       corpus_csc = mycorpus
    vocab = list(mydict.token2id.keys())
    beta = 0.01
    fnames_argsort = np.asarray(list(mydict.token2id.values()), dtype=np.int_)
    term_freqs = corpus_csc.sum(axis=1).A.ravel()[fnames_argsort]
    term_freqs[term_freqs == 0] = beta
    doc_lengths = corpus_csc.sum(axis=0).A.ravel()

    assert term_freqs.shape[0] == len(mydict),\
        'Term frequencies and mydict have different shape {} != {}'.format(
            term_freqs.shape[0], len(mydict))
    assert doc_lengths.shape[0] == len(mycorpus),\
        'Document lengths and mycorpus have different sizes {} != {}'.format(doc_lengths.shape[0], len(mycorpus))

    if hasattr(lda, 'lda_alpha'):
        num_topics = len(lda.lda_alpha)
    else:
        num_topics = lda.num_topics

    if hasattr(lda, 'lda_beta'):
            gamma = lda.inference(mycorpus)
    else:
            gamma, _ = lda.inference(mycorpus)

    doc_topic_dists = gamma / gamma.sum(axis=1)[:, None]

    assert doc_topic_dists.shape[1] == num_topics,\
          'Document topics and number of topics do not match {} != {}'.format(
         doc_topic_dists.shape[1], num_topics)

# get the topic-term distribution straight from gensim without
# iterating over tuples
    if hasattr(lda, 'lda_beta'):
       topic = lda.lda_beta
    else:
       topic = lda.state.get_lambda()
       topic = topic / topic.sum(axis=1)[:, None]
       topic_term_dists = topic[:, fnames_argsort]
    assert doc_topic_dists.shape[1] == num_topics,\
        'Document topics and number of topics do not match {} != {}'.format(doc_topic_dists.shape[1], num_topics)
    #print(term_freqs)
    topic_term_dists = _df_with_names(topic_term_dists, 'topic', 'term')
    doc_topic_dists = _df_with_names(doc_topic_dists, 'doc', 'topic')
    term_frequency = _series_with_name(term_freqs, 'term_frequency')
    doc_lengths = _series_with_name(doc_lengths, 'doc_length')
    vocab = _series_with_name(vocab, 'vocab')
    topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()
    topic_proportion = (topic_freq / topic_freq.sum())
    topic_order = topic_proportion.index
    # reorder all data based on new ordering of topics
    topic_freq = topic_freq[topic_order]
    topic_term_dists = topic_term_dists.iloc[topic_order]
    doc_topic_dists = doc_topic_dists[topic_order]
    term_topic_freq = (topic_term_dists.T * topic_freq).T
    term_frequency = np.sum(term_topic_freq, axis=0)
    topic_coordinates = _topic_coordinates(topic_term_dists, topic_proportion)
    # print(topic_coordinates)
    return topic_coordinates
