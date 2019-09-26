from __future__ import division, print_function

import numpy as np
import lda
import lda.datasets


# document-term matrix
X = lda.datasets.load_reuters()
print(X)
# the vocab
vocab = lda.datasets.load_reuters_vocab()

# titles for each story
titles = lda.datasets.load_reuters_titles()

# train the model
model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(X)

# get results
topic_word = model.topic_word_
doc_topic = model.doc_topic_

# print topic probabiities for each document
n_docs = 395
n_topics = 20

with open('/home/falck/lda/topic_table.csv', 'w') as f:
    # create header
    header = 'document'
    for k in range(n_topics):
        header += ', pr_topic_{}'.format(k)
    f.write(header + '\n')

    # write one row for each document
    # col 1 : document number
    # cols 2 -- : topic probabilities
    for k in range(n_docs):
        # format probabilities into string
        str_probs = ','.join(['{:.5e}'.format(pr) for pr in doc_topic[k,:]])
        # write line to file
f.write('{}, {}\n'.format(k, str_probs))