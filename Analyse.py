#! /usr/bin/env python3
# coding: utf-8
from gensim import corpora
from gensim.utils import simple_preprocess
import pandas as pd
import re
import ACP
import outils
from gensim.models import LdaMulticore
import csv

def prepare(docterm):
    df = pd.read_csv(docterm, sep="\t", index_col=0, header=0)
    document = list()
    # Consititution du dictionnaire avec n * le terme dans le document ( n est égal à la valeur de la matrice df[col][row]
    listmot = list()
    for col in df.columns:
        for row in df[col].iteritems():
            if row[1] > 0:
                for i in range(0, row[1]):
                    # A AMELIORER
                    # remplace les caractères non alpha ( chiffres et accent) par _Ω_
                    text = re.sub('[0-9]+|[^a-zA-Z ]+', '_Ω_', row[0])
                    listmot.append(text.replace(" ", "_"))
        document.append(' '.join(listmot))
        del listmot[:]
    tokenized_list = [simple_preprocess(doc, max_len=50) for doc in document]
    mydict = corpora.Dictionary()
    #On récupére le texte et l'index de chaque document et l'enregistre dans documents.csv pour attacher les métadonnées ultérieurement
    with open('document.csv', 'w') as csvfile:
        fieldnames = ['Index', 'Filename', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,  delimiter='\t')
        writer.writeheader()
        for index in range(len(df.columns)):
            writer.writerow({'Index': index, 'Filename': df.columns[index].split('.')[0], 'Text': tokenized_list[index]})

    mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
    #word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]
    # print("COMTPE MOT")
    # print(word_counts)
    mydict.save_as_text("dict.txt")
    # Save the dict and corpus
    mydict.save('mydict.dict')  # save dict
    corpora.MmCorpus.serialize('bow_corpus.mm', mycorpus)


def LDA(dictionnaire, corpus, nbtopic=5):
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionnaire, random_state=100, num_topics=nbtopic, passes=10,
                             chunksize=1000,
                             batch=False, alpha='asymmetric', decay=0.5, offset=64, eta=None, eval_every=0,
                             iterations=100, gamma_threshold=0.001, per_word_topics=True)
    #  save the lda model
    lda_model.save('lda_model.model')


def ACPJson(lda_model, corpus, dictionnaire, path="./coordonnees.json"):
    coordonnees = ACP.acp_lda(lda_model, corpus, dictionnaire)
    doctopic = outils.topicsdoc(lda_model, corpus)
    outils.coordonneestoJson(lda_model, coordonnees, doctopic, path)
