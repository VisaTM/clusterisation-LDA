#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import os
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import pyLDAvis.gensim
import json
import csv
import pandas as pd


def topicswords(lda, nb=200):
    # affiche et sauvegarde les 200 mots les plus pertinents dans chaque topic
    with open("topicLDA.txt", "w+") as f:
        for t in range(lda.num_topics):
            f.write("TOPIC " + str(t + 1) + "\n")
            for word in lda.show_topic(t, nb):
                f.write("Term : " + str(word[0]) + ", Weight : " + str(word[1]) + "\n")
            f.write("\n")
        f.close()


def topicsdoc3(lda, corpus):
    with open("topicLDADoc.json", "w") as f:
        all_topics = lda.get_document_topics(corpus, per_word_topics=True)
        test = "{\"Documents\": ["
        i = 1
        for doc_topics, word_topics, phi_values in all_topics:
            test = test + "{\"Filename\" : " + "\"Document" + str(i) + "\" , \"Clusters\" :["
            for topics in doc_topics:
                test = test + "{\"ID\" : \"" + str(topics[0] + 1) + "\" , \"Weight\" : \" " + str(topics[1]) + '\"} ,'
            test = test[:len(test) - 1] + "]},"
            i += 1
        test = test[:len(test) - 1] + "]}"
        f.write(test)
        f.close()


def topicsdoc2(lda, corpus):
    with open("topicLDADoc.txt", "w") as f:
        all_topics = lda.get_document_topics(corpus, per_word_topics=True)
        i = 1
        print(all_topics)
        for doc_topics, word_topics, phi_values in all_topics:
            f.write('Document num: ' + str(i) + '\n')
            f.write('Document topics:' + str(doc_topics) + "\n")
            f.write('Word topics:' + str(word_topics) + "\n")
            f.write('Phi values:' + str(phi_values) + "\n")
            f.write(" ")
            f.write('-------------- \n')
            i += 1
        f.close()


def topicsdoc(lda, corpus):
    all_topics = lda.get_document_topics(corpus, per_word_topics=True)
    df = pd.DataFrame(np.zeros((corpus.num_docs, lda.num_topics)))
    df = df.rename(columns=lambda x: 'Topic' + str(x + 1))
    df = df.rename(index=lambda x: 'Document' + str(x + 1))
    i = 1
    for doc_topics, word_topics, phi_values in all_topics:
        for topics, poids in doc_topics:
            df.iloc[int(i - 1), topics] = poids
        i += 1
    return df


def coordonneestoJson(ldamodel, coordonnees, doctopic, path="coordonnees.json"):
    coordonnees.to_json('tmp.json', orient='split')
    with open(path, 'w') as dest_file:
        cluster_data = ""
        with open('tmp.json', 'r') as source_file:
            for line in source_file:
                element = json.loads(line.strip())
                if 'columns' in element:
                    del element['columns']
                if 'index' in element:
                    del element['index']
                if 'data' in element:
                    cluster_data = '{"Total" : ' + str(len(element["data"])) + ' , "Clusters" : ['
                    for cluster in element["data"]:
                        cluster_data = cluster_data + '{ "Coordinates" :"' + str(cluster[0]) + ',' + str(cluster[1]) + \
                                       '", "Name" : "' + str(ldamodel.show_topic(cluster[2]-1, 1)[0][0]) + '", "Id":"' + str(
                            cluster[2]) + '", "Inertia": ' \
                                       + str(cluster[3]) + ', "Terms" : ['
                        # Les 200 termes qui constituent le cluster
                        for word in ldamodel.show_topic(int(cluster[2] - 1), 200):
                            cluster_data = cluster_data + '{ "Term" :"' + str(word[0]) + \
                                           '", "Frequency" : ' + str(word[1]) + "},"
                        cluster_data = cluster_data[:len(cluster_data) - 1] + '], "Documents" : ['
                        nbdoc = 0
                        # si la somme des topics pour chaque document est nulle
                        if doctopic.sum()[cluster[2] - 1] == 0:
                            cluster_data = cluster_data + ' ]'
                        df = pd.read_csv('metadata.tsv', sep='\t')
                        df = df.drop(df.columns[0], axis=1)
                        df = df.sort_values('Index', ascending=True)
                        dftmp = pd.concat([doctopic.reset_index(drop=True), df], axis=1)
                        dftmp=dftmp.sort_values(dftmp.columns[cluster[2] - 1], ascending=False)
                        for i in range(len(dftmp)):
                            if dftmp.iloc[i, 0] != 0:
                                nbdoc += 1
                                dfmeta = dftmp.iloc[i, len(doctopic.columns):]
                                dfmeta["Document"]=str(dftmp.index.values[i])
                                dfmeta["Weight"] = str(dftmp.iloc[i, cluster[2] - 1])
                                df_json = dfmeta.to_json(orient='index')
                                cluster_data = cluster_data + df_json +","
                        cluster_data = cluster_data[:len(cluster_data) - 1] + '], "Number of documents" : ' + str(
                            nbdoc) + ', "Number of terms" : ' + str(
                            len(ldamodel.show_topic(int(cluster[2] - 1), 200))) + ' } ,'

                    cluster_data = cluster_data[:len(cluster_data) - 1] + "]}"
                    del element['data']
                dest_file.write(cluster_data)
    if os.path.exists("tmp.json"):
        os.remove('tmp.json')


# Réduction de matrice à 500 lignes par 50 colonnes aléatoires
def echantillonMatrix(dataframe, path="./exemple.txt"):
    rows = np.random.choice(dataframe.index.values, 500)
    df200 = dataframe.loc[rows]
    df500 = df200.sample(50, axis=1)
    df500.to_csv(path, sep='\t', encoding='utf-8')


# #VISUALIZATION
#
def LDAvis(ldamodel, corpus, dictionnaire):
    data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionnaire)
    pyLDAvis.save_html(data, 'vis.html')


def Nuagesmots(ldamodel):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    cloud = WordCloud(background_color='white', width=2500, height=1800, max_words=10, colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i], prefer_horizontal=1.0)
    topics = ldamodel.show_topics(formatted=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
