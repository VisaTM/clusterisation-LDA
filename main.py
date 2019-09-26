#! /usr/bin/env python3
# coding: utf-8
from gensim import corpora
import warnings
warnings.simplefilter("ignore", UserWarning)
import os.path
import Analyse
import outils
import pandas as pd
import sys
from gensim.models import LdaModel

if __name__ == '__main__':
    if len(sys.argv) < 2 & len(sys.argv) > 4:
        print("USAGE: argument expected")
        sys.exit(1)

    if os.path.isfile(sys.argv[1]):
        print("Lecture de la matrice")
        Analyse.prepare(sys.argv[1])
        print("lecture du dictionnaire")
        loaded_dict = corpora.Dictionary.load('mydict.dict')
        print("lecture du corpus")
        corpus = corpora.MmCorpus('bow_corpus.mm')
        nbtopic = 5
        output = './coordonnees.json'
        if len(sys.argv) > 2:
            if int(sys.argv[2]) > 1 & int(sys.argv[2]) < 200:
                nbtopic = int(sys.argv[2])
            else:
                if os.path.isfile(sys.argv[2]):
                    print("Attachement des Metadatas")
                    a = pd.read_csv(sys.argv[3], sep="\t", )
                    b = pd.read_csv("./document.csv", sep="\t")
                    merged = a.merge(b, on='Filename')
                    merged.to_csv('./metadata.tsv', sep="\t")
                else:
                    print("Argument" + sys.argv[2] + " non ")
                    sys.exit(3)

        print("Lancement LDA")
        Analyse.LDA(loaded_dict, corpus, nbtopic)
        lda_model = LdaModel.load('lda_model.model')
        print("Lancement de l'ACP")
        if len(sys.argv) == 4:
            if os.path.isfile(sys.argv[3]):
                print("Attachement des Metadatas")
                a = pd.read_csv(sys.argv[3], sep="\t",)
                b = pd.read_csv("./document.csv",  sep="\t")
                merged = a.merge(b, on='Filename')
                merged.to_csv('./metadata.tsv', sep="\t")
            else:
                sys.exit(3)
        if os.path.exists("./document.csv"):
            os.remove("./document.csv")
        Analyse.ACPJson(lda_model, corpus, loaded_dict, output)
    else:
        sys.exit(2)
