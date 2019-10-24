from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
import csv
import pickle
import re
from operator import itemgetter
import pymorphy2 as pm2
pmm = pm2.MorphAnalyzer()


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = [pmm.normal_forms(x)[0] for x in text.split()]
    return text


with open('quora_question_pairs_rus.csv', 'r', encoding='utf-8') as q:
    str_corpus = csv.reader(q)
    file = list(str_corpus)

if not os.path.exists('documents.pickle'):
    docs = []
    for idx, line in enumerate(file):
        if idx != 0 and idx < 5002:
            docs.append(line[2])
    with open("documents.pickle", "wb") as c:
        pickle.dump(docs, c)
else:
    with open("documents.pickle", "rb") as c:
        docs = pickle.load(c)

if not os.path.exists('vect_corpus.pickle'):
    corpus = []
    for idx, sent in enumerate(docs):
        if idx < 5001:
            corpus.append(clean_text(sent))
        else:
            break
        with open("vect_corpus.pickle", "wb") as c:
            pickle.dump(corpus, c)
else:
    with open("vect_corpus.pickle", "rb") as c:
        corpus = pickle.load(c)

ft_model_file = 'fasttext/model.model'
ft_model = KeyedVectors.load(ft_model_file)

def create_ft_matrix(corpus):
    matrix = []
    for doc in corpus:
        doc_vectors = np.zeros((len(doc), ft_model.vector_size))
        vec = np.zeros((ft_model.vector_size,))

        for idx, lemma in enumerate(doc):
            if lemma in ft_model.vocab:
                doc_vectors[idx] = ft_model.wv[lemma]

        if doc_vectors.shape[0] is not 0:
            vec = np.mean(doc_vectors, axis=0)
        matrix.append(vec)
    return np.array(matrix)


def get_ft_matrix(corpus):
    if os.path.exists('ft_matrix.pickle'):
        with open("ft_matrix.pickle", "rb") as m:
            ft_matrix = pickle.load(m)
    else:
        ft_matrix = create_ft_matrix(corpus)
        with open("ft_matrix.pickle", "wb") as m:
            pickle.dump(ft_matrix, m)
    return ft_matrix


def query2vec(query):
    query = clean_text(query)
    lemmas_vectors = np.zeros((len(query), ft_model.vector_size))
    vec = np.zeros((ft_model.vector_size,))

    for idx, lemma in enumerate(query):
        if lemma in ft_model.vocab:
            lemmas_vectors[idx] = ft_model.wv[lemma]

    if lemmas_vectors.shape[0] is not 0:
        vec = np.array(np.mean(lemmas_vectors, axis=0))
    return vec


def get_cos_sim(v1, v2):
    cos_sim = dot(v1, v2) / (norm(v1) * norm(v2))
    return cos_sim


def ft_search(query, docs):
    response = []
    vec = query2vec(query)
    ft_matrix = get_ft_matrix(corpus)
    for idx, doc in enumerate(docs):
        if idx < len(ft_matrix):
            doc_score = get_cos_sim(vec, ft_matrix[idx])
            response.append((docs[idx], doc_score))
    response = sorted(response, key=itemgetter(1), reverse=True)
    return response


#response = ft_search('разница между Китаем и Соединенными Штатами', docs)
#print(response[:5])