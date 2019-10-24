from math import log
import numpy as np
import csv
import os
import re
import pickle
from operator import itemgetter
import nltk
from nltk import ngrams
from nltk.text import Text
from nltk.corpus import stopwords
import pymorphy2 as pm2
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
pmm = pm2.MorphAnalyzer()
#nltk.download("stopwords")

# Create lemmatizer and stopwords list
russian_stopwords = stopwords.words("russian")


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[A-Za-z]', '', text)
    text = [pmm.normal_forms(x)[0] for x in text.split() if x not in
            russian_stopwords]
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

if not os.path.exists('corpus.pickle'):
    corpus = []
    for idx, sent in enumerate(docs):
        if idx < 5001:
            corpus.append(' '.join(clean_text(sent)))
        else:
            break
        with open("corpus.pickle", "wb") as c:
            pickle.dump(corpus, c)
else:
    with open("corpus.pickle", "rb") as c:
        corpus = pickle.load(c)

N = len(corpus)

X = vectorizer.fit_transform(corpus)
f_matrix = X.toarray()
tf_matrix = np.transpose(f_matrix)

doc_words = vectorizer.get_feature_names()

all_n = {}
for word in doc_words:
    w_idx = doc_words.index(word)
    all_n[word] = np.count_nonzero(tf_matrix[w_idx])

docs_len = {}
whole_len = 0
for doc in corpus:
    doc_len = len(doc.split())
    docs_len[doc] = doc_len
    whole_len += doc_len
avgdl = whole_len/N


def create_matrix_bm25(all_n, docs_len, tf_matrix, corpus, doc_words, avgdl,
                       N):
    k = 2.0
    b = 0.75
    bm_matrix = []
    for idx, word in enumerate(doc_words):
        bm_matrix.append([])
        for doc in corpus:
            bm = 0
            if word in doc:
                w_idx = doc_words.index(word)
                d_idx = corpus.index(doc)
                doc_len = docs_len[doc]
                tf = tf_matrix[w_idx][d_idx]/doc_len
                n = all_n[word]
                idf = log((N-n+0.5)/(n+0.5))
                bm = idf * ((tf * (k+1))/(tf + k * (1 - b + (b * (doc_len/avgdl)))))
            bm_matrix[idx].append(bm)
    return np.array(bm_matrix)


def get_bm25_matrix(all_n, docs_len, tf_matrix, corpus, doc_words, avgdl, N):
    if os.path.exists('bm_matrix.pickle'):
        with open("bm_matrix.pickle", "rb") as m:
            bm_matrix = pickle.load(m)
    else:
        bm_matrix = create_matrix_bm25(all_n, docs_len, tf_matrix, corpus,
                                       doc_words, avgdl, N)
        with open("bm_matrix.pickle", "wb") as m:
            pickle.dump(bm_matrix, m)
    return bm_matrix


def vect_bm25(query, k, doc_words, all_n, N):
    q_words = clean_text(query)
    counts = {}
    for word in q_words:
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] += 1
    v_query = []
    for word in doc_words:
        bm = 0
        if word in counts:
            doc_len = len(q_words)
            n = all_n[word]
            tf = counts[word]/doc_len
            idf = log((N + 1) / (n + 1))
            bm = idf * ((tf * (k+1))/(tf + k))
        v_query.append(bm)
    v_query = np.array(v_query)
    return v_query


def bm25_search(query):
    doc_words = vectorizer.get_feature_names()
    bm_matrix = get_bm25_matrix(all_n, docs_len, tf_matrix, corpus, doc_words,
                                avgdl, N)
    k = 2.0
    b = 0.75
    q_vect = vect_bm25(query, k, doc_words, all_n, N)
    doc_score = q_vect.dot(bm_matrix)
    response = list(zip(docs, doc_score))
    response = sorted(response,key=itemgetter(1), reverse = True)
    return response


#response = bm25_search('разница между Китаем и Соединенными Штатами')
#print(response[:5])