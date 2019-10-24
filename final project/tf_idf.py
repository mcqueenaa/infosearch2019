import os
import numpy as np
import re
import csv
import pickle
from math import log
from nltk.corpus import stopwords
from operator import itemgetter
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


def create_tf_idf_matrix(tf_matrix, N, all_n, doc_words):
    tf_idf_matrix = []
    for idx, word in enumerate(doc_words):
        tf_idf_matrix.append([])
        for doc in corpus:
            tf_idf = 0
            if word in doc:
                w_idx = doc_words.index(word)
                d_idx = corpus.index(doc)
                tf = tf_matrix[w_idx][d_idx]/len(doc)
                n = all_n[word]
                idf = log(N/n)
                tf_idf = tf * idf
            tf_idf_matrix[idx].append(tf_idf)
    return np.array(tf_idf_matrix)


def query2vec_tf_idf(query, doc_words, N, all_n):
    query = clean_text(query)
    counts = {}
    for word in query:
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] += 1
    v_query = []
    for word in doc_words:
        if word in counts:
            tf = counts[word]/len(query)
            n = all_n[word]
            idf = log((N+1)/(n+1))
            tf_idf = tf * idf
            v_query.append(tf_idf)
        else:
            v_query.append(0)
    return np.array(v_query)


def tf_idf_search(query):
    v_query = query2vec_tf_idf(query, doc_words, N, all_n)
    tf_idf_matrix = get_tf_idf_matrix(tf_matrix, N, all_n, doc_words)
    doc_score = v_query.dot(tf_idf_matrix)
    response = list(zip(docs, doc_score))
    response = sorted(response, key=itemgetter(1), reverse=True)
    return response


def get_tf_idf_matrix(tf_matrix, N, all_n, doc_words):
    if os.path.exists('tf_idf_matrix.pickle'):
        with open("tf_idf_matrix.pickle", "rb") as m:
            tf_idf_matrix = pickle.load(m)
    else:
        tf_idf_matrix = create_tf_idf_matrix(tf_matrix, N, all_n, doc_words)
        with open("tf_idf_matrix.pickle", "wb") as m:
            pickle.dump(tf_idf_matrix, m)
    return tf_idf_matrix


#response = tf_idf_search('хочу купить подарок')
#print(response[:5])
