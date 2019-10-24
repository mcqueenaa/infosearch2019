import tensorflow
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
import os
import csv
from operator import itemgetter
from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings

elmo_path = 'elmo'

batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(elmo_path)


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

if not os.path.exists('elmo_corpus.pickle'):
    corpus = []
    for idx, sent in enumerate(docs):
        if idx < 1001:
            corpus.append(tokenize(sent))
        else:
            break
        with open("elmo_corpus.pickle", "wb") as c:
            pickle.dump(corpus, c)
else:
    with open("elmo_corpus.pickle", "rb") as c:
        corpus = pickle.load(c)


def get_vect(vect, sent):
    vector = vect[:len(sent), :]
    vector = np.mean(vector, axis=0)
    return vector


def create_elmo_matrix(corpus, batcher, sentence_character_ids,
                       elmo_sentence_input):
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        matrix = []

        for i in range(200, len(corpus) + 1, 200):
            sentences = corpus[i - 200: i]
            elmo_vectors = get_elmo_vectors(sess, sentences, batcher,
                                            sentence_character_ids,
                                            elmo_sentence_input)

            for vect, sent in zip(elmo_vectors, sentences):
                vector = get_vect(vect, sent)
                matrix.append(vector)
    return matrix


def get_elmo_matrix(corpus):
    if os.path.exists('elmo_matrix.pickle'):
        with open("elmo_matrix.pickle", "rb") as m:
            elmo_matrix = pickle.load(m)
    else:
        elmo_matrix = create_elmo_matrix(corpus, batcher,
                                         sentence_character_ids,
                                         elmo_sentence_input)
        with open("elmo_matrix.pickle", "wb") as m:
            pickle.dump(elmo_matrix, m)
    return elmo_matrix


def elmo_query2vec(query, batcher, sentence_character_ids,
                   elmo_sentence_input):
    query = tokenize(query)
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        vector = get_vect(get_elmo_vectors(sess, query, batcher,
                                           sentence_character_ids,
                                           elmo_sentence_input)[0], query[0])
    return vector


def get_cos_sim(v1, v2):
    cos_sim = dot(v1, v2) / (norm(v1) * norm(v2))
    return cos_sim


def elmo_search(query, batcher, sentence_character_ids, elmo_sentence_input,
                docs):
    q_vec = elmo_query2vec(query, batcher, sentence_character_ids,
                           elmo_sentence_input)
    response = []
    elmo_matrix = get_elmo_matrix(corpus)

    for idx, doc_vec in enumerate(elmo_matrix):
        doc_score = get_cos_sim(q_vec, doc_vec)
        response.append((docs[idx], doc_score))
    response = sorted(response, key=itemgetter(1), reverse=True)
    return response


#response = elmo_search('разница между Китаем и Соединенными Штатами', batcher,
                       #sentence_character_ids, elmo_sentence_input, docs)
