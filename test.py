import pandas as pd
import csv
import re
import gensim
import nltk
from gensim import corpora, models
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from utils.np_extractor import np_extractor
import numpy
import math


# def get_ngrams(text, n):
#     n_grams = ngrams(word_tokenize(text), n)
#     return [' '.join(grams) for grams in n_grams]

# read data
dataset = pd.read_csv('record_results/titles_abstracts_mesh.csv')
# titles = dataset['title']
abstracts = dataset['abstracts']
mesh = dataset['MeSH']

# remove stopwords
# text_nostopwords = [
#         ' '.join([word for word in title.lower().split() if word not in stopwords.words('english')])
#         for title in titles]
text_nostopwords = [
        ' '.join([nltk.wordnet.WordNetLemmatizer().lemmatize(word) for word in abstract.lower().split() if word not in stopwords.words('english')])
        for abstract in abstracts if str(abstract) != 'nan']
#
# # remove punctuation
# text_nopunc = []
# for text in text_nostopwords:
#     text_nopunc.append(re.sub('\W+', ' ', text))
#
# # tokenize, generate ngram, currently n = 2
# text_processed = []
# for text in text_nopunc:
#     text_processed.append(np_extractor(text))


# tokenize, generate ngram, currently n = 2
text_processed = []
for text in text_nostopwords:
    text_processed.append(np_extractor(text))


# building dict
from collections import defaultdict
frequency = defaultdict(int)
for text in text_processed:
    for token in text:
        frequency[token] += 1
# # frequency.pop('don t')
# with open('dict.csv', 'w') as outcsv:
#     writer = csv.writer(outcsv, lineterminator='\n')
#     for key, value in frequency.items():
#         writer.writerow([key, value])

# remove top freq terms
dataset1 = pd.read_csv('freq_list.csv')
freqTerms = dataset1['FreqTerms']
for term in freqTerms:
    frequency.pop(term, None)

text_freq = [[token for token in text if frequency[token] > 3] for text in text_processed]
dictionary = corpora.Dictionary(text_freq)
corpus = [dictionary.doc2bow(text) for text in text_freq]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]



# # building model
# hdp = models.HdpModel(corpus_tfidf, id2word=dictionary, gamma=1, alpha = 1)
# #hdp.optimal_ordering()
# corpus_hdp = hdp[corpus_tfidf]
#
# # tuning gamma and alpha
# g = 1
# a = 1
# while g <= 10:
#     if a <= 10:
#         hdp = models.HdpModel(corpus_tfidf, id2word=dictionary, gamma=g, alpha=a)
#         corpus_hdp = hdp[corpus_tfidf]
#         docs = []
#         for doc in corpus_hdp:
#             docs.append(doc)
#         topic_id = 0  # belongs to which topic
#         article_topic = [[] * 2 for i in range(len(docs))]
#         for i in range(len(docs)):
#             j = 0
#             max_score = 0
#             while j < len(docs[i]):
#                 if docs[i][j][1] > max_score:
#                     max_score = docs[i][j][1]
#                     topic_id = docs[i][j][0]
#                 j = j + 1
#             article_topic[i].append((i, topic_id))
#         topic_list = []
#         for i in range(len(article_topic)):
#             topic_list.append(article_topic[i][0][1])
#         topic_list = set(topic_list)
#         print(g, a, len(topic_list))
#         a = a + 0.5
#     else:
#         g = g + 0.5
#         a = 1

# # tuning gamma and alpha
# g = 1
# a = 1
# while g > 0:
#     if a > 0:
#         hdp = models.HdpModel(corpus_tfidf, id2word=dictionary, gamma=g, alpha=a)
#         corpus_hdp = hdp[corpus_tfidf]
#         docs = []
#         for doc in corpus_hdp:
#             docs.append(doc)
#         topic_id = 0  # belongs to which topic
#         article_topic = [[] * 2 for i in range(len(docs))]
#         for i in range(len(docs)):
#             j = 0
#             max_score = 0
#             while j < len(docs[i]):
#                 if abs(docs[i][j][1]) > max_score:
#                     max_score = docs[i][j][1]
#                     topic_id = docs[i][j][0]
#                 j = j + 1
#             article_topic[i].append((i, topic_id))
#         topic_list = []
#         for i in range(len(article_topic)):
#             topic_list.append(article_topic[i][0][1])
#         topic_list = set(topic_list)
#         print(g, a, len(topic_list))
#         a = a - 0.1
#     else:
#         g = g - 0.1
#         a = 1


# # saving and printing
# docs = []
# for doc in corpus_hdp:
#     docs.append(doc)
#
# topic_id = 0  # belongs to which topic
# article_topic = [[] * 2 for i in range(len(docs))]
# for i in range(len(docs)):
#     j = 0
#     max_score = 0
#     while j < len(docs[i]):
#         if abs(docs[i][j][1]) > max_score:
#             max_score = docs[i][j][1]
#             topic_id = docs[i][j][0]
#         j = j + 1
#     article_topic[i].append((i, topic_id))
#
# title_topic = list(zip(titles, article_topic))
# #title_topic = title_topic[1:]
#
# # number of topics
# topic_list = []
# for i in range(len(article_topic)):
#     topic_list.append(article_topic[i][0][0])
# topic_list = set(topic_list)


# filename1 = 'topic_results/hdp_title_topics.csv'
# with open(filename1, 'w') as outcsv:
#     writer = csv.writer(outcsv, lineterminator='\n')
#     writer.writerow(['Topic_ID', 'Title'])
#     for item in title_topic:
#         writer.writerow([item[1][0][1], item[0]])
#
# topics_desc = hdp.print_topics(num_topics=-1, num_words=10)
# filename2 = 'topic_results/hdp_topics_desc.csv'
# with open(filename2, 'w') as outcsv:
#     writer = csv.writer(outcsv, lineterminator='\n')
#     writer.writerow(['Topic_Desc'])
#     for item in topics_desc:
#         writer.writerow([item])



# # LSI
# # building model
# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
# corpus_lsi = lsi[corpus_tfidf]
#
# #saving and printing
# docs = []
# for doc in corpus_lsi:
#     docs.append(doc)
#
# topic_id = 0  # belongs to which topic
# article_topic = [[] * 2 for i in range(len(docs))]
# for i in range(len(docs)):
#     j = 0
#     max_score = 0
#     while j < len(docs[i]):
#         if abs(docs[i][j][1]) > max_score:
#             max_score = docs[i][j][1]
#             topic_id = docs[i][j][0]
#         j = j + 1
#     article_topic[i].append((i, topic_id))
#
# title_topic = list(zip(titles, article_topic))
#
# # number of topics
# topic_list = []
# for i in range(len(article_topic)):
#     topic_list.append(article_topic[i][0][0])
# topic_list = set(topic_list)
#
#
# filename1 = 'topic_results/lsi_title_topics_10.csv'
# with open(filename1, 'w') as outcsv:
#     writer = csv.writer(outcsv, lineterminator='\n')
#     writer.writerow(['Topic_ID', 'Title'])
#     for item in title_topic:
#         writer.writerow([item[1][0][1], item[0]])
#
# topics_desc = lsi.print_topics(num_topics=-1, num_words=5)
# filename2 = 'topic_results/lsi_topics_desc_10.csv'
# with open(filename2, 'w') as outcsv:
#     writer = csv.writer(outcsv, lineterminator='\n')
#     writer.writerow(['Topic_Desc'])
#     for item in topics_desc:
#         writer.writerow([item])


# LDA
# building model
num_topics = 29
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.
lda = models.LdaModel(corpus=corpus_tfidf, id2word=dictionary, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)
corpus_lda = lda[corpus_tfidf]

#saving and printing
docs = []
for doc in corpus_lda:
    docs.append(doc)

topic_id = 0  # belongs to which topic
article_topic = [[] * 2 for i in range(len(docs))]
for i in range(len(docs)):
    j = 0
    max_score = 0
    while j < len(docs[i]):
        if abs(docs[i][j][1]) > max_score:
            max_score = docs[i][j][1]
            topic_id = docs[i][j][0]
        j = j + 1
    article_topic[i].append((i, topic_id))

title_topic = list(zip(abstracts, article_topic, mesh))

# number of topics
topic_list = []
for i in range(len(article_topic)):
    topic_list.append(article_topic[i][0][0])
topic_list = set(topic_list)


filename1 = 'topic_results/lda_abstract_topics_29_removefreqterms.csv'
with open(filename1, 'w') as outcsv:
    writer = csv.writer(outcsv, lineterminator='\n')
    writer.writerow(['Topic_ID', 'Title'])
    for item in title_topic:
        writer.writerow([item[1][0][1], item[0]])

topics_desc = lda.print_topics(num_topics=-1, num_words=5)
filename2 = 'topic_results/lda_abstract_desc_29_removefreqterms.csv'
with open(filename2, 'w') as outcsv:
    writer = csv.writer(outcsv, lineterminator='\n')
    writer.writerow(['Topic_Desc'])
    for item in topics_desc:
        writer.writerow([item])


# # calculate text similarity
# # use MeSH terms to build a model
# # test mesh terms
# # sample MeSH terms
# # "Aged, Cardiovascular Diseaseschemically induced, Estrogen Replacement Therapyadverse effects, Evidence-Based Medicine, Female, Humans"
# meshterms = [['Aged'], ['Cardiovascular Diseaseschemically induced'], ['Estrogen Replacement Therapyadverse effects'], ['Evidence-Based Medicine'], ['Female'], ['Humans']]
# # testing purpose
# # meshterms3 = [['Aged'], ['Cardiovascular Diseaseschemically induced'], ['Estrogen Replacement Therapyadverse effects'], ['Evidence-Based Medicine'], ['Female'], ['Humans'], ['acute respiratory distress syndrome'], ['case report'], ['critical care'], ['cardiac catheterization'], ['cryptococcal meningitis']]
# model_wv = models.Word2Vec(meshterms, min_count =1, hs = 1, negative = 0)
# # calculate similarity of top 5 keywords from each topic
# # model_wv.score(['acute respiratory distress syndrome', 'case report', 'critical care', 'cardiac catheterization', 'cryptococcal meningitis'])
# model_wv.score([['acute respiratory distress syndrome'], ['case report'], ['critical care'], ['cardiac catheterization'], ['cryptococcal meningitis']])
# model_wv.similarity('Female', 'Humans')
# model_wv['Female']

# # loop calculate accuracy
# # word2vec model
# score = 0
# for i in range(0,10):
#     meshterms = list()
#     topic_keywords = list()
#     for j in range(len(title_topic)):
#         if title_topic[j][1][0][1] == i:
#             meshterms.append(title_topic[j][2])
#             meshterms = [x for x in meshterms if str(x) != 'nan']
#     for k in range(0,10):
#         topic_keywords.append(lda.show_topic(i)[k][0])
#     model_wv = models.Word2Vec(meshterms, min_count=1, hs=1, negative=0)
#     score = score + numpy.mean(model_wv.score(topic_keywords))*len(meshterms)
# accuracy = score/len(titles)
# accuracy

# loop calculate accuracy
# occurrence
score = 0
for i in range(0, num_topics):
    score_topic = 0
    meshterms = list()
    newMeSH = list()
    topic_keywords = list()
    #get all keywords for this topic
    for k in range(0, 10):
        topic_keywords.append(lda.show_topic(i)[k][0])
    #get all mesh terms for this topic
    for j in range(len(title_topic)):
        if title_topic[j][1][0][1] == i:
            meshterms.append(title_topic[j][2])
            meshterms = [x for x in meshterms if str(x) != 'nan']
        for i in range(len(meshterms)):
            a = meshterms[i].split(";")
            for j in range(len(a)):
                newMeSH.append(a[j].replace("*", "").replace("/", " "))
    newMeSH = set(newMeSH)
    #check keywords occurrences in mesh terms
    for item in topic_keywords:
        if item in newMeSH:
            score_topic = score_topic + 1
    score = score + len(meshterms)*score_topic/num_topics
accuracy = score/len(abstracts)
print("the accuracy score is ", accuracy)


# # get all meshterms in one topic out into a new list
# # also, remove *
# newMeSH = list()
# for i in range(len(meshterms)):
#     a = meshterms[i].split(";")
#     for j in range(len(a)):
#         newMeSH.append(a[j].replace("*", ""))
# newMeSh = set(newMeSH)
# len(newMeSH)

# # Training LDA
# # Set training parameters.
# num_topics = 5
# chunksize = 2000
# passes = 20
# iterations = 400
# eval_every = None  # Don't evaluate model perplexity, takes too much time.

# # calculating topic coherence
# while num_topics < 30:
#     #build LDA
#     model = models.LdaModel(corpus=corpus_tfidf, id2word=dictionary, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations,
#                             num_topics=num_topics, passes=passes, eval_every=eval_every)
#     top_topics = model.top_topics(corpus_tfidf)
#     # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
#     avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
#     print(num_topics, avg_topic_coherence)
#     num_topics = num_topics + 1


# # calculating perplexity using model.bound()
# while num_topics < 30:
#     #build LDA
#     model = models.LdaModel(corpus=corpus_tfidf, id2word=dictionary, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations,
#                             num_topics=num_topics, passes=passes, eval_every=eval_every)
#     perplexity_score = model.bound(corpus=corpus_tfidf)
#     print(num_topics, perplexity_score)
#     num_topics = num_topics + 1

# # calculating perplexity using model.log_perplexity()
# while num_topics < 30:
#     #build LDA
#     model = models.LdaModel(corpus=corpus_tfidf, id2word=dictionary, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations,
#                             num_topics=num_topics, passes=passes, eval_every=eval_every)
#     perplex = model.log_perplexity(corpus_tfidf, total_docs=len(corpus_tfidf))
#     print(num_topics, math.pow(2, perplex))
#     num_topics = num_topics + 1