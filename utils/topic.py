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


# # generate ngram, currently n = 2
# def get_ngrams(text, n):
#     n_grams = ngrams(word_tokenize(text), n)
#     return [' '.join(grams) for grams in n_grams]


def topic_modeling(dataset):
    # read data
    titles = dataset['title']
    # abstracts = dataset['abstracts']
    # corpus of documents
    # each consisting of only title, abstract, or paper

    # String to Vectors
    # remove stopwords
    text_nostopwords = [
        ' '.join([word for word in title.lower().split() if word not in stopwords.words('english')])
        for title in titles]

    # # remove punctuation
    # text_nopunc = []
    # for text in text_nostopwords:
    #     text_nopunc.append(re.sub('\W+', ' ', text))

    # tokenize, generate noun phrase
    text_processed = []
    for text in text_nostopwords:
    	text_processed.append(np_extractor(text))

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in text_processed:
        for token in text:
            frequency[token] += 1
    text_freq = [[token for token in text if frequency[token] > 1] for text in text_processed]

    # create dictionary
    dictionary = corpora.Dictionary(text_freq)
    # dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
    # convert tokenized documents to vectors
    corpus = [dictionary.doc2bow(text) for text in text_freq]
    # corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) # store to disk, for later use
    # Transformation interface
    # step 1 -- initialize a model
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # LDA
    # building model
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10, alpha='auto', eval_every=5, iterations=100)
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
    
    title_topic = list(zip(titles, article_topic))
    
    # number of topics
    topic_list = []
    for i in range(len(article_topic)):
        topic_list.append(article_topic[i][0][0])
    topic_list = set(topic_list)
    
    
    filename1 = 'topic_results/lda_title_topics_10.csv'
    with open(filename1, 'w') as outcsv:
        writer = csv.writer(outcsv, lineterminator='\n')
        writer.writerow(['Topic_ID', 'Title'])
        for item in title_topic:
            writer.writerow([item[1][0][1], item[0]])
    
    topics_desc = lda.print_topics(num_topics=-1, num_words=5)
    filename2 = 'topic_results/lda_topics_desc_10.csv'
    with open(filename2, 'w') as outcsv:
        writer = csv.writer(outcsv, lineterminator='\n')
        writer.writerow(['Topic_Desc'])
        for item in topics_desc:
            writer.writerow([item])

    # # LSI
    # # building model
    # lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
    # corpus_lsi = lsi[corpus_tfidf]

    # # saving and printing
    # docs = []
    # for doc in corpus_lsi:
    #     docs.append(doc)

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

    # title_topic = list(zip(titles, article_topic))

    # filename1 = 'topic_results/lsi_title_topics.csv'
    # with open(filename1, 'w') as outcsv:
    #     writer = csv.writer(outcsv, lineterminator='\n')
    #     writer.writerow(['Topic_ID', 'Title'])
    #     for item in title_topic:
    #         writer.writerow([item[1][0][1], item[0]])

    # topics_desc = lsi.print_topics(num_topics=-1, num_words=5)
    # filename2 = 'topic_results/lsi_topics_desc.csv'
    # with open(filename2, 'w') as outcsv:
    #     writer = csv.writer(outcsv, lineterminator='\n')
    #     writer.writerow(['Topic_Desc'])
    #     for item in topics_desc:
    #         writer.writerow([item])

    # # HDP
    # # hdp transformation, train HDP model, tuning gamma and alpha needed
    # hdp = models.HdpModel(corpus_tfidf, id2word=dictionary, gamma=1, alpha=1)
    # # a trained HDP model can be used to transform the corpus into HDP topic distributions
    # corpus_hdp = hdp[corpus_tfidf]
    #
    # # article belongs to which topic
    # docs = []
    # for doc in corpus_hdp:
    #     docs.append(doc)
    #
    # # calculate largest score to decide which article belongs to which topic
    # topic_id = 0  # belongs to which topic
    # # topic_ids = []
    # article_topic = [[] * 2 for i in range(len(docs))]
    # for i in range(len(docs)):
    #     j = 0
    #     max_score = 0
    #     while j < len(docs[i]):
    #         if abs(docs[i][j][1]) > max_score:
    #             max_score = docs[i][j][1]
    #             topic_id = docs[i][j][0]
    #         # topic_ids.append(topic_id)
    #         j = j + 1
    #     article_topic[i].append((i, topic_id))
    #
    # # show topics with top 10 related words
    # # topics_desc = hdp.print_topics(topics=len(set(topic_ids)), topn=10)
    #
    # title_topic = list(zip(titles, article_topic))
    # title_topic = title_topic[1:]
    #
    # # write file
    # filename1 = 'topic_results/title_topics.csv'
    # with open(filename1, 'w') as outcsv:
    #     writer = csv.writer(outcsv, lineterminator='\n')
    #     writer.writerow(['Topic_ID', 'Title'])
    #     for item in title_topic:
    #         writer.writerow([item[1][0][1], item[0]])
    #
    # topics_desc = hdp.print_topics(num_topics=-1, num_words=10)
    # filename2 = 'topic_results/topics_desc.csv'
    # with open(filename2, 'w') as outcsv:
    #     writer = csv.writer(outcsv, lineterminator='\n')
    #     writer.writerow(['Topic_Desc'])
    #     for item in topics_desc:
    #         writer.writerow([item])

    # Evaluation
    # similarity score between mesh terms and topic keywords
    # calculate accuracy using Word2vec model
    # score = 0
    # for i in range(0,10):
    # 	meshterms = list()
    # 	topic_keywords = list()
    # 	for j in range(len(title_topic)):
    # 		if title_topic[j][1][0][1] == i:
    # 			meshterms.append(title_topic[j][2])
    # 			meshterms = [x for x in meshterms if str(x) != 'nan'
    # 	for k in range(0,10):
    # 		topic_keywords.append(lda.show_topic(i)[k][0])
    # 	model_wv = models.Word2Vec(meshterms, min_count=1, hs=1, negative=0)
    # 	score = score + numpy.mean(model_wv.score(topic_keywords))*len(meshterms)
    # 	accuracy = score/len(titles)


    # calculate accuracy using keyword occurrence
	score = 0
	for i in range(0,10):
	    score_topic = 0
	    meshterms = list()
	    newMeSH = list()
	    topic_keywords = list()
	    #get all keywords for this topic
	    for k in range(0,10):
	        topic_keywords.append(lda.show_topic(i)[k][0])
	    #get all mesh terms for this topic
	    for j in range(len(title_topic)):
	        if title_topic[j][1][0][1] == i:
	            meshterms.append(title_topic[j][2])
	            meshterms = [x for x in meshterms if str(x) != 'nan']
	        for i in range(len(meshterms)):
	            a = meshterms[i].split(";")
	            for j in range(len(a)):
	                newMeSH.append(a[j].replace("*", ""))
	    newMeSH = set(newMeSH)
	    #check keywords occurrences in mesh terms
	    for item in topic_keywords:
	        if item in newMeSH:
	            score_topic = score_topic + 1
	    score = score + len(meshterms)*score_topic/10
	accuracy = score/len(titles)
	accuracy

    return accuracy
