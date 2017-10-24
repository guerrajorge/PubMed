import csv
import gensim
import nltk
from gensim import corpora, models
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# generate ngram, currently n = 2
def get_ngrams(text, n):
    n_grams = ngrams(word_tokenize(text), n)
    return [' '.join(grams) for grams in n_grams]


def topic_modeling(dataset):
    # read data
    titles = dataset['title']
    # corpus of documents
    # each consisting of only title, abstract, or paper

    # String to Vectors
    # remove common words and tokenize
    # stoplist = set('a about above after again all am among an and any are as at be because been before being below between both but by can do did does doing down during each few for from had has have he her here hers him herself himself how i if in into is its itself me much more most my myself no nor not of off on once only or other our ours out over own same she should so some such than that the their they theirs them then there hey this these those through to too under until up very was we were what when where which while who why with would you your yours yourself just via it vs.'.split())
    # stop = stopwords.words('english') + string.punctuation
    # texts = [[word for word in title.lower().split() if word not in stoplist] for title in titles]
    text_nostopwords = [
        ' '.join([word for word in title.lower().split() if word not in stopwords.words('english') and len(word) > 2])
        for title in titles]

    # generate ngram, currently n = 2
    text_ngram = []
    for text in text_nostopwords:
        text_ngram.append(get_ngrams(text, 2))

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in text_ngram:
        for token in text:
            frequency[token] += 1
    text_freq = [[token for token in text if frequency[token] > 1] for text in text_ngram]

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

    # hdp
    # hdp transformation, train HDP model, tuning gamma and alpha needed
    hdp = models.HdpModel(corpus_tfidf, id2word=dictionary, gamma=1, alpha=1)
    # ordering the topics
    hdp.optimal_ordering()
    # show topics with top 10 most probable words
    topics_desc = hdp.print_topics(num_topics=-1, num_words=10)
    # hdp.update(corpus_tfidf)
    # create new corpus over the original corpus
    # a trained HDP model can be used to transform the corpus into HDP topic distributions
    corpus_hdp = hdp[corpus_tfidf]
    # hdp.update(corpus_hdp)
    # hdp.evaluate_test_corpus(corpus_hdp)
    # article belongs to which topic
    docs = []
    for doc in corpus_hdp:
        docs.append(doc)

    # calculate largest score to decide which article belongs to which topic
    topic_id = 0  # belongs to which topic
    # topic_ids = []
    article_topic = [[] * 2 for i in range(len(docs))]
    for i in range(len(docs)):
        j = 0
        max_score = 0
        while j < len(docs[i]):
            if docs[i][j][1] > max_score:
                max_score = docs[i][j][1]
                topic_id = docs[i][j][0]
            # topic_ids.append(topic_id)
            j = j + 1
        article_topic[i].append((i, topic_id))

    # show topics with top 10 related words
    # topics_desc = hdp.print_topics(topics=len(set(topic_ids)), topn=10)

    title_topic = list(zip(titles, article_topic))
    title_topic = title_topic[1:]

    # write file
    filename1 = 'topic_results/title_topics.csv'
    with open(filename1, 'w') as outcsv:
        writer = csv.writer(outcsv, lineterminator='\n')
        writer.writerow(['Topic_ID', 'Title'])
        for item in title_topic:
            writer.writerow([item[1][0][1], item[0]])

    filename2 = 'topic_results/topics_desc.csv'
    with open(filename2, 'w') as outcsv:
        writer = csv.writer(outcsv, lineterminator='\n')
        writer.writerow(['Topic_Desc'])
        for item in topics_desc:
            writer.writerow([item])

    # filename3 = 'topic_score_' + str(year) + '.csv'
    # with open(filename3, 'w') as outcsv:
    #     writer = csv.writer(outcsv, lineterminator = '\n')
    #     writer.writerow(['Topic_Desc'])
    #     for item in docs:
    #         writer.writerow([item])
    return len(topics_desc)
