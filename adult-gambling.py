"""

train LDA model with 1 or more topics per leaf category
add topic probabilities to unigram features

Taboo;
train LDA model on taboo corpus with 1 or more topics per taboo category
add topic probabilities to TFIDF/etc. features for classifier


get list of current publishers
create google cse to retrieve documents for taboo term queries from only those domains

"""
__author__ = 'gavin'
import logging
from gensim import corpora, models, similarities

stoplist = set('for a of the and to in'.split())

raw_corpus = [l.lower().strip() for l in open('corpora/adult-gambling.corpus')]

dictionary = corpora.Dictionary(line.lower().split() for line in open('corpora/adult-gambling.corpus'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
dictionary.compactify()


class MyCorpus(object):
    def __iter__(self):
        for line in open('corpora/adult-gambling.corpus'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus = [c for c in MyCorpus()]

import operator

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
doc = 'free online blackjack and poker virtual casino'
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
#print vec_lsi

index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi] # perform a similarity query against the corpus

s = list(enumerate(sims))
print s.sort(key=operator.itemgetter(1), reverse=True)

for i in s:# print (document_number, document_similarity) 2-tuples
    print i, raw_corpus[i[0]][:80]