"""

"""
__author__ = "gavin"
import logging
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


documents = [
    'Human machine interface for lab abc computer applications',
    'A survey of user opinion of computer system response time',
    'The EPS user interface management system',
    'System and human system engineering testing of EPS',
    'Relation of user perceived response time to error measurement',
    'The generation of random binary unordered trees',
    'The intersection graph of paths in trees',
    'Graph minors IV Widths of trees and well quasi ordering',
    'Graph minors A survey'
]

# tokenize and stop filter
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.save('dict/deerwester.dict')

print dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('bow/deerwester.mm', corpus)


class MyCorpus(object):
    def __iter__(self):
        for line in open('corpora/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()

dictionary = corpora.Dictionary(line.lower().split() for line in open('corpora/mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
dictionary.compactify()


## LSI
dictionary = corpora.Dictionary.load('dict/deerwester.dict')
corpus = corpora.MmCorpus('bow/deerwester.mm')
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print vec_lsi

index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi] # perform a similarity query against the corpus
print list(enumerate(sims)) # print (document_number, document_similarity) 2-tuples