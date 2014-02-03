"""
http://radimrehurek.com/2014/02/word2vec-tutorial/#app

"""
import os
import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Word2Vec accepts a list of sentences (a sentence is a list of words)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                # do other preprocessing
                yield line.split()

sentences = MySentences('corpora')
model = gensim.models.Word2Vec(sentences, workers=4)

