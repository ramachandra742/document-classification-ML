""" To Read pickle files from Corpus """

import nltk
import pickle

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

doc_pattern = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
pkl_pattern = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
cat_pattern = r'([a-z_\s]+)/.*'

class PickledCorpusReader(CategorizedCorpusReader,CorpusReader):

    def __init__(self, root, fileids=pkl_pattern, **kwargs):
        """
        Initialize the corpus reader. Categorized arguments
        ('cat_pattern', 'cat_map', and 'cat_file') are passed
        to the  'CategorizedCorpusReader' constructor. The remaining
        arguments are passed to the CorpusReader constructor.
        """
        # Add the default category pattern if not passed into the class
        if not any (key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = cat_pattern

        # Initialize NLP Corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function.
        """
        if fileids is not None and categories is not None:
            raise ValueError ("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the document from a pickled object for each file in corpus.
        """
        #List the fileids & categories
        fileids = self.resolve(fileids, categories)
        # Load one document into memory at a time
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path,'rb') as f:
                yield pickle.load(f)

    def paragraphs(self, fileids=None, categories=None):
        """
        Returns a genetator where each paragraph contains a list of sentences.
        """
        for doc in self.docs(fileids, categories):
            for paragraph in doc:
                yield paragraph

    def sentences(self, fileids=None, categories=None):
        """
        Returns a generator where each sentence contains a list of tokens
        """
        for paragraph in self.paragraphs(fileids, categories):
            for sent in paragraph:
                yield sent

    def tokens(self, fileids=None, categories=None):
        """
        Returns a list of tokens.
        """
        for sent in self.sentences(fileids,categories):
            for token in sent:
                yield token

    def words(self, fileids=None, categories=None):
        """
        Returns a list of (token, tag) tuples.
        """
        for token in self.tokens(fileids, categories):
            yield token[0]

if __name__ == '__main__':
    from collections import Counter

    corpus = PickledCorpusReader('../corpus')
    words = Counter(corpus.words())

    print("{:,} vocabulary {:,} word count".format(len(words.keys()),sum(words.values())))
