""" Corpus Loader """

from sklearn.model_selection import KFold
import numpy as np

class CorpusLoader(object):
    # Initialize class variables
    def __init__(self, reader, folds=None, categories=None, shuffle=True):
        self.reader=reader
        self.folds=KFold(n_splits=folds,shuffle=shuffle)
        self.files=np.asarray(self.reader.fileids(categories=categories))

    # Method to access a listing of fileids by foldID.
    def fileids(self, idx=None):
        if idx is None:
            return self.files
        return self.files[idx]

    # Returns a genetator with one document at a time
    def documents(self,idx=None):
        for fileid in self.fileids(idx):
            yield list(self.reader.docs(fileids=[fileid]))

    # To look up label from the corpus & returns a label for each document.
    def labels(self, idx=None):
        return [
        self.reader.categories(fileids=[fileid])[0]
        for fileid in self.fileids(idx)
        ]

    # Iterator to split training & test data for each fold using KFold's Split()
    def __iter__(self):
        for train_index, test_index in self.folds.split(self.files):
            X_train = self.documents(train_index)
            y_train = self.labels(train_index)

            X_test = self.documents(test_index)
            y_test = self.labels(test_index)

            yield X_train, X_test, y_train, y_test

if __name__ == '__main__':
    from PickledCorpusReader import PickledCorpusReader as reader

    corpus = reader('../corpus')
    loader=CorpusLoader(corpus,folds=5)
