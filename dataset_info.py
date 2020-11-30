""" Dataset stats """
from PickledCorpusReader import PickledCorpusReader

reader = PickledCorpusReader('../corpus')

for category in reader.categories():

    n_docs = len(reader.fileids(categories=[category]))
    n_words = sum(1 for word in reader.words(categories=[category]))

    print(f'{category} contains {n_docs} documents and {n_words} words')
