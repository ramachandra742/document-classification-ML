""" ML models implementation """
import numpy as np
import nltk
import time
import json
import unicodedata
from CorpusLoader import CorpusLoader
from PickledCorpusReader import PickledCorpusReader
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')

def identity(words):
    return words

class TextNormalizer(BaseEstimator,TransformerMixin):

    def __init__(self, language ='english'):
        self.stopwords = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        """
        Check the punctuation character.
        """
        return all (
        unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self,token):
        return token.lower() in self.stopwords

    def doc_normalize(self, doc):
        """
        Removes stopwords and punctuation, lowercases, lemmatizes
        """
        return [
        self.lemmatize(token,tag).lower()
        for paragraph in doc
        for sent in paragraph
        for (token,tag) in sent
        if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        """
        Return the WordNet POS tag from the Penn Treebank tag
        """
        tag = {
        'N' : wn.NOUN,
        'V' : wn.VERB,
        'R' : wn.ADV,
        'J' : wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for doc in documents:
            yield self.doc_normalize(doc[0])

# Create pipeline for all models.
def pipeline(estimator, reduction=False):
    steps=[
    ('normalize',TextNormalizer()),
    ('vectorize',TfidfVectorizer(
    tokenizer = identity, preprocessor=None, lowercase=False
    ))
    ]

    if reduction:
        steps.append((
        'reduction',TruncatedSVD(n_components=1000)
        ))
    # Add the estimator
    steps.append(('classifier',estimator))
    return Pipeline(steps)

labels = ['books','cinema','cooking','gaming','sports','tech','data_science',\
            'design','news','politics','do_it_yourself','business']
reader = PickledCorpusReader('../corpus')
loader = CorpusLoader(reader, 5, shuffle=True, categories=labels)

models = []
names = [LogisticRegression, SGDClassifier]
for model in names:
    models.append(pipeline(model(),True))
    models.append(pipeline(model(),False))
models.append(pipeline(MultinomialNB(),False))
models.append(pipeline(GaussianNB(), True))

def model_scores(models,loader):
    for model in models:
        name = model.named_steps['classifier'].__class__.__name__
        if 'reduction' in model.named_steps:
            name +=" (TruncatedSVD)"

        scores = {
        'model':str(model),
        'name':name,
        'accuracy':[],
        'precision':[],
        'recall':[],
        'f1':[],
        'time':[]
        }

        for X_train, X_test, y_train, y_test in loader:
            start = time.time()
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            scores['time'].append(time.time() - start)
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred,average='weighted'))
            scores['recall'].append(recall_score(y_test,y_pred,average='weighted'))
            scores['f1'].append(f1_score(y_test,y_pred,average='weighted'))

        scores['accuracy']=np.mean(scores['accuracy'])
        scores['precision']=np.mean(scores['precision'])
        scores['recall']=np.mean(scores['recall'])
        scores['f1']=np.mean(scores['f1'])
        yield scores

if __name__ == '__main__':
    for scores in model_scores(models, loader):
        with open('results.json','a') as f:
            f.write(json.dumps(scores)+'\n')
