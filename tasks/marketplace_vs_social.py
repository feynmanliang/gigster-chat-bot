#!/usr/bin/env python


import spacy
from nltk.corpus import stopwords


from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.extmath import density
from sklearn.externals import joblib

from time import time
from collections import Counter

import os.path
import string
import re
from pprint import pprint

from scripts.make_clean_dataset import GigInstance, make_clean_dataset, load_dataset

MODEL_OUTPATH = 'data/market_vs_social_pipeline.pkl'
DATASET_PATH = './data/clean_dataset.pkl'

MARKETPLACE = {
    'EBay',
    'Rideshare',
    'TaskRabbit',
    'InstaCart',
    'Lyft',
    'DoorDash',
    'appointments',
    'booking-flights',
    'booking-hotels',
    'crowdfunding',
    'ecommerce',
    'food-delivery',
    'job-search',
    'logistics',
    'marketplace',
    'marketplace-b2c',
    'marketplace-c2c',
    'marketplace-deals',
    'Uber',
    'Lyft',
    'marketplace-web',
    'real-estate-search',
    'realtime-marketplace',
    'reservation',
    'travel-booking',
    'shopping-community'}

SOCIAL = {
    'community',
    'community-anonymous',
    'realtime-social',
    'social',
    'social-clips',
    'social-dating',
    'social-discovery',
    'social-events',
    'social-lending',
    'social-media-management',
    'social-music',
    'social-news',
    'social-photos',
    'social-professional',
    'social-questions',
    'social-videos',
    'Instagram'}

def make_label(gigInstance):
    "Generates marketplace vs social template labels for a `GigInstance`."
    cnts = Counter()
    # NOTE: multiple templates exist for some gigs, by not deduping we asssume
    # when a template appears twice then it's twice as relevant
    for template in gigInstance.templates:
        if template in SOCIAL:
            cnts['social'] += 1
        elif template in MARKETPLACE:
            cnts['marketplate'] += 1

    # return the majority template, or None if empty or tied
    if len(cnts) == 0:
        return None
    else:
        best_count = cnts[max(cnts)]
        best_labels = [label for label,count in cnts.items() if count == best_count]
        if len(best_labels) == 1: # unique majority template type
            return best_labels[0]
        else: # tie
            return None

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))

# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]

# Every step in a pipeline needs to be a "transformer". 
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")

    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)

    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")

    # lowercase
    text = text.lower()

    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):
    parser = spacy.load('en')

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    # remove urls
    # urlFinder = re.compile(
    #     r".*app\.gigster\.com.*" + '|' +
    #     r".*google\.com.*",
    #     re.IGNORECASE)
    # tokens = filter(
    #     lambda tok: not urlFinder.match(tok),
    #     tokens)

    return tokens

def printNMostInformative(vectorizer, clf, N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)

def prepare_train_test():
    dataset = load_dataset()

    # data
    X = []
    y = []
    for gigInstance in dataset:
        label = make_label(gigInstance)
        if label:
            X.append('\n'.join(map(lambda m: m.text, gigInstance.messages)))
            y.append(label)
    return train_test_split(X, y, test_size=0.1, random_state=42)

def make_preprocessing_pipeline():
    # the vectorizer and classifer to use, the tokenizer in CountVectorizer uses a custom function (spaCy's tokenizer)
    cleaner = CleanTextTransformer()
    tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5)

    # the pipeline to clean, tokenize, vectorize, and classify
    pipe = Pipeline([
        ('cleanText', cleaner),
        ('tfidf', tfidf)])
    return pipe

# TODO: actually run this and tune the classifier
def do_grid_search(pipe):
    "Grid searches pipeline parameters."
    from sklearn.grid_search import GridSearchCV
    parameters = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False)
    }
    gs_pipe = GridSearchCV(pipe, parameters, n_jobs=-1).fit(train, labelsTrain)
    best_parameters, score, _ = max(gs_pipe.grid_scores_, key=lambda x: x[1])
    print("Score: {}".format(score))
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

# this method shows L-SVM with L1 Var Selection does best (78% acc)
def evaluate_classifiers():
    train, test, labelsTrain, labelsTest = prepare_train_test()
    pipe = make_preprocessing_pipeline()
    pipe.fit(train, labelsTrain)

    def benchmark(clf):
        "Benchmarks an algorithm."
        print('_' * 10)
        print(clf)
        t0 = time()
        clf.fit(pipe.transform(train), labelsTrain)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(pipe.transform(test))
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = accuracy_score(labelsTest, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time

    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        print('=' * 10)
        print(name)
        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        print('=' * 10)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                    dual=False, tol=1e-3)))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty)))

    # Train SGD with Elastic Net penalty
    print('=' * 10)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet")))

    # Train NearestCentroid without threshold
    print('=' * 10)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid()))

    # Train sparse Naive Bayes classifiers
    print('=' * 10)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01)))
    results.append(benchmark(BernoulliNB(alpha=.01)))

    print('=' * 10)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
      ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
      ('classification', LinearSVC())
    ])))

def make_classification_pipeline():
    steps = make_preprocessing_pipeline().steps + [
        ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
        ('classification', LinearSVC())
    ]
    return Pipeline(steps)

def evaluate(pipe, test, labelsTest):
    preds = pipe.predict(test)
    # print("----------------------------------------------------------------------------------------------")
    # print("results:")
    # for (sample, pred, label) in zip(test, preds, labelsTest):
    #     print(sample)
    #     print('--')
    #     print('prediction: {}, label: {}'.format(pred, label))
    #     print('--')

    print("----------------------------------------------------------------------------------------------")
    print("accuracy:", accuracy_score(labelsTest, preds))
    print(classification_report(labelsTest, preds))
    print(confusion_matrix(labelsTest, preds))


    print("----------------------------------------------------------------------------------------------")
    print("Top 10 features used to predict: ")
    # show the top features
    vectorizer = pipe.steps[1][1]
    clf = pipe.steps[2][1]
    printNMostInformative(vectorizer, clf, 10)

if __name__ == '__main__':
    train, test, labelsTrain, labelsTest = prepare_train_test()
    if os.path.exists(MODEL_OUTPATH):
        print("Loading pipeline from: {}".format(MODEL_OUTPATH))
        pipe = joblib.load(MODEL_OUTPATH)
    else:
        print("Training pipeline")
        pipe = make_classification_pipeline()
        pipe.fit(train, labelsTrain)

    # test
    evaluate(pipe, test, labelsTest)

    # save
    joblib.dump(pipe, MODEL_OUTPATH)
