#!/usr/bin/env python

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import string
import re
from pprint import pprint

from scripts.make_clean_dataset import GigInstance, make_clean_dataset, load_dataset

parser = spacy.load('en')

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
    from collections import Counter
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


if __name__ == '__main__':
    dataset = load_dataset()

    # the vectorizer and classifer to use, the tokenizer in CountVectorizer uses a custom function (spaCy's tokenizer)
    cleaner = CleanTextTransformer()
    vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
    tfidf = TfidfTransformer(use_idf=True)
    clf = LinearSVC(loss='hinge', penalty='l2', random_state=42)

    # the pipeline to clean, tokenize, vectorize, and classify
    pipe = Pipeline([
        ('cleanText', cleaner),
        ('vectorizer', vectorizer),
        ('tfidf', tfidf),
        ('clf', clf)])

    # data
    X = []
    y = []
    for gigInstance in dataset:
        label = make_label(gigInstance)
        if label:
            X.append('\n'.join(map(lambda m: m.text, gigInstance.messages)))
            y.append(label)
    train, test, labelsTrain, labelsTest = train_test_split(X, y, test_size=0.1, random_state=42)

    # train
    pipe.fit(train, labelsTrain)

    # test
    preds = pipe.predict(test)
    print("----------------------------------------------------------------------------------------------")
    print("results:")
    for (sample, pred, label) in zip(test, preds, labelsTest):
        print(sample)
        print('--')
        print('prediction: {}, label: {}'.format(pred, label))
        print('--')

    print("----------------------------------------------------------------------------------------------")
    print("accuracy:", accuracy_score(labelsTest, preds))
    print(classification_report(labelsTest, preds))
    print(confusion_matrix(labelsTest, preds))


    print("----------------------------------------------------------------------------------------------")
    print("Top 10 features used to predict: ")
    # show the top features
    printNMostInformative(vectorizer, clf, 10)
