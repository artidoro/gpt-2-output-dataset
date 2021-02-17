import os
import json
import pickle
import time

import fire
import numpy as np
from scipy import sparse

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import features

def _load_split(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)['text'])
    return texts

def _load_split_grover(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    texts = []
    labels = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        elt = json.loads(line)
        texts.append(elt['text'])
        labels.append(elt['label'] == 'machine')
    return texts, labels

def load_split(data_dir, sources, split, n=np.inf):
    texts = []
    labels = []
    for source in sources.split(';'):
        if len(source) == 0:
            continue
        if not 'generator' in source:
            webtext = _load_split(data_dir, 'webtext', split, n=n//2)
            gen = _load_split(data_dir, source, split, n=n//2)
            t = webtext+gen
            l = [0]*len(webtext)+[1]*len(gen)
        else:
            t, l = _load_split_grover(data_dir, source, split, n)
        texts += t
        labels += l
    return texts, labels

def main(
        data_dir, 
        log_dir, 
        source='xl-1542M-k40', 
        n_train=500000, 
        n_valid=10000, 
        n_test=np.inf,
        n_jobs=None, 
        verbose=False,
        save_featureizer=False,
        save_model=False,
        save_features=False,
        load_featureizer=None,
        load_features=None,
        load_model=None,
        no_hyperparam_search=False,
        custom_features_only=False,
    ):
    start_time = time.time()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Loading data.
    train_texts, train_labels = load_split(data_dir, source, 'train', n=n_train)
    valid_texts, valid_labels = load_split(data_dir, source, 'valid', n=n_valid)
    test_texts, test_labels = load_split(data_dir, source, 'test', n=n_test)

    cur_time = time.time()
    print(f'{cur_time - start_time:.2f}\tFinished loading data.')
    start_time = cur_time

    # Extracting features.
    if not load_features:
        if not load_featureizer:
            if not custom_features_only:
                vect = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21)
            else:
                vect = features.CustomFeatures(n_jobs=n_jobs)
            train_features = vect.fit_transform(train_texts)
        else:
            with open(os.path.join(load_featureizer, 'featureizer.pickle'), 'rb') as infile:
                vect = pickle.load(infile)
                train_features = vect.transform(train_texts)
        valid_features = vect.transform(valid_texts)
        test_features = vect.transform(test_texts)
    else:
        with open(os.path.join(log_dir, 'train_features.pickle'), 'rb') as infile:
            train_features = pickle.load(infile)
        with open(os.path.join(log_dir, 'valid_features.pickle'), 'rb') as infile:
            valid_features = pickle.load(infile)
        with open(os.path.join(log_dir, 'test_features.pickle'), 'rb') as infile:
            test_features = pickle.load(infile)

    cur_time = time.time()
    print(f'{cur_time - start_time:.2f}\tFinished extracting features.')
    start_time = cur_time

    # Training the model.
    if not load_model:
        model = LogisticRegression(solver='liblinear')
        if not no_hyperparam_search:
            params = {'C': [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]}
            split = PredefinedSplit([-1]*n_train+[0]*n_valid)
            search = GridSearchCV(model, params, cv=split, n_jobs=n_jobs, verbose=verbose, refit=False)
            search.fit(sparse.vstack([train_features, valid_features]), train_labels+valid_labels)
            model = model.set_params(**search.best_params_)
            cur_time = time.time()
            print(f'{cur_time - start_time:.2f}\tFinished hyperparam search.')
            start_time = cur_time
        model.fit(train_features, train_labels)
    else:
        with open(os.path.join(load_model, 'model.pickle'), 'rb') as infile:
            model = pickle.load(infile)
    cur_time = time.time()
    print(f'{cur_time - start_time:.2f}\tFinished training model.')
    start_time = cur_time

    # Scoring the model.
    valid_accuracy = model.score(valid_features, valid_labels)*100.
    test_accuracy = model.score(test_features, test_labels)*100.
    data = {
        'source':source,
        'n_train':n_train,
        'valid_accuracy':valid_accuracy,
        'test_accuracy':test_accuracy
    }
    cur_time = time.time()
    print(f'{cur_time - start_time:.2f}\tFinished evaluating model.')
    start_time = cur_time
    print(data)
    json.dump(data, open(os.path.join(log_dir, f'stats.json'), 'w'))

    # Saving the model.
    if save_features:
        with open(os.path.join(log_dir, 'train_features.pickle'), 'wb') as outfile:
            pickle.dump(train_features, outfile)
        with open(os.path.join(log_dir, 'valid_features.pickle'), 'wb') as outfile:
            pickle.dump(valid_features, outfile)
        with open(os.path.join(log_dir, 'test_features.pickle'), 'wb') as outfile:
            pickle.dump(test_features, outfile)
        
        cur_time = time.time()
        print(f'{cur_time - start_time:.2f}\tFinished saving features.')
        start_time = cur_time
    if save_model:
        with open(os.path.join(log_dir, 'model.pickle'), 'wb') as outfile:
            pickle.dump(model, outfile)
        cur_time = time.time()
        print(f'{cur_time - start_time:.2f}\tFinished saving model.')
        start_time = cur_time
    if save_featureizer:
        with open(os.path.join(log_dir, 'featureizer.pickle'), 'wb') as outfile:
            pickle.dump(vect, outfile)
            cur_time = time.time()
            print(f'{cur_time - start_time:.2f}\tFinished saving featureizer.')
            start_time = cur_time

if __name__ == '__main__':
    fire.Fire(main)
