from collections import defaultdict, Counter
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
import spacy

nlp = spacy.load("en_core_web_sm")

def walk_tree(node, depth):
    if depth >= 100:
        print(f'Warning: max depth reached with node: {node}')
        return 100
    elif node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth

def list_encode(name, l, args):
    stats = dict()
    if args['encode_min']:
        stats[name+'_min'] = np.min(l)
    if args['encode_max']:
        stats[name+'_max'] = np.max(l)
    if args['encode_std']:
        stats[name+'_std'] = np.std(l)
    if args['encode_avg']:
        stats[name+'_avg'] = np.average(l)
    if args['encode_full']:
        d = Counter(l)
        for k,v in d.items():
            stats[f'{name}_{k}'] = v
    return stats

def parse_data(text, args):
    dict_features = []
    doc = nlp(text)
    sents = list(doc.sents)
    doc_num_sents = len(sents)
    doc_num_tokens = len(list(doc))

    word_lens = defaultdict(int)
    sent_lens = defaultdict(int)
    parse_tree_depths = defaultdict(int)
    for sent in sents:
        tokens = list(sent)
        sent_lens[len(tokens)] += 1

        depth = walk_tree(sent.root, 0)
        parse_tree_depths[depth] += 1

        for token in tokens:
            word_lens[len(token.text)] += 1

    word_lens = list(word_lens.values())
    sent_lens = list(sent_lens.values())
    parse_tree_depths = list(parse_tree_depths.values())
    dict_features.append({
        'doc_num_sents': doc_num_sents,
        'doc_num_tokens': doc_num_tokens,
        **list_encode('word_lens', word_lens, args),
        **list_encode('sent_lens', sent_lens, args),
        **list_encode('parse_tree_depths', parse_tree_depths, args),
    })
    return dict_features[0]

def multiprocess_parse_data(texts, n_jobs, args):
    p = Pool(n_jobs)
    all_args = [(elt, args) for elt in texts]
    dict_features = p.starmap(parse_data, all_args, chunksize=2)
    # dict_features = p.starmap(parse_data, all_args)
    return dict_features

class CustomFeatures(TransformerMixin):
    def __init__(
        self,
        n_jobs=1,
        num_sents=True,
        num_tokens=True,
        word_len=True,
        sent_len=True,
        parse_tree_depth=True,
        encode_min=False,
        encode_max=True,
        encode_avg=True,
        encode_std=True,
        encode_full=False,
    ):
        self.n_jobs=n_jobs
        self.dict_vectorizer = DictVectorizer()
        self.args = {
            'num_sents':num_sents,
            'num_tokens':num_tokens,
            'word_len':word_len,
            'sent_len':sent_len,
            'parse_tree_depth':parse_tree_depth,
            'encode_min':encode_min,
            'encode_max':encode_max,
            'encode_avg':encode_avg,
            'encode_std':encode_std,
            'encode_full':encode_full,
        }

    def fit(self, X, y=None, **fit_params):
        dict_features = multiprocess_parse_data(X, self.n_jobs, self.args)
        self.dict_vectorizer = self.dict_vectorizer.fit(dict_features)
        return self

    def transform(self, X):
        dict_features = multiprocess_parse_data(X, self.n_jobs, self.args)
        return self.dict_vectorizer.transform(dict_features)

    def fit_transform(self, X, y=None, **fit_params):
        dict_features = multiprocess_parse_data(X, self.n_jobs, self.args)
        self.dict_vectorizer = self.dict_vectorizer.fit(dict_features)
        return self.dict_vectorizer.transform(dict_features)