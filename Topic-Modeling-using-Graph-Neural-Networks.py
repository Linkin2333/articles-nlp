##########
#
# Source code for Medium.com articles discussing Natural Language Processing (NLP) Techniques
# written by [Eric Broda](https://www.linkedin.com/in/ericbroda/)
#
# Full article: (medium.com)
#
# Instructions:
#
# 1. Install the dependant packages:
#
#       pip install -r requirements.txt
#
# 2. Install the spacy vocabulary:
#
#       python -m spacy download en_core_web_lg
#
# 3.  Get the data:
#
#       [command to download the data]
#
# 4.  Run the python code:
#
#       python Graph-Neural-Networks-for-Topic-Modeling.py
#
##########
import networkx as nx
import random
import numpy as np
import pandas as pd
import math
import collections
import statistics
import pprint
import pickle

import stellargraph as sg
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler

from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.cluster import KMeans
import hdbscan

import os
import tensorflow as tf
import keras
from keras import backend as K

import spacy
import uuid
import re
import json
from collections import Counter
import logging.config

# LOCAL IMPORTS
# import sys
# import_dir = os.path.expanduser("../src")
# import_dir = os.path.abspath(import_dir)
# sys.path.append(import_dir)  # insert at 1, 0 is the script path (or '' in REPL)
# import common as common


RANDOM_SEED = 42
BATCH_SIZE = 50
NUM_SAMPLES = [5, 5]
EPOCHS = 3

PCT_CLUSTER_CLEAN_THRESHOLD = 0.25  # HDBSCAN parameters
MIN_KMEANS_CLUSTERS = 3       # Minimum number of clusters for KMEANS
MAX_KMEANS_CLUSTERS = 15      # Maximum number of clusters for KMEANS

TRANSFORMATION = "UMAP"  # PCA is reproducible, TSNE and UMAP are fast but results vary each run, but clusters seem to align

VOCABULARY = "en_core_web_lg"    # Spacy vocabulary to use
XGRAM_THRESHOLD = 3              # Max size of x-grams for phrases / noun-chunks
XGRAM_FREQUENCY_CUTOFF = 100     # Min occurrences of an phrase (xgram) for it to be considered relevant (ie. used in) processing
MAX_TOKENS_IN_SENTENCE = 300     # Maximum number of tokens for a valid sentence (over this is ignored due to bad puctuation)
MAX_TOKEN_LENGTH = 25            # Maximum characters in token text

# IMPORTANT_TAGS = ["noun", "propn", "x", "adj"]                # Spacy POS tags that are considered important
IMPORTANT_TAGS = ["noun", "propn", "x"]                         # Spacy POS tags that are considered important
CONSUMED_TAGS = IMPORTANT_TAGS                                  # Spacy POS tags that are considered useful (others are not included in analysis)
# CONSUMED_TAGS = ["noun", "adj", "verb", "adv", "propn", "x"]  # Spacy POS tags that are considered useful (others are not included in analysis)

FACTOR_IMPORTANCE = 0.5                                   # Scaling factor to simplify term importance calculation
HIGH_IMPORTANCE = 1.0                                     # High (very relevant) term importance (distance)
MEDIUM_IMPORTANCE = HIGH_IMPORTANCE * FACTOR_IMPORTANCE   # Medium term importance
LOW_IMPORTANCE = MEDIUM_IMPORTANCE * FACTOR_IMPORTANCE    # Low term importance
VLOW_IMPORTANCE = LOW_IMPORTANCE * FACTOR_IMPORTANCE      # Very low term importance
NO_IMPORTANCE = 0.0                                       # Term has no importance

TFIDF_KEEP_PCT = 0.98               # Pct of most useful TFIDF words to keep
FILTER_PRESENT_MIN_DOCS = 3         # Ignore words that appear in less than this NUMBER of documents; HIGHER absolute number filter/remove MORE words
FILTER_PRESENT_MAX_DOCS = 0.8       # Ignore words that appear in more than this PCT documents; HIGHER pct (0-1) filter/remove FEWER words
FILTER_BIG_WORD_LENGTH = 8          # Filter small words less than or equal to this
MAX_TFIDF_FEATURES = 10000          # Max TFIDF features to consider

MAX_DEGREE_FILTERING = 0.00         # Filter this amount of nodes, higher numbers filters nodes with smaller degrees

MIN_FILTERED_WORD_LEN = 5           # Filtered word length must be smaller or equal to this

READER_DIRECTORY = "dir"      # CLI argument, indicates directory input type
READER_CSV = "csv"            # CLI argument, indicates CSV input type
READER_JSON = "json"          # CLI argument, indicates JSON input type

OOV_FILE = "/Users/ericbroda/Development/python/gnn/tests/oov.json"

# Number of rows statistics:
# Time to complete model: 1000: 75sec, 10000:500sec (~8 min), 100000: 55min
# - Size of model: 1000: 6MB, 2400: 10MB
NUM_ROWS = 100

INPUT_FILE = "/Users/ericbroda/Data/misc/amazon-fine-food-reviews/train-0.10.csv"
FIELDS = ["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "Time", "Summary", "Text"]
DATA_FIELD = "Text"

# INPUT_FILE = "/Users/ericbroda/Data/misc/tech-topics/tech-topics-small.csv"
# FIELDS = ["content"]
# DATA_FIELD = "content"

# INPUT_FILE = "/Users/ericbroda/Data/kaggle/abcnews-date-text.csv"
# FIELDS = ["publish_date", "headline_text"]
# DATA_FIELD = "headline_text"


##########
#
# Classes
#
##########
class LoaderDocument:

    def __init__(self):
        self._documents = None
        self._documents_by_label = None
        self._documents_original = None
        self._tokens_by_idx = None
        self._tokens_by_label = None
        self._tokens_by_document = None
        self._tokens_avg = None
        self._token_count = None
        self._filtered_words = None
        self._nlp = None
        self._oov_map = None
        self._oov_vectors = None
        self._graph = None
        self._features = None
        self._feature_columns = None
        self._feature_labels = None
        self._ssentiments = None
        self._dsentiments = None
        self._tsentiments = None

    def load(self, input_spec, OOV_FILE, fields, data_field, num_rows=None, frac=1.0, seed=RANDOM_SEED):
        """
        Load documents from input specification (directory) using
        OOV (out of vocabulary) input file (oov.json) if it is present
        and create graph, features, and document-node mapping
        """

        log().debug(f"Reading filename: {input_spec}, fields: {fields} data_field: {data_field} num_rows: {num_rows} frac: {frac}")
        df = pd.read_csv(input_spec, engine="python", encoding="ISO-8859-1", names=fields, header=0)
        log().debug(f"Loaded dataframe, shape: {df.shape}: \n{df.head()}")

        # Sample the data (if num_rows exists then use it, otherwise use frac (default is 100% of data)
        if num_rows:
            total_rows = df.shape[0]
            if num_rows > total_rows:
                num_rows = total_rows
            df = df.sample(n=num_rows, random_state=seed)
        else:
            df = df.sample(frac=frac, random_state=seed)
        log().debug(f"Input (sampled) dataframe, shape: {df.shape}: \n{df.head()}")

        # Get the raw original documents prior to any processing
        documents_original = df[data_field].values.tolist()

        # Acquire and clean the raw data
        documents = []
        for index, row in df.iterrows():
            data = row[data_field]
            data = self._clean_data(data)
            documents.append([data])

        # Create a list of documents (each document is a list item)
        data = []
        for content in documents:
            content = " ".join(content)
            data.append(content)

        log().info("Loading spacy vocabulary: {}".format(VOCABULARY))
        nlp = spacy.load(VOCABULARY)
        log().info("Loading spacy vocabulary completed: {}".format(VOCABULARY))

        # Read OOV map file if it exists
        log().debug(f"Using OOV file: {OOV_FILE}")
        with open(OOV_FILE) as f:
            oov_map = json.load(f)

        oov_vectors = {}
        for item in oov_map.keys():
            word = oov_map[item]
            word_id = nlp.vocab.strings[word]
            word_vector = nlp.vocab.vectors[word_id]
            oov_vectors[item] = word_vector
        self._oov_map = oov_map
        self._oov_vectors = oov_vectors

        log().debug(f"OOV Mapping: {self._oov_map}")
        # log().debug(f"_oov_vectors: {self._oov_vectors}")

        # After much testing it appears that batch size has a strong impact on throughput.
        # Large batch sizes seem to hinder multi-threading and ultimately reduces CPU consumption
        # at least on smaller machines.  It seems that maximum thoughput and CPU usage
        # occurs with a very small batch size, hence it is set to 1 right now.
        # batch_size = int(len(data) / 1000)
        batch_size = 1

        # Add a test pipeline element (just to show how it can be done)
        nlp.add_pipe(self._pipeline, name="filter", last=True)

        log().info(f"Processing {len(data)} documents")
        from tqdm import tqdm
        docs = tqdm(nlp.pipe(data, batch_size=batch_size, n_threads=20))
        docs = list(docs)
        log().info(f"Processing {len(data)} documents completed")

        phrases = self._find_phrases(docs)
        phrases = sorted(phrases.items(), key=lambda kv: kv[1], reverse=True)
        # self._show_phrases(phrases)

        log().debug("Creating document tokens")
        documents = self._process_documents(docs)

        log().debug("Reconciling phrases")
        documents = self._reconcile_phrases(documents, phrases)

        log().debug("Reconciling children")
        documents = self._reconcile_children(documents)

        labels = [token["label"] for document in documents for token in document]
        token_count = Counter(labels)
        # log().debug(f"Most common token labels: {token_count.most_common()[0:20]}")

        documents_by_label = collections.defaultdict(list)
        tokens_by_label = collections.defaultdict(list)
        tokens_by_document = collections.defaultdict(list)
        for i, document in enumerate(documents):
            for token in document:
                label = token["label"]
                documents_by_label[label].append(i)
                tokens_by_label[label].append(token)
                tokens_by_document[i].append(token)

        self._documents_original = documents_original
        self._documents = documents
        self._documents_by_label = documents_by_label
        self._tokens_by_idx = {token["idx"]: token for document in documents for token in document}
        self._tokens_by_label = tokens_by_label
        self._tokens_by_document = tokens_by_document
        self._token_count = token_count
        self._nlp = nlp

        log().debug("Calculating filter words")
        filtered_words = self._find_filtered()
        self._filtered_words = filtered_words

        # log().debug(f"filtered_words: {len(filtered_words)} \n {filtered_words}")
        # log().debug(f"tokens_by_document: {[str(d) + ':' + t['lemma'] for d in tokens_by_document.keys() for t in tokens_by_document[d]]}")

    def features(self):
        """
        Return the features for the loaded/input data
        """

        log().info(f"Analysing features")

        if not self._graph:
            raise Exception("This loader requires that the graph be created before features - please call graph() first")

        log().info(f"Calculating metrics")
        ranks, hits, authorities, degree_centrality = calc_metrics(self._graph)
        log().info(f"Calculating metrics complete")

        log().info(f"Creating token features")
        features = []
        for node in self._graph.nodes:
            if node not in self._tokens_by_label.keys():
                log().debug(f"features node: {node} not in LABEL dictionary: {pretty(self._tokens_by_label.keys())}")
                continue
            matches = self._tokens_by_label[node]
            token = matches[0]

            # rank = ranks[token["label"]]
            # hit = hits[token["label"]]
            # auth = authorities[token["label"]]
            # dc = degree_centrality[token["label"]]

            # f = self._create_features(token, rank, hit, auth, dc)
            f = self._create_features(token)
            features.append(f)

        log().info(f"Creating token features complete")

        df = self._create_feature_df(features)

        # for col in df.columns.values:
        #     uniques = df[col].unique().tolist()
        #     num_uniques = len(uniques)
        #     if num_uniques == 1:
        #         log().debug(f"Column {col} has a single ({num_uniques}) unique values")
        #     else:
        #         log().debug(f"Column {col} has {num_uniques} unique values")

        # Sort the column order
        cols = df.columns.tolist()
        cols.sort()
        df = df[cols]

        # Establish the labe as the index
        label_column = "label"
        df.set_index(label_column, inplace=True)

        nodes = list(self._graph.nodes())
        nodes.sort()

        num_nodes = len(nodes)
        (num_features, num_columns) = df.shape
        if num_nodes != num_features:
            log().debug(f"nodes: {num_nodes} {nodes}")
            msg = f"Nodes and features length do not match, nodes: {num_nodes} features: {num_features}"
            raise Exception(msg)

        log().info(f"Analysing features complete")

        df = df.reset_index(level=0)
        labels = df["label"].to_list()
        columns = list(df.columns.values)
        df = df.drop("label", axis=1)
        features = df.to_numpy()

        self._features = features
        self._feature_columns = columns
        self._feature_labels = labels
        return features, labels, columns

    def graph(self):
        """
        Return the graph for the loaded/input data
        """

        documents = self._documents
        # tokens = [token["idx"] for document in documents for token in document]
        # token_lemmas = [token["lemma"] for document in documents for token in document]
        # log().debug(f"Document tokens: \n{pretty(token_lemmas)}")

        filtered_words = self._filtered_words
        log().debug("Calculating edges, docs: {}".format(len(documents)))
        edges = self._calc_edges(documents, filtered_words)

        log().debug("Creating graph, edges: {}".format(len(edges)))
        graph = create_graph(edges, graph_type="multigraph")

        node_degrees = list(graph.degree())
        distribution = Counter([degree for (node, degree) in node_degrees])

        total_nodes = len(node_degrees)
        sum_nodes = 0
        min_degree_threshold = 0
        pct = 0
        for (degree, count) in distribution.most_common():
            sum_nodes += count
            pct = sum_nodes / total_nodes
            if pct > MAX_DEGREE_FILTERING:
                min_degree_threshold = degree
                break
        min_degree_threshold = 0
        # log().debug(f"min_degree_threshold: {min_degree_threshold} pct: {pct} pruning: {sum_nodes} of total: {total_nodes}")

        for (node, degree) in node_degrees:
            if degree < min_degree_threshold:
                # log().debug(f"Pruning node: {node} with degree: {degree}")
                # log().debug(f"Pruning node: {node} with degree: {degree}")
                prune_node(graph, node)

        # Remove nodes which have no links (ie. degree == 0) which result from pruning
        node_degrees = list(graph.degree())
        for (node, degree) in node_degrees:
            if degree == 0:
                # log().debug(f"Removing node: {node} with degree: {degree}")
                graph.remove_node(node)

        node_degrees = list(graph.degree())
        distribution = Counter([degree for (node, degree) in node_degrees])
        log().debug(f"Filtered degree distribution: {distribution.most_common()}")

        self._graph = graph
        return self._graph

    def document_tokens(self):
        """
        Return the dictionary of document tokens (document id is key, mapped to list of tokens)
        """
        return self._tokens_by_document

    def documents(self):
        """
        Return the list of original documents
        """
        return self._documents_original

    def document_labels(self):
        """
        Return the list of document-to-label mappings
        """
        return self._documents_by_label

    def filtered_words(self):
        """
        Return the list of document-to-node mappings
        """
        return self._filtered_words

    ##########
    # INTERNAL FUNCTIONS
    ##########

    def _all_weights_pos(self):
        """
        return a dictionary of POS weights
        """
        x = {
            "adj": LOW_IMPORTANCE,      # adjective: big, old, green, incomprehensible, first
            "adp": LOW_IMPORTANCE,      # adposition: in, to, during
            "adv": LOW_IMPORTANCE,      # adverb: very, tomorrow, down, where, there
            "aux": VLOW_IMPORTANCE,     # auxiliary: is, has (done), will (do), should (do)
            "conj": LOW_IMPORTANCE,     # conjunction: and, or, but
            "cconj": LOW_IMPORTANCE,    # coordinating conjunction: and, or, but
            "det": VLOW_IMPORTANCE,     # determiner: a, an, the
            "intj": LOW_IMPORTANCE,     # interjection: psst, ouch, bravo, hello
            "num": VLOW_IMPORTANCE,     # num: one, 2, third
            "noun": HIGH_IMPORTANCE,    # noun: girl, cat, tree, air, beauty
            "part": VLOW_IMPORTANCE,    # particle: ’s, not,
            "pron": LOW_IMPORTANCE,     # pronoun: I, you, he, she, myself, themselves, somebody
            "propn": HIGH_IMPORTANCE,   # proper noun: Mary, John, London, NATO, HBO
            "punct": VLOW_IMPORTANCE,   # punctuation: ., (, ), ?
            "sconj": LOW_IMPORTANCE,    # subordinating conjunction: if, while, that
            "sym": VLOW_IMPORTANCE,     # symbol: zz$, %, §, ©, +, −, ×, ÷, =, :)
            "verb": MEDIUM_IMPORTANCE,  # verb: run, runs, running, eat, ate, eating
            "x": MEDIUM_IMPORTANCE,     # other: sfpksdpsxmsa
            "space": VLOW_IMPORTANCE,   # space:
        }
        return x

    def _all_weights_tag(self):
        """
        Return a dictionary of TAG weights
        """
        x = {
            ".": VLOW_IMPORTANCE,       # punctuation mark, sentence closer
            ",": VLOW_IMPORTANCE,       # punctuation mark, comma
            "-lrb-": VLOW_IMPORTANCE,   # left round bracket
            "-rrb-": VLOW_IMPORTANCE,   # right round bracket
            "``": VLOW_IMPORTANCE,      # opening quotation mark
            '""': VLOW_IMPORTANCE,      # closing quotation mark
            "''": VLOW_IMPORTANCE,      # closing quotation mark
            ":": VLOW_IMPORTANCE,       # punctuation mark, colon or ellipsis
            "$": VLOW_IMPORTANCE,       # symbol, currency
            "#": VLOW_IMPORTANCE,       # symbol, number sign
            "afx": VLOW_IMPORTANCE,     # affix
            "cc": MEDIUM_IMPORTANCE,    # conjunction, coordinating
            "cd": VLOW_IMPORTANCE,      # cardinal number
            "dt": VLOW_IMPORTANCE,      # determiner
            "ex": VLOW_IMPORTANCE,      # existential there
            "fw": MEDIUM_IMPORTANCE,    # foreign word
            "hyph": VLOW_IMPORTANCE,    # punctuation mark, hyphen
            "in": MEDIUM_IMPORTANCE,    # conjunction, subordinating or preposition
            "jj": LOW_IMPORTANCE,       # adjective
            "jjr": LOW_IMPORTANCE,      # adjective, comparative
            "jjs": LOW_IMPORTANCE,      # adjective, superlative
            "ls": VLOW_IMPORTANCE,      # list item marker
            "md": MEDIUM_IMPORTANCE,    # verb, modal auxiliary
            "nil": VLOW_IMPORTANCE,     # missing tag
            "nn": HIGH_IMPORTANCE,      # noun, singular or mass
            "nnp": HIGH_IMPORTANCE,     # noun, proper singular
            "nnps": HIGH_IMPORTANCE,    # noun, proper plural
            "nns": HIGH_IMPORTANCE,     # noun, plural
            "pdt": LOW_IMPORTANCE,      # predeterminer
            "pos": VLOW_IMPORTANCE,     # possessive ending
            "prp": VLOW_IMPORTANCE,     # pronoun, personal
            "prp$": VLOW_IMPORTANCE,    # pronoun, possessive
            "rb": LOW_IMPORTANCE,       # adverb
            "rbr": LOW_IMPORTANCE,      # adverb, comparative
            "rbs": LOW_IMPORTANCE,      # adverb, superlative
            "rp": LOW_IMPORTANCE,       # adverb, particle
            "sp": VLOW_IMPORTANCE,      # space
            "sym": VLOW_IMPORTANCE,     # symbol
            "to": VLOW_IMPORTANCE,      # infinitival to
            "uh": VLOW_IMPORTANCE,      # interjection
            "vb": MEDIUM_IMPORTANCE,    # verb, base form
            "vbd": MEDIUM_IMPORTANCE,   # verb, past tense
            "vbg": MEDIUM_IMPORTANCE,   # verb, gerund or present participle
            "vbn": MEDIUM_IMPORTANCE,   # verb, past participle
            "vbp": MEDIUM_IMPORTANCE,   # verb, non-3rd person singular present
            "vbz": MEDIUM_IMPORTANCE,   # verb, 3rd person singular present
            "wdt": LOW_IMPORTANCE,      # wh-determiner
            "wp": LOW_IMPORTANCE,       # wh-pronoun, personal
            "wp$": LOW_IMPORTANCE,      # wh-pronoun, possessive
            "wrb": LOW_IMPORTANCE,      # wh-adverb
            "add": MEDIUM_IMPORTANCE,   # email
            "nfp": VLOW_IMPORTANCE,     # superfluous punctuation
            "gw": MEDIUM_IMPORTANCE,    # additional word in multi-word expression
            "xx": MEDIUM_IMPORTANCE,    # unknown
            "bes": VLOW_IMPORTANCE,     # auxiliary "be"
            "hvs": VLOW_IMPORTANCE,     # forms of "have"
            "_sp": VLOW_IMPORTANCE,     # ??
        }
        return x

    def _all_weights_dep(self):
        """
        Return a dictionary of DEP weights
        """

        # Gramatic terms are documented at:
        # https://spacy.io/api/annotation and
        # https://github.com/explosion/spacy/issues/233
        x = {
            "acl": MEDIUM_IMPORTANCE,        # clausal modifier of noun: experienced, schooled, raising, announced
            "acomp": MEDIUM_IMPORTANCE,      # adjective complement: sure, clear, right, small, worse, easier
            "advcl": MEDIUM_IMPORTANCE,      # adverbial clause modifier: seeking, reported, keep, have, worked
            "advmod": VLOW_IMPORTANCE,       # adverbial modifier: together, relatively, early, often, back, recently
            "adp": MEDIUM_IMPORTANCE,        # ??: seems to include bi/tri-grams
            "agent": VLOW_IMPORTANCE,        # ??: by
            "amod": MEDIUM_IMPORTANCE,       # adjectival modifier: last, little, special, low, own, tough
            "appos": HIGH_IMPORTANCE,        # appositional modifier: member, father, author, chairman, leader
            "attr": HIGH_IMPORTANCE,         # attribute: step, differences, winners, prominent, risk
            "aux": VLOW_IMPORTANCE,          # auxiliary: had, has would, will, is, do
            "auxpass": VLOW_IMPORTANCE,      # passive auxiliary: be, are, been, was, is, am
            "case": VLOW_IMPORTANCE,         # passive auxiliary: 's
            "cc": VLOW_IMPORTANCE,           # coordinating conjunction: and, more, but, nor, merge
            "ccomp": LOW_IMPORTANCE,         # clausal complement: heard, encourage, won, see, knew, read, felt, become
            "compound": HIGH_IMPORTANCE,     # compound: Mr., Mrs., [proper noun], [name]
            "conj": MEDIUM_IMPORTANCE,       # conjunct: craving, returned, filtered, voice, prepared, wish
            "csubj": MEDIUM_IMPORTANCE,      # clausal subject: served, saying, briefed, standing, held
            "csubjpass": LOW_IMPORTANCE,     # clausal passive subject: ??
            "dative": HIGH_IMPORTANCE,       # dative: [proper nouns], someone, detectives, party, [country]
            "dep": MEDIUM_IMPORTANCE,        # unspecified dependency: phoned, canceled, brought, known, meals, zoom
            "det": LOW_IMPORTANCE,           # determiner: what, which, that, both
            "dobj": HIGH_IMPORTANCE,         # direct object: job, rivals, formula, success, divisions, risks, show
            "expl": VLOW_IMPORTANCE,         # expletive: ??
            "intj": VLOW_IMPORTANCE,         # interjection: yes, hey, no, un, was, lke, goes, please
            "mark": VLOW_IMPORTANCE,         # marker: that, once, like, which, so
            "meta": MEDIUM_IMPORTANCE,       # ??: Object, architect
            "neg": VLOW_IMPORTANCE,          # negation modifier: not, n't
            "nn": MEDIUM_IMPORTANCE,         # noun compound modifier: ??
            "nounmod": MEDIUM_IMPORTANCE,    # noun modifier: ??
            "npmod": MEDIUM_IMPORTANCE,      # noun phrase as adverbial modifier: ??
            "npadvmod": VLOW_IMPORTANCE,     # date-time modifier: minutes, hour, workday, [day of week], little
            "nsubj": HIGH_IMPORTANCE,        # nominal subject: thing, program, [proper nouns], rivals
            "nsubjpass": HIGH_IMPORTANCE,    # passive nominal subject: society, actity, stocks, investors, information, provisions
            "nmod": MEDIUM_IMPORTANCE,       # nominal modifier: move, office, laptop, drinking, building, forces
            "nummod": VLOW_IMPORTANCE,       # numeric modifier: dozen, few, half, triple
            "num": VLOW_IMPORTANCE,          # number: 1, 2, three, four, fifth, sixth
            "oprd": MEDIUM_IMPORTANCE,       # object predicate: unfair, model, naked, mornings, dead, unclear, unpopular
            "obj": MEDIUM_IMPORTANCE,        # object: ??
            "obl": MEDIUM_IMPORTANCE,        # oblique nominal: ??
            "parataxis": MEDIUM_IMPORTANCE,  # parataxis: said, suggests, argued, say, told, conceded
            "pcomp": LOW_IMPORTANCE,         # complement of preposition: involves, hearing, listening, acted
            "pobj": HIGH_IMPORTANCE,         # object of preposition: leader, predecessor, constrains, remarks, drawings
            "poss": VLOW_IMPORTANCE,         # possession modifier: your, their, her, its, whose,
            "preconj": LOW_IMPORTANCE,       # pre-correlative conjunction: not, both, either
            "predet": LOW_IMPORTANCE,        # Predeterminer: all, such, both, quite
            "prep": LOW_IMPORTANCE,          # prepositional modifier: including, compared, including, before, according
            "prt": LOW_IMPORTANCE,           # particle: in, around, out, up, on
            "punct": VLOW_IMPORTANCE,        # punctuation: did, =, .5
            "quantmod": LOW_IMPORTANCE,      # modifier of quantifier: hundreds, dozen, three
            "relcl": LOW_IMPORTANCE,         # relative clause modifier: sought, elect, controls, telling, improve
            "root": MEDIUM_IMPORTANCE,       # root of sentence, typically this is a verb
            "subtok": MEDIUM_IMPORTANCE,     # sub token: anti (part of full word), "-"
            "xcomp": LOW_IMPORTANCE,         # open clausal complement: appreciate, begin, keeping, buy, increase
            "x": MEDIUM_IMPORTANCE,          # unknown POS: typically includes bi/tri-grams which are gramatically hard to decipher
            " ": NO_IMPORTANCE,              # unknown POS: typically includes bi/tri-grams which are gramatically hard to decipher
        }

        return x

    def _calc_edges(self, docs, filtered_words):
        """
        Return a list of edges based upon the input document terms, and filter words to remove less relevant terms
        """

        # for i, doc in enumerate(docs):
        #     texts = [(x["idx"], x["label"]) for x in doc]
        #     log().debug(f"doc {i}: {texts}")
        # log().debug(f"filtered_words: {filtered_words}")

        edges = []
        for i, doc in enumerate(docs):

            # Connect tokens within a single sentence (this is recursive starting root)
            roots = [x for x in doc if x["is_root"]]
            for root in roots:

                graph = self._parse_sentence(root)
                edges_labels = [
                    (self._tokens_by_idx[p]["label"], self._tokens_by_idx[t]["label"]) for (p, t) in graph.edges
                ]
                self._filter_graph(graph, filtered_words)

                # If there is only 1 node in the graph then and it is considered an important node
                # then make a self referential edge to ensure it is not lost (a single node without an
                # edge will be ignored)
                if len(graph.nodes()) == 1:
                    ps = list(graph.nodes())
                    p = ps[0]
                    edge = self._tokens_by_idx[p]["label"]
                    edges_label = (edge, edge)
                    edges.append(edges_label)

                edges_labels = [
                    (self._tokens_by_idx[p]["label"], self._tokens_by_idx[t]["label"]) for (p, t) in graph.edges
                ]
                if len(edges_labels) > 1:
                    edges_labels = [
                        (self._tokens_by_idx[p]["label"], self._tokens_by_idx[t]["label"]) for (p, t) in graph.edges
                        if self._tokens_by_idx[p]["label"] != self._tokens_by_idx[t]["label"]
                    ]
                edges.extend(edges_labels)

        return edges

    def _calc_tfidf(self, documents):
        """
        Return the TFIDF term array of a set of documents
        """

        # Transform data such that each document is a space separated set of tokens
        # which is required by the vectorizer
        docs = []
        for document in documents:
            items = [x for x in document if "punct:" not in x and "part:" not in x]
            words = " ".join(items)
            docs.append(words)

        # Split the input (text string of words separated by single space) into a list of tokens
        # Note the token is converted to lowercase.
        def local_tokenizer(x):
            tokens = x.lower().split(" ")
            return tokens

        from sklearn.feature_extraction.text import TfidfVectorizer
        tf_idf_vect = TfidfVectorizer(ngram_range=(1, 1), max_features=MAX_TFIDF_FEATURES, analyzer="word", tokenizer=local_tokenizer)
        tf_idf_vect.fit(docs)
        vocabulary = tf_idf_vect.vocabulary_

        scores = {}
        idf = tf_idf_vect.idf_
        for word in vocabulary.keys():
            id = vocabulary[word]
            score = idf[id]
            scores[word] = score

        return scores

    def _clean_data(self, data):
        """
        Return cleaned version of input data
        """

        # Lower case all data
        data = data.lower()

        # Remove Emails
        data = re.sub(r"\S*@\S*\s?", "", data)

        # Remove pronoun indicators
        data = re.sub(r"(mr\.|mrs\.)", "", data)
        data = re.sub(r"(mrs|mr)\b", "", data)
        data = re.sub(r"(mrs_|mr_)", " ", data)

        # Remove new line characters
        data = re.sub(r"\s+", " ", data)

        # Dont remove end of line punctation
        # data = re.sub(r"\,", " ", data)
        # data = re.sub(r"\.", " ", data)
        # data = re.sub(r"\;", " ", data)
        # data = re.sub(r"\?", " ", data)
        # data = re.sub(r"\!", " ", data)
        # data = re.sub(r",", " ", data)

        # Remove misc symbols
        data = re.sub(r"\#", " ", data)
        data = re.sub(r"\%", " ", data)
        data = re.sub(r"_", " ", data)
        data = re.sub(r"\|", " ", data)
        data = re.sub(r"\-", " ", data)
        data = re.sub(r"\+", "", data)
        data = re.sub(r"\=", "", data)
        data = re.sub(r"\$", "", data)
        data = re.sub(r"\/", "", data)
        data = re.sub(r"\\", "", data)
        data = re.sub(r"\(", "", data)
        data = re.sub(r"\)", "", data)
        data = re.sub(r"\^", "", data)

        # Remove types of brackets
        data = re.sub(r"\[", "", data)
        data = re.sub(r"\]", "", data)
        data = re.sub(r"\{", "", data)
        data = re.sub(r"\}", "", data)
        data = re.sub(r">", " ", data)
        data = re.sub(r"<", " ", data)

        # Dont remove single quotes (don't, isn't etc)
        # data = re.sub(r"\‘", " ", data)
        # data = re.sub(r"\’", " ", data)
        # data = re.sub(r"\`", " ", data)
        # data = re.sub(r"\'", " ", data)
        # data = re.sub(r"\’", " ", data)

        # Remove double quotes (they add no intrinsic value)
        data = re.sub(r"\“", " ", data)
        data = re.sub(r"\”", " ", data)
        data = re.sub(r"\"", " ", data)
        data = re.sub(r"\.\.\.", ".", data)
        data = re.sub(r"___", "", data)
        # data = re.sub(r"\W", " ", data)

        # Remove multiple spaces
        data = re.sub(r"\s\s+", " ", data)

        # Remove single letter words
        # data = re.sub(r"\b[a-zA-Z]\b", "", data)

        # Remove small words (2 or less)
        # data = re.sub(r'\b\w{1,2}\b', '', data)

        # Remove words that are all numbers (no alphabetic chars), but not ones that contain a number
        # data = re.sub(r"\b\d+\b", "", data)

        return data

    def _create_features(self, token):

        f = {}
        # Index
        f["label"] = token["label"]

        # Unique identifier (redundant?)
        f["idx"] = token["idx"]

        # Below have only a single value and hence are useless
        # f["is_ascii"] = token["is_ascii"]
        # f["is_digit"] = token["is_digit"]
        # f["is_lower"] = token["is_lower"]
        # f["is_upper"] = token["is_upper"]
        # f["is_title"] = token["is_title"]
        # f["like_url"] = token["like_url"]
        # f["like_num"] = token["like_num"]
        # f["like_email"] = token["like_email"]
        # f["is_punct"] = token["is_punct"]

        # Boolean attributes
        f["is_alpha"] = token["is_alpha"]
        f["is_oov"] = token["is_oov"]
        f["is_stop"] = token["is_stop"]

        # Encoded attributes
        f["pos_"] = token["pos_"]
        f["tag_"] = token["tag_"]
        f["dep_"] = token["dep_"]
        # f["lang_"] = token["lang_"]
        f["ent_type_"] = token["ent_type_"]
        f["ent_iob_"] = token["ent_iob_"]
        # f["l2_norm"] = token["l2_norm"]

        # Numeric attributes
        f["x-posi"] = token["x-posi"]  # Already scaled between 0-1
        f["x-tagi"] = token["x-tagi"]  # Already scaled between 0-1
        f["x-depi"] = token["x-depi"]  # Already scaled between 0-1
        f["x-token_polarity"] = token["token_polarity"]
        f["x-token_subjectivity"] = token["token_subjectivity"]
        f["x-token_positive"] = token["token_positive"]
        f["x-token_neutral"] = token["token_neutral"]
        f["x-token_negative"] = token["token_negative"]
        f["x-token_compound"] = token["token_compound"]
        f["x-sentence_polarity"] = token["sentence_polarity"]
        f["x-sentence_subjectivity"] = token["sentence_subjectivity"]
        f["x-sentence_positive"] = token["sentence_positive"]
        f["x-sentence_neutral"] = token["sentence_neutral"]
        f["x-sentence_negative"] = token["sentence_negative"]
        f["x-sentence_compound"] = token["sentence_compound"]
        # f["x-doc_polarity"] = token["doc_polarity"]
        # f["x-doc_subjectivity"] = token["doc_subjectivity"]
        # f["x-doc_positive"] = token["doc_positive"]
        # f["x-doc_neutral"] = token["doc_neutral"]
        # f["x-doc_negative"] = token["doc_negative"]
        # f["x-doc_compound"] = token["doc_compound"]

        for i, item in enumerate(token["vector"]):
            cname = "v-" + str(i)
            f[cname] = item

        # f["rank"] = rank
        # f["hit"] = hit
        # f["auth"] = auth
        # f["degree_centrality"] = degree_centrality
        # f["eigenvector_centrality"] = eigenvector_centrality

        return f

    def _create_feature_df(self, features, redim=False):

        if redim:
            # Redim the large vector space into something much smaller
            # Step 1: get new vector
            # Step 2: Remove old vector data
            # Step 3: Add new vector data

            # Step 1: Get new vector data (reduce dimensions)
            (nrows, nfeatures) = features.shape
            components = int(nfeatures / 4)
            log().debug(f"Reducing dimensions from {nfeatures}to: {components}")
            vector = self._redim_vectors(features, components=components)

            # Step 2: Remove the old vector data
            log().debug(f"Removing old vector data")
            for feature in features:
                for col in list(feature.keys()):
                    if col.startswith("v-"):
                        del feature[col]

            # Step 3:Add the new vector data
            log().debug(f"Adding new vector data")
            for i, feature in enumerate(features):
                for j, item in enumerate(vector[i]):
                    cname = "v-" + str(j)
                    feature[cname] = item

        # Create the feature dataframe and: remove str idx and change booleans to integer
        df = pd.DataFrame(features)
        # log().debug(f"Creating feature dataframe using features: {features}")

        # Drop the "idx" column (redundant to the actual index)
        df.drop(["idx"], axis=1, inplace=True)

        # Switch boolean to integer
        df[["is_alpha", "is_oov", "is_stop"]] *= 1

        # Encode POS
        # categories = list(self._all_weights_pos().keys())
        categories = ["adj", "adv", "noun", "propn", "verb", "x"]
        column = "pos_"
        df = self._encode(df, column, categories)

        # Encode DEP
        # categories = list(self._all_weights_dep().keys())
        categories = [
            "acl", "acomp", "advcl", "advmod", "amod", "appos", "attr", "aux", "ccomp", "compound", "conj", "csubj", "dobj",
            "npadvmod", "nsubj", "nsubjpass", "nmod", "pcomp", "pobj", "poss", "prep", "relcl", "root", "xcomp"
        ]
        column = "dep_"
        df = self._encode(df, column, categories)

        # Encode TAG
        # categories = list(self._all_weights_tag().keys())
        categories = [
            "fw", "jj", "nn", "nns", "xx"
        ]
        column = "tag_"
        df = self._encode(df, column, categories)

        # Encode ENT_IOB
        categories = ["i", "b", "o"]
        column = "ent_iob_"
        df = self._encode(df, column, categories)

        # Encode ENT_TYPE
        categories = [
            "x", "date", "percent", "cardinal", "ordinal", "time", "org", "person", "quantity", "gpe"
        ]
        column = "ent_type_"
        df = self._encode(df, column, categories)

        self._describe(df, ["is_oov", "is_stop"], values=True)

        return df

    def _create_label(self, token):
        """
        Return a custom token label based upon the input token
        """

        lemma = None
        if token["type"] == "phrase":
            lemma = token["lemma"].replace(" ", "_")
        else:
            lemma = token["lemma"]

        # NOTE: other functions expect "noun:" and other POS constructs - removing them
        # from the label will cause zero data to be processed

        # tname = token["pos_"] + ":" + token["tag_"] + ":" + lemma
        tname = token["pos_"] + ":" + lemma

        if token["tag_"] != "_sp" and token["dep_"] not in self._all_weights_dep().keys() and len(token["text"]) != 0:
            log().debug("Unusual DEP: {}, dep: {}, lemma: {}, pos: {}, tag: {}".format(tname, token["dep_"], lemma, token["pos_"], token["tag_"]))
            tname = None

        return tname

    def _create_phrase_token(self, document, start, end):
        """
        Return a phrase token composed from the input document and start/end items
        """

        # Root is the last token in the list of input tokens (noun phrase, by definition has the noun last)
        root = document[end]
        tokens = document[start:end + 1]

        token = root
        x = {
            "type": "phrase",
            "text": " ".join([x["text"] for x in tokens]),
            "lemma": "_".join([x["lemma"] for x in tokens if x["pos_"] in CONSUMED_TAGS]),
            "is_root": token["is_root"],
            "is_alpha": token["is_alpha"],
            "is_ascii": token["is_ascii"],
            "is_digit": token["is_digit"],
            "is_lower": token["is_lower"],
            "is_upper": token["is_upper"],
            "is_title": token["is_title"],
            "is_oov": token["is_oov"],
            "is_stop": token["is_stop"],
            "is_punct": token["is_punct"],
            "like_url": token["like_url"],
            "like_num": token["like_num"],
            "like_email": token["like_email"],
            "pos": token["pos"],
            "pos_": token["pos_"],
            "x-posi": token["x-posi"],
            "tag": token["tag"],
            "tag_": token["tag_"],
            "x-tagi": token["x-tagi"],
            "dep": token["dep"],
            "dep_": token["dep_"],
            "x-depi": token["x-depi"],
            "lang": token["lang"],
            "lang_": token["lang_"],
            "prob": token["prob"],
            "ent_type": token["ent_type"],
            "ent_type_": token["ent_type_"],
            "ent_iob": token["ent_iob"],
            "ent_iob_": token["ent_iob_"],

            "token_polarity": token["token_polarity"],
            "token_subjectivity": token["token_subjectivity"],
            "token_positive": token["token_positive"],
            "token_neutral": token["token_neutral"],
            "token_negative": token["token_negative"],
            "token_compound": token["token_compound"],

            "sentence_polarity": token["sentence_polarity"],
            "sentence_subjectivity": token["sentence_subjectivity"],
            "sentence_positive": token["sentence_positive"],
            "sentence_neutral": token["sentence_neutral"],
            "sentence_negative": token["sentence_negative"],
            "sentence_compound": token["sentence_compound"],

            # "doc_polarity": token["doc_polarity"],
            # "doc_subjectivity": token["doc_subjectivity"],
            # "doc_positive": token["doc_positive"],
            # "doc_neutral": token["doc_neutral"],
            # "doc_negative": token["doc_negative"],
            # "doc_compound": token["doc_compound"],
        }

        label = self._create_label(x)
        x["label"] = label

        composition = [x["idx"] for x in tokens]
        x["composition"] = composition

        # x["idx"] = calc_hash(json.dumps(x))
        x["idx"] = token["idx"]

        # Vector cannot be hashed - add it here
        x["vector"] = token["vector"]

        children = []
        for token in tokens:
            for child in token["children"]:
                children.append(child)

        for cidx in list(children):
            for i in range(start, end):
                merged_token = document[i]
                if cidx == merged_token["idx"]:
                    cchildren = merged_token["children"]
                    children.extend(cchildren)
                    children.remove(cidx)
                    break
        x["children"] = children
        return x

    def _create_token(self, token):
        """
        Return a custom token based upon the input spacy token
        """

        # didx = calc_hash(str(token.sent.doc.text))
        # if didx not in self._dsentiments:
        #     log().debug(f"Document sentiment idx not found: {didx} text: {token.doc.text[0:25]}")
        # dsentiment = self._dsentiments[didx]

        ssentiment = {
            "polarity": 0,
            "subjectivity": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "compound": 0
        }
        sidx = calc_hash(token.sent.text)
        if sidx not in self._ssentiments:
            log().debug(f"Sentence sentiment idx not found: {sidx} text: {token.sent.text[0:25]}")
        else:
            ssentiment = self._ssentiments[sidx]

        tsentiment = {
            "polarity": 0,
            "subjectivity": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "compound": 0
        }
        tidx = calc_hash(token.text)
        if tidx not in self._tsentiments:
            log().debug(f"Token sentiment idx not found: {tidx} text: {token.text[0:25]}")
        else:
            tsentiment = self._tsentiments[tidx]

        x = {
            "type": "token",
            "spacy_token": token,
            "text": token.text.lower(),
            "lemma": token.lemma_.lower(),
            "is_root": (token == token.sent.root),
            "is_alpha": token.is_alpha,
            "is_ascii": token.is_ascii,
            "is_digit": token.is_digit,
            "is_lower": token.is_lower,
            "is_upper": token.is_upper,
            "is_title": token.is_title,
            "is_oov": token.is_oov,
            "is_stop": token.is_stop,
            "is_punct": token.is_punct,
            "like_url": token.like_url,
            "like_num": token.like_num,
            "like_email": token.like_email,
            "pos": token.pos,
            "pos_": token.pos_.lower(),
            "x-posi": self._all_weights_pos()[token.pos_.lower()],
            "tag": token.tag,
            "tag_": token.tag_.lower(),
            "x-tagi": self._all_weights_tag()[token.tag_.lower()],
            "dep": token.dep,
            "dep_": token.dep_.lower(),
            "x-depi": self._all_weights_dep()[token.dep_.lower()],
            "lang": token.lang,
            "lang_": token.lang_.lower(),
            "prob": token.prob,
            "ent_type": token.ent_type,
            "ent_type_": token.ent_type_.lower(),
            "ent_iob": token.ent_iob,
            "ent_iob_": token.ent_iob_.lower(),

            "token_polarity": tsentiment["polarity"],
            "token_subjectivity": tsentiment["subjectivity"],
            "token_positive": tsentiment["positive"],
            "token_neutral": tsentiment["neutral"],
            "token_negative": tsentiment["negative"],
            "token_compound": tsentiment["compound"],

            "sentence_polarity": ssentiment["polarity"],
            "sentence_subjectivity": ssentiment["subjectivity"],
            "sentence_positive": ssentiment["positive"],
            "sentence_neutral": ssentiment["neutral"],
            "sentence_negative": ssentiment["negative"],
            "sentence_compound": ssentiment["compound"],

            # "doc_polarity": dsentiment["polarity"],
            # "doc_subjectivity": dsentiment["subjectivity"],
            # "doc_positive": dsentiment["positive"],
            # "doc_neutral": dsentiment["neutral"],
            # "doc_negative": dsentiment["negative"],
            # "doc_compound": dsentiment["compound"],
        }

        # Cleanup token
        if x["ent_type_"] == "":
            x["ent_type_"] = "x"

        label = self._create_label(x)
        x["label"] = label

        # UUID idx for each token representing exact information from parsing
        x["idx"] = str(uuid.uuid4())

        # Vector cannot be hashed - add it here
        x["vector"] = token.vector
        import numpy as np
        if np.sum(token.vector) == 0 or token.is_oov:
            # Map OOV vectors to defined values - note that is_oov is now marked as False
            # and only non-mapped values will be designated true is_oov
            if self._oov_vectors and label in self._oov_vectors:
                x["is_oov"] = False
                x["vector"] = self._oov_vectors[label]
                # log().debug(f"Substituting vector for label: {label} replacing with vector for word: {self._oov_map[label]}")
            else:
                x["is_oov"] = True
                log().debug(f"OOV (out of vocabulary, unknown) label: {label}")

        # Create empty children list (filled later, once all regular tokens have an index)
        x["children"] = []

        return x

    def _describe(self, df, cols, values=False):
        output = ""
        total_rows = len(df)
        for col in cols:
            xdf = df[df[col] == 1]
            xdf.reset_index(inplace=True, drop=True)
            (xrows, xcols) = xdf.shape
            output += f"\n --- {col}: rows: {xrows:5} of total rows: {total_rows:5} (pct: {xrows / total_rows:0.4f}) ---"
            output += f"\n{xdf[col].describe()}"
            if values:
                output += f"\n {xdf['label'].sort_values()}"
        log().debug(output)

    def _encode(self, df, column, categories):
        log().debug(f"Encoding column: {column}, categories: {categories}")
        ohe = OneHotEncoder(sparse=False, categories=[categories], handle_unknown="ignore")

        encode_columns = [column]
        ohe.fit(df[encode_columns])
        encoded = pd.DataFrame(ohe.transform(df[encode_columns]), columns=categories, dtype=int)
        xdf = pd.concat([df.drop(encode_columns, 1), encoded], axis=1).reindex()

        # Add prefix based upon original column name to all new columns
        name_map = {x: column + "_" + x for x in categories}
        xdf = xdf.rename(columns=name_map)

        return xdf

    def _filter_graph(self, graph, filtered_words):
        for idx in list(graph.nodes):
            if idx not in self._tokens_by_idx.keys():
                log().debug(f"filter idx: {idx} not in dictionary")
                log().debug(f"filter idx: {idx} not in dictionary")
                continue
            token = self._tokens_by_idx[idx]

            # log().debug(f"consuming token: {token['lemma']}")
            should_filter, reason = self._should_filter_token(token, filtered_words)
            if should_filter:
                self_reference = self._is_important(token)
                # log().debug(f"filtering word: {token['lemma']} self_reference: {self_reference} reason: {reason}")
                prune_node(graph, idx, self_reference=self_reference)

    def _filter_common(self, documents, keep=0.90):
        """
        Return a list of terms to filter which are considered less relevant based upon TFIDF analysis
        """
        tfidf = self._calc_tfidf(documents)
        tfidf_items = list(tfidf.items())
        tfidf_items.sort(key=lambda x: x[1], reverse=True)
        # log().debug(f"tfidf_items: {tfidf_items}")

        num = int(len(tfidf_items) * keep)
        filtered_scores = tfidf_items[num:]
        # log().debug(f"filtered_scores: {filtered_scores}")

        output = ""
        output += "\n --- Filtered Words (Common): (keep: {}, {} of total words: {}) ---".format(keep, len(filtered_scores), len(tfidf_items))
        for (word, score) in filtered_scores:
            output += "\n {:>30}:{:0.4f}".format(word, score)
        log().debug(output)

        filtered_words = [term for (term, score) in filtered_scores]

        return filtered_words, tfidf

    def _filter_extremes(self, documents):
        """
        Return list of terms in the document considered extreme/outliers (gensim functionality)
        """

        # log().debug("_filter_extremes documents: {}".format(documents))

        documents = [x.split() for document in documents for x in document]
        # log().debug("1. documents: {}".format(documents))
        # documents = documents.split()
        # log().debug("2. documents: {}".format(documents))

        from gensim import corpora
        original = corpora.Dictionary(documents)
        dct = corpora.Dictionary(documents)
        ntotal_words = len(dct.keys())

        # log().debug("1. dct: {}: {}".format(len(dct), dct))
        no_below = FILTER_PRESENT_MIN_DOCS  # HIGHER absolute number filter/remove MORE words
        no_above = FILTER_PRESENT_MAX_DOCS  # HIGHER pct filter/remove FEWER words
        dct.filter_extremes(no_below=no_below, no_above=no_above)
        # dct.filter_extremes(no_below=3, no_above=0.4)
        log().debug(f"Extreme Filter Dictionary: {len(dct)}: {dct}")

        filtered_words = [original[x] for x in original.keys() if x not in dct.keys()]
        if not filtered_words:
            log().debug("No extreme filtered words")
            return filtered_words

        nextreme_words = len(filtered_words)

        lens = [len(i) for i in filtered_words]
        avglens = np.mean(lens)
        stdlens = np.std(lens)
        log().debug(f"Candidate Filtered Words: num: {len(filtered_words)} length avg: {avglens} std: {stdlens}")

        filtered_words = [x for x in filtered_words if len(x) <= FILTER_BIG_WORD_LENGTH]
        lens = [len(i) for i in filtered_words]
        avglens = np.mean(lens)
        stdlens = np.std(lens)
        log().debug(f"Candidate Filtered Words (kept only bigger words): num: {len(filtered_words)} length avg: {avglens} std: {stdlens}")

        filtered_words = [x for x in filtered_words if x.split(":")[0] not in CONSUMED_TAGS]
        lens = [len(i) for i in filtered_words]
        avglens = np.mean(lens)
        stdlens = np.std(lens)
        log().debug(f"Final Filtered Words (keep consumable POS): num: {len(filtered_words)} length avg: {avglens} std: {stdlens}")

        nremoved_words = len(filtered_words)
        nkept_words = ntotal_words - nremoved_words

        filtered_words = sorted(filtered_words)
        lens = [len(i) for i in filtered_words]
        avglens = np.mean(lens)
        stdlens = np.std(lens)

        output = ""
        output += "\n--- Filtered Words (Extremes) total: {} kept: {} extremes: {} removed: {} present min: {} present max: {} ---".format(
            ntotal_words, nkept_words, nextreme_words, nremoved_words, no_below, no_above)
        for word in filtered_words:
            output += "\n " + word
        log().debug(output)

        return filtered_words

    def _find_filtered(self):
        # Find filtered words (note: TFIDF removes most common words which may not be useful for clustering...)
        # NOTE: input is a list of list of terms separated by spaces (see example below):
        # [['this is a test document.'], ['this is another document.']]

        documents = []
        for docid in self._tokens_by_document.keys():
            document = []
            for token in self._tokens_by_document[docid]:
                document.append(token["label"])
            documents.append(document)
        # log().debug(f"TFID documents: {len(documents)} {documents}")

        filtered_tfidf, tfidf = self._filter_common(documents, keep=TFIDF_KEEP_PCT)
        # log().debug(f"filtered_tfidf: {filtered_tfidf}")
        # log().debug(f"tfidf: {tfidf}")
        filtered_extremes = self._filter_extremes(documents)
        candidate_words = filtered_tfidf + filtered_extremes

        # Replace any filtered words that mistakenly came with punctuation
        candidate_words = " ".join(candidate_words)
        candidate_words = re.sub(r"\.", "", candidate_words)
        candidate_words = re.sub(r"\?", "", candidate_words)
        candidate_words = re.sub(r"\;", "", candidate_words)

        candidate_words = candidate_words.split()

        output = ""
        output += "\n --- Candidate Filtered Words (Common): {} ---".format(len(candidate_words))
        for word in candidate_words:
            output += f"\n {word}"
        log().debug(output)

        exception_words = [word for word in candidate_words if word.split(":")[0] in CONSUMED_TAGS and len(word.split(":")[1]) >= MIN_FILTERED_WORD_LEN]
        output = ""
        output += "\n --- Exception Filtered Words: {} ---".format(len(exception_words))
        for word in exception_words:
            output += f"\n {word}"
        log().debug(output)

        filtered_words = [word for word in candidate_words if word not in exception_words]
        output = ""
        output += "\n --- Final Filtered Words (Common): {} ---".format(len(filtered_words))
        for word in filtered_words:
            output += f"\n {word}"
        log().debug(output)

        return filtered_words

    def _find_phrases(self, docs):
        """
        Return a dictionary of phrases (key: text phrase) from the input document
        """

        chunks = {}
        list_docs = list(docs)
        for doc in list_docs:
            ncs = list(doc.noun_chunks)
            for nc in ncs:
                if " " not in nc.text:
                    continue
                items = []
                for token in nc:
                    if token.pos_.lower() not in CONSUMED_TAGS:
                        continue
                    value = token.lemma_
                    items.append(value)
                if len(items) <= 1:
                    continue

                if len(items) > XGRAM_THRESHOLD:
                    continue

                text = " ".join(items)
                if text not in chunks:
                    chunks[text] = 0
                chunks[text] += 1

        phrases = {}
        for text in chunks.keys():
            count = chunks[text]
            if count >= XGRAM_FREQUENCY_CUTOFF:
                phrases[text] = count

        return phrases

    def _format_phrases(self, phrases):
        """
        Return phrases in human readable format
        """

        output = ""
        output += "\n --- Useful Phrases ---"
        for item in phrases:
            (phrase, count) = item
            output += "\n {:5} {}".format(count, phrase)
        return output

    def _get_dictionary(self, doc):
        """
        Return a token dictionary for a document
        """

        token_dictionary = {x["idx"]: x for x in doc}
        return token_dictionary

    def _label_data(self, docs):
        """
        Return documents where original document terms have been replaced by custom label tokens
        """

        documents = []
        for doc in docs:
            document = []
            for token in doc:
                label = self._create_label(token)
                if label:
                    document.append(label)
            documents.append(document)
        return documents

    def _parse_item(self, depth, edges, parent, token):
        """
        Recursively parse an item and create edges
        """

        # log().debug(f"Parsing item, parent: {parent} token: {token}")
        if parent and token:
            edge = (parent["idx"], token["idx"])
            # log().debug(f"Creating edge: parent: {parent['label']}/{parent['text']}/{parent['idx']} token: {parent['label']}/{parent['text']}/{parent['idx']}")
            edges.append(edge)

        # if depth > 50:
        #     log().debug(f"Max depth, depth: {depth} edge: par: {parent['label']}/{parent['text']}/{parent['idx']} tok: {parent['label']}/{parent['text']}/{parent['idx']}")
        #     return False

        for idx in token["children"]:
            if idx not in self._tokens_by_idx:
                log().debug(f"Unknown child idx: {idx} for token: {token['label']}")
                continue
            child = self._tokens_by_idx[idx]

            if token["idx"] == child["idx"]:
                log().debug(f"token: {['idx']} same as child: {child['idx']}")
                # log().debug(f"token: {token['label']}/{token['text']}/{token['idx']}")
                # log().debug(f"children: {token['children']}")
                continue

            depth += 1
            self._parse_item(depth, edges, token, child)

    def _parse_sentence(self, root):
        """
        Return the graph for an individual sentence (defined by its root)
        """

        edges = []
        # log().debug(f"Parsing sentence, root: {root['label']}/{root['text']}/{root['idx']}")
        depth = 0
        parent = None

        self._parse_item(depth, edges, parent, root)

        graph = create_graph(edges)
        return graph

    def _pipeline(self, doc):
        # This is a test version of a pipeline - it gets called for every document
        # Examples can be found at: https://spacy.io/usage/processing-pipelines
        return doc

    def _process_documents(self, docs):
        """
        Return a list of processed documents where each document is composed of a list of tokens
        """

        log().info(f"Processing tokens from documents: {len(docs)}")

        # Capture the sentiments
        log().debug(f"Processing sentiments")
        tsentiments = {}
        ssentiments = {}
        dsentiments = {}
        for i, doc in enumerate(docs):
            document = []
            # Calculate the document sentiment

            didx = calc_hash(doc.text)
            if didx not in dsentiments.keys():
                dsentiment = calc_sentiment(doc.text)
                dsentiments[didx] = dsentiment
                # log().debug(f"doc text: {doc.text} dsentiment: {dsentiment}")

            if i % 100 == 0:
                log().debug(f"Processing sentiments for document: {i}")

            for j, sent in enumerate(doc.sents):

                # Calculate the sentence sentiment
                sidx = calc_hash(sent.text)
                if sidx not in ssentiments.keys():
                    ssentiment = calc_sentiment(sent.text)
                    ssentiments[sidx] = ssentiment
                    # log().debug(f"sent text: {sent.text} ssentiment: {ssentiment}")

                for k, token in enumerate(sent):
                    if token.text == "" or token.text == " ":
                        log().debug(f"Token text is empty: {token}")
                        continue

                    if len(token.text) >= MAX_TOKEN_LENGTH:
                        log().debug(f"Token too long: {token.text}, len: {len(token.text)}")
                        continue

                    tidx = calc_hash(token.text)
                    if tidx not in tsentiments.keys():
                        tsentiment = calc_sentiment(token.text)
                        tsentiments[tidx] = tsentiment

        self._tsentiments = tsentiments
        self._ssentiments = ssentiments
        self._dsentiments = dsentiments

        # Process the document tokens
        oov_terms = []
        documents = []
        for i, doc in enumerate(docs):
            document = []

            if i % 100 == 0:
                log().debug(f"Processing tokens for document: {i}")

            for j, sent in enumerate(doc.sents):

                if self._should_filter_sentence(sent):
                    log().debug(f"Invalid sentence: {sent}")
                    continue
                for k, token in enumerate(sent):
                    if token.text == "" or token.text == " ":
                        log().debug(f"Token text is empty: {token}")
                        continue

                    if len(token.text) >= 50:
                        log().debug(f"Token too long: {token.text}, len: {len(token.text)}")
                        continue

                    x = self._create_token(token)
                    document.append(x)
                    if x["is_oov"]:
                        oov_terms.append(x["label"])
            documents.append(document)

        # Create a token dictionary (by idx) to allow children indexing
        log().debug(f"Creating token dictionary")
        token_dict = {}
        for document in documents:
            for token in document:
                spacy_token = token["spacy_token"]
                spacy_doc = str(spacy_token.doc[0:25])
                key = spacy_doc + "." + str(spacy_token.idx)
                key = calc_hash(key)
                # log().debug(f"key: {key}")
                token_dict[key] = token

        # log().debug(f"Token dictionary: {token_dict}")

        # Assign children to tokens
        log().debug(f"Assigning children tokens, tokens: {len(token_dict.keys())}")
        for token_key in token_dict.keys():
            token = token_dict[token_key]
            spacy_token = token["spacy_token"]
            for child in spacy_token.children:
                child_doc = str(spacy_token.doc[0:25])
                key = child_doc + "." + str(child.idx)
                key = calc_hash(key)
                if key not in token_dict:
                    log().debug(f"Could not find key: {key} child doc: {child.doc[0:20]} child idx: {child.idx}")
                    continue
                child_token = token_dict[key]
                token["children"].append(child_token["idx"])
                # log().debug(f"token {token['idx']} has child_token: {child_token['idx']}")

        count = Counter(oov_terms)
        log().debug(f"Remaining OOV terms: {count.most_common()}")

        log().info(f"Processing tokens complete")

        return documents

    def _reconcile_children(self, documents):
        """
        Return a list of documents where individual tokens have been replaced and reconciled back to document token order
        """

        # With phrases now included in the document, each token's children need to be
        # reconciled to the new phrase-tokens.  Add all tokens except where they are
        # a phrase, in which case replace the composition elements with the phrase

        # rdocuments = []
        for document in documents:
            token_dictionary = {x["idx"]: x for x in document}
            for token in document:
                for idx in token["children"]:
                    if idx not in token_dictionary.keys():
                        token["children"].remove(idx)

        return documents

    def _reconcile_phrases(self, documents, phrases):
        """
        Return a list of documents where individual tokens have been replaced by phrases
        """

        # Sort the phrases from the largest to the smallest - this will stop
        # smaller phrases with common words as larger ones from trampling each other
        phrases = sorted(phrases, key=lambda kv: kv[1], reverse=False)

        rdocuments = []
        for document in documents:
            # log().debug("--- current")
            # show_dictionary(document)

            rdocument = [x for x in document]
            lemmas = [x["lemma"] for x in rdocument]

            for item in phrases:
                (phrase, count) = item
                ptokens = phrase.split(" ")
                ntokens = len(ptokens)

                for i, lemma in enumerate(list(lemmas)):
                    start = i
                    end = start + ntokens
                    lphrase = " ".join(lemmas[start:end])
                    if lphrase == phrase:
                        beginning = rdocument[:start]
                        ending = rdocument[end:]
                        phrase_token = self._create_phrase_token(rdocument, start, end - 1)
                        rdocument = []
                        # log().debug("1. rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))
                        rdocument.extend(beginning)
                        # log().debug("2. rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))
                        rdocument.append(phrase_token)
                        # log().debug("3. rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))
                        rdocument.extend(ending)
                        # log().debug("4. rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))

                        # log().debug("beginning: {}".format("|".join([x["lemma"] for x in beginning])))
                        # log().debug("phrase_token: {}".format(phrase_token["lemma"]))
                        # log().debug("ending: {}".format("|".join([x["lemma"] for x in ending])))
                        # log().debug("NEW rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))
                        lemmas = [x["lemma"] for x in rdocument]

            # log().debug("--- revised")
            # show_dictionary(rdocument)

            rdocuments.append(rdocument)

        return rdocuments

    def _redim_vectors(self, features, components=2):
        data = []
        for feature in features:
            row = []
            for col in feature.keys():
                if col.startswith("v-"):
                    value = feature[col]
                    row.append(value)
            data.append(row)
        data = np.array(data)
        # TODO: should create common utilities
        data = redim_data(data, components=components)
        return data

    def _show_phrases(self, phrases):
        """
        Log phrases in human readable format
        """
        log().debug(self._format_phrases(phrases))

    def _should_filter_sentence(self, sentence):
        """
        Return True if the sentence should be filters (criteria: too many tokens, which usually means a
        poorly parsed document).  Otherwise return False
        """

        if len(sentence) > MAX_TOKENS_IN_SENTENCE:
            return True
        return False

    def _should_filter_token(self, token, filtered_words):
        """
        Return True if the token should be filtered, indicating it is not relevant
        """
        reason = None

        # # Keep roots
        # if token["is_root"]:
        #     return False, reason

        # Filter stop words
        if token["is_stop"]:
            reason = "stop word"
            return True, reason

        # Filter bogus labels
        if ":" not in token["label"]:
            reason = "invalid label, does not contain ':'"
            return True, reason

        # Filter bogus lemma
        if len(token["lemma"]) == 1:
            reason = "lemma too short"
            return True, reason

        # Filter punctuation
        if token["is_punct"]:
            reason = "is punctation"
            return True, reason

        # Filter "who/what..." etc
        if "wp" in token["tag_"]:
            reason = "tag contains wp"
            return True, reason

        # Filter spaces
        if "_sp" in token["tag_"]:
            reason = "tag contains _sp"
            return True, reason

        # Filter pronouns
        if token["lemma"] == "-PRON-":
            reason = "lemma is -PRON-"
            return True, reason

        # Must be a consumed tag
        if token["pos_"] not in CONSUMED_TAGS:
            reason = f"pos_ {token['pos_']} not in consumed tags: {CONSUMED_TAGS}"
            return True, reason

        # Filter quotes
        if token["text"] == "’" or token["lemma"] == "’":
            reason = "text or lemma is quote string"
            return True, reason

        label = token["label"]
        if label in filtered_words:
            reason = f"label: {label} is in filtered words: {filtered_words}"
            return True, reason

        return False, reason

    def _is_important(self, token):
        """
        Return True if this toke is considered important
        Note: Important tokens are consummable tags
        """
        if token["pos_"] in CONSUMED_TAGS:
            return False


##########
#
# Utility Functions - Initialization
#
##########
def init():
    """
    Initialize random state and logs
    """
    np.set_printoptions(threshold=100)
    np.set_printoptions(precision=4)

    pd.set_option('display.max_rows', 10000)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 150)

    ###########
    #
    # NOTE: in an attempt to get reproducible models, all of the random seeds (python, numpy, tensorflow, etc)
    # need to be set each time the model is to be trained... otherwise the model provides materially
    # different accuracy/model and hence different embeddings
    # NOTE: even doing this, the embeddings are occasionally quite close (but still a tiny bit different)
    #
    ###########

    # 1. Set PYTHONHASHSEED environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # 2. Set python built-in pseudo-random generator at a fixed value
    random.seed(RANDOM_SEED)

    # 3. Set numpy pseudo-random generator at a fixed value
    np.random.seed(RANDOM_SEED)

    # 4. Set tensorflow pseudo-random generator at a fixed value
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(RANDOM_SEED)

    # 5. Configure a new global tensorflow session
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Establish logger, levels: "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
    level = "INFO"
    logging.config.dictConfig(configure_log(level))


##########
#
# Utility Functions - General
#
##########
def calc_hash(s):
    """
    Return the hash value for a string
    Note: different algos were used for this, hence encapculating
    a relatively simple function
    """
    # s = s.encode('utf-8')
    idx = hash(s)
    return idx


def configure_log(level):
    """
    Return the log configuration
    """

    cfg = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "datefmt": "%I:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "level": level,
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": level,
            },
        }
    }
    return cfg


def filelist(directory):
    """
    Return list of files in a directory
    """

    items = []
    for root, subdirs, files in os.walk(directory):
        fqpath = [os.path.join(root, x) for x in files]
        items.extend(fqpath)
    return items


def log(name="clusters"):
    """
    Return logger
    """
    logger = logging.getLogger(name)
    return logger


def pretty(x):
    """
    Return human readable view of data
    """

    pp = pprint.PrettyPrinter(width=200, compact=True)
    return pp.pformat(x)


def scan(embeddings, metric="euclidean", size=5, samples=1):
    """
    Return DBSCAN data for embeddings
    """
    dbscan = hdbscan.HDBSCAN(
        metric=metric,
        min_cluster_size=size,
        min_samples=samples
    ).fit(embeddings)
    return dbscan


def show_topics(data, num_rows=25, summary=True):
    """
    Log formatted attributes of topics
    """

    output = f"\n--- TOPIC SUMMARY ---"
    output += "\n{:7} {}".format("TOPIC", "PROB")
    for cluster in data.keys():
        items = data[cluster]
        output += f"\n{cluster:7}"
        for item in items:
            # print(f"item: {item}")
            output += f" {item['label']}*{float(item['c_prob']):0.3f}"

    output += f"\n\n--- TOPIC DETAIL ---"
    for cluster in data.keys():
        items = data[cluster]
        output += f"\n\n TOPIC: {cluster} (items: {len(items)}, showing first {num_rows} rows for each cluster)"
        output += "\n{:20} {:6} {:8}".format("LABEL", "DEGREE", "PROB")
        for i, item in enumerate(items):
            output += "\n{:20} {:6} {:0.6f}".format(item["label"], item["degree"], item["c_prob"])
            if i > num_rows:
                break

    log().info(output)


def show_document_tokens(documents, num_docs=25):
    """
    Log formatted attributes of document tokens
    """

    output = ""
    output += f"\n --- DOCUMENTS: {len(documents)} (showing: {num_docs} docs) ---"
    for i, doc_idx in enumerate(documents.keys()):
        document = documents[doc_idx]
        # print(f"document: {document}")

        output += f"\n - DOCUMENT (doc number: {doc_idx}) -"
        output += "\n {:20} {:5} {:5} {:5} {:5} {:5} {:5} {:15} {:15} {:5} {}".format(
            "LABEL", "POS", "TAG", "DEP", "OOV", "PUNCT", "ROOT", "LEMMA", "TEXT", "IDX", "CHILDREN")
        for token in document:
            idx = token["idx"]
            # idx = idx.split("-")[-1]
            idx = idx[0:4]

            children = []
            for child in token["children"]:
                # child = child.split("-")[-1]
                child = child[0:4]
                children.append(child)

            output += "\n {:20} {:5} {:5} {:5} {:5} {:5} {:5} {:15} {:15} {:5} {}".format(
                token["label"], token["pos_"], token["tag_"], token["dep_"][0:5], str(token["is_oov"]), str(token["is_punct"]), str(token["is_root"]),
                token["lemma"], token["text"], idx, children)
        if i >= num_docs:
            break
    log().info(output)
    return output


def show_document_topics(data, documents, num_docs=25):
    """
    Log formatted attributes of document topics
    """

    first_doc = list(data.keys())[0]
    first_item = data[first_doc][0]
    num_probs = len(first_item["y_prob"])
    # print(f"first_item: {first_item} num_probs: {num_probs}")
    prob_hdr = " ".join(f"{x}".center(6, "-") for x in range(num_probs))

    output = ""
    output += f"\n --- DOCUMENTS: {len(data.keys())} (showing: {num_docs} docs) ---"
    for i, doc_idx in enumerate(data.keys()):
        document = data[doc_idx]
        # print(f"document: {document}")

        output += f"\n\nDOCUMENT (doc number: {doc_idx}) -"
        dstr = documents[doc_idx]
        if len(dstr) > 100:
            dstr = dstr + f"... ({len(dstr) - 100} more chars)"
        output += f"\n{dstr}"
        output += "\n{:20} {:>6} {:>6} {}".format("", "", "", "CLUSTER-PROBABILITIES")
        output += "\n{:20} {:>6} {:>6} {}".format("LABEL", "PRED", "PROB", prob_hdr)
        y_probs = []
        c_probs = []
        tmp_outputs = []  # Need to capture the detail output but it needs to be rendered after mean ouput
        for item in document:
            # print(f"item: {item}")
            s_probs = " ".join("{:0.4f}".format(x) for x in item["y_prob"])
            y_pred = item["y_pred"]
            y_prob = item["y_prob"]
            c_prob = item["y_prob"][y_pred]
            c_probs.append(c_prob)
            y_probs.append(y_prob)
            tmp_outputs += "\n  {:18} {:6} {:0.4f} {}".format(item["label"], y_pred, c_prob, s_probs)

        m_probs = np.mean(y_probs, axis=0)
        m_pred = int(np.argmax(m_probs))
        mc_probs = m_probs[m_pred]
        sm_probs = " ".join("{:0.4f}".format(x) for x in list(m_probs))
        output += "\n{:20} {:6} {:0.4f} {}".format("MEAN", m_pred, mc_probs, sm_probs)
        output += "".join(tmp_outputs)

        if i >= num_docs:
            break
    log().info(output)
    return output


def show_graph(graph, title="Graph", show_details=True, extended=False, rows=5):
    """
    Log formatted attributes of a graph
    """

    num_nodes, num_edges, details = calc_graph_stats(graph)
    degrees = [x["degree"] for x in details]

    output = ""

    output += "\n --- {} (showing: {}) ({} nodes, {} edges, degrees: {}) ---".format(title, rows, num_nodes, num_edges, sum(degrees))
    if not show_details:
        return output

    output += "\n {:30} {:5} {:>8} {}".format("NODE", "DATA", "EDGES", "EDGE-NODES")

    details = sorted(details, key=lambda x: x["degree"], reverse=True)
    for i, detail in enumerate(details):
        label = detail["label"]
        data = detail["data"]
        # degree = detail["degree"]
        edges = detail["edges"]

        if i < rows:
            # Sort the edges from closest (min distance, more important) to furthest (max distance, less important)
            linked_nodes = [p2 for (p1, p2) in edges]
            linked_nodes = sorted(linked_nodes)

            linked_nodes_data = linked_nodes
            if not extended:
                linked_nodes_count = Counter(linked_nodes)
                linked_nodes_data = linked_nodes_count.most_common()[0:25]

            output += "\n {:30} {:5} {:8} {}".format(label, str(data), len(linked_nodes), linked_nodes_data)

    stddeg = 0
    avgdeg = 1
    if len(degrees) != 0:
        stddeg = statistics.stdev(degrees)
        avgdeg = statistics.mean(degrees)

    output += "\n {:30} {:5} {:8.1f} {}".format("AVG", " ", avgdeg, " ")
    output += "\n {:30} {:5} {:8.1f} {}".format("STD", " ", stddeg, " ")

    log().info(output)


def show_words(data):
    """
    Log formatted view of word data
    """

    k = list(data.keys())[0]
    num_probs = len(data[k]["y_prob"])
    prob_hdr = " ".join(f"{x}".center(6, "-") for x in range(num_probs))

    rows = len(data.keys())
    output = f"\n--- Words (total: {rows})"
    output += "\n{:20} {:6} {:4} {:>6} {}".format("LABEL", "DEGREE", "PRED", "PROB", "CLUSTER-PROBABILITIES")
    output += "\n{:20} {:6} {:4} {:>6} {}".format("", "", "", "", prob_hdr)
    sorted_labels = sorted(data.keys())
    for label in sorted_labels:
        item = data[label]
        c = int(item["y_pred"])
        c_prob = item["y_prob"][c]
        s_probs = " ".join("{:0.4f}".format(x) for x in item["y_prob"])
        output += "\n{:20} {:6} {:4} {:0.4f} {}".format(
            label, item["degree"], item["y_pred"], c_prob, s_probs)
    log().info(output)


def save_model(graph, labels, features, feature_names, embeddings, clusters, model_file):
    """
    Save a model (which includes its graph, features, embeddings, and clusters)
    """

    edges = graph.edges(data=True)

    data = {
        "labels": labels,
        "feature_names": feature_names,
        "features": features,
        "edges": edges,
        "embeddings": embeddings,
        "clusters": clusters
    }

    with open(model_file, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_model(model_file):
    """
    Load a model and return its graph, labels, features, embeddings, and clusters
    """

    data = None
    with open(model_file, "rb") as f:
        data = pickle.load(f)

    labels = data["labels"]
    feature_names = data["feature_names"]
    edges = data["edges"]
    features = data["features"]
    embeddings = data["embeddings"]
    clusters = data["clusters"]

    graph = create_graph(edges)

    print(f"graph: {nx.info(graph)}")
    print(f"labels: {labels}")
    print(f"feature_names: {feature_names}")
    print(f"features: {features[0]}")
    print(f"clusters: {clusters}")

    return graph, labels, features, embeddings, clusters


##########
#
# Utility Functions - Graphs
#
##########
def calc_centers(xy, labels, degrees):
    """
    Return weighted cluster centers
    """
    items = {}
    for i, x in enumerate(labels):
        emb = [xy[i] for j in range(0, degrees[i])]
        if x not in items:
            items[x] = []
        items[x].extend(emb)
    centers = {}
    for x in items.keys():
        avg = np.mean(items[x], axis=0)
        centers[x] = avg
    return centers


def calc_clusters(embeddings, method="KMEANS"):
    """
    Return cluster labels (each label is assigned a cluster)
    """

    # NOTE: the clusters are identified close to identical with some identifiable differences
    # when using raw or dim reduced embeddings... use raw embeddings (this takes more computing cycles)
    cluster_labels = None
    if method == "KMEANS":
        cluster_labels = kmeans_hyperparameters(embeddings)
    elif method == "HDBSCAN":
        cluster_labels = dbscan_hyperparameters(embeddings)
        # Note that hdbscan finds unknown clusters with index -1, and indexes start at 0
        # To allow cluster labels to be used as indexes, add 1 to then.
        # This means that the "unknown" cluster is 0, and all real clusters begin at 1.
        cluster_labels += 1
    if len(cluster_labels) <= 1:
        return
    return cluster_labels


def calc_distance(p1, p2):
    """
    Calculate distance between two points
    """
    from scipy.spatial import distance
    p = [p1]
    vectors = [p2]
    x = distance.cdist(p, vectors)
    x = x[0][0]
    return x


def calc_graph_stats(graph):
    """
    Return calculated statistics for a graph
    """

    # cluster_coefficient = nx.average_clustering(graph)
    items = []
    degrees = {node: val for (node, val) in graph.degree()}

    for node in graph.nodes(data=True):
        label = node[0]
        data = node[1]
        degree = degrees[label]

        edges = graph.edges([label])

        item = (label, str(data), degree, edges)
        items.append(item)

    # Sort the data by the number of edges (idx: 2)
    sorted_nodes = sorted(items, key=lambda x: x[2], reverse=True)
    num_nodes = len(sorted_nodes)
    num_edges = len(graph.edges)

    details = []
    for node in sorted_nodes:
        detail = {}
        label = node[0]
        data = node[1]
        degree = node[2]
        edges = node[3]

        detail["label"] = label
        detail["data"] = data
        detail["degree"] = degree
        detail["edges"] = edges
        details.append(detail)

    return num_nodes, num_edges, details


def calc_metrics(graph):
    """
    Calculate key graph metrics and return pagerank, hits, authorities, degree centrality
    and eigenvector centrality
    """

    # Calculate node attributes ... note: pagerank and hists take a LONg time - ignore them for now
    log().info(f"Calcuating ranks")
    ranks = np.zeros(len(list(graph.nodes())))
    # ranks = nx.pagerank_numpy(graph)
    # log().debug(f"ranks: {ranks}")
    log().info(f"Calcuating ranks complete")

    log().info(f"Calcuating authorities")
    hits = np.zeros(len(list(graph.nodes())))
    authorities = np.zeros(len(list(graph.nodes())))
    # hits, authorities = nx.hits_numpy(graph)
    # log().debug(f"hits: {hits}")
    # log().debug(f"authorities: {authorities}")
    log().info(f"Calcuating authorities complete")

    log().info(f"Calcuating degree_centrality")
    degree_centrality = nx.degree_centrality(graph)  # quick
    # log().debug(f"degree_centrality: {degree_centrality}")
    log().info(f"Calcuating degree_centrality complete")

    # eigenvector_centrality = nx.eigenvector_centrality(graph)  # quick, but does not work with multi-graph
    # log().debug(f"eigenvector_centrality: {eigenvector_centrality}")

    # closeness_centrality = nx.closeness_centrality(graph)  # long time
    # log().debug(f"closeness_centrality: {closeness_centrality}")

    # current_flow_betweenness_centrality = nx.current_flow_betweenness_centrality(graph)  # very, very, long time
    # log().debug(f"current_flow_betweenness_centrality: {current_flow_betweenness_centrality}")

    # return ranks, hits, authorities, degree_centrality, eigenvector_centrality
    return ranks, hits, authorities, degree_centrality


def calc_probability(d, m=2):
    """
    Return the probability for a distance.
    Note: An exponential probability
    curve (full area == 1.0) is used to calculate probabilities.  A data point
    at zero distance will have a 1.0 probability and a point at the max distance
    will have a very small (about 0.000001) probability.  Max distance is (mean +/- 3*std).
    """

    # p = m * math.exp((-1 * m * (d)))
    p = math.exp(-1 * m * d)
    return p


def create_graph(edges, graph_type="graph"):
    """
    Return a graph which is created from the input edges; duplicate edges have distance modified to be closer (more relevance)
    """

    # Build the base graph
    graph = None
    if graph_type == "graph":
        graph = nx.Graph()
    elif graph_type == "directedgraph":
        graph = nx.DiGraph()
    elif graph_type == "multigraph":
        graph = nx.MultiGraph()

    graph.add_edges_from(edges)
    return graph


def merge_graph(graph, new_edges):

    # Eliminate for duplicate edges and only consider new edges
    x = []
    for edge in new_edges:
        (a, b) = edge
        if graph.has_edge(a, b):
            # log().debug(f"Edge exists: {edge}")
            continue
        if graph.has_edge(b, a):
            # log().debug(f"Edge exists (reverse): {edge}")
            continue
        x.append(edge)
    new_edges = x

    # Find the merged edges from the existing graph and new edges
    existing_edges = set(graph.edges())
    new_edges = set(new_edges)
    new_edges_not_existing_edges = new_edges - existing_edges
    merged_edges = list(existing_edges) + list(new_edges_not_existing_edges)

    # Create a new graph from the merged edges
    xgraph = nx.Graph()
    xgraph.add_edges_from(merged_edges)

    # Determine new nodes
    new_graph = nx.MultiGraph()
    new_graph.add_edges_from(new_edges)
    new_nodes = set(new_graph.nodes())
    existing_nodes = set(graph.nodes())
    new_nodes_not_existing_nodes = new_nodes - existing_nodes
    # merged_nodes = list(existing_nodes) + list(new_nodes_not_existing_nodes)

    # log().debug(f"existing_nodes: {len(existing_nodes)} {existing_nodes}")
    # log().debug(f"new_nodes: {len(new_nodes)} {new_nodes}")
    # log().debug(f"new_nodes_not_existing_nodes: {len(new_nodes_not_existing_nodes)} {new_nodes_not_existing_nodes}")
    # log().debug(f"merged_nodes: {len(merged_nodes)} {merged_nodes}")

    # log().debug(f"existing_edges: {existing_edges}")
    # log().debug(f"new_edges: {new_edges}")
    # log().debug(f"new_edges_not_existing_edges: {new_edges_not_existing_edges}")
    # log().debug(f"merged_edges: {merged_edges}")

    return xgraph, sorted(list(new_nodes_not_existing_nodes)), sorted(list(new_edges_not_existing_edges))


def plot(embeddings, clusters, labels):
    """
    Plot the data on a 2D chart.  Note that raw embeddings were re-dimensioned to 2D.
    This will provide a general view of the clusters - note that the plotting
    is on the reduced dimensional data so the clustering visualization may be
    a bit inaccurate.
    """
    emb = redim_data(embeddings, components=2, transformation=TRANSFORMATION)
    x = [xx[0] for xx in emb]
    y = [yy[1] for yy in emb]
    plot_2D(x, y, clusters, labels)


def plot_2D(x, y, clusters, labels):
    """
    Plot dataframe in 2D
    """

    # log().debug(f"plot x: {len(x)} {x}")
    # log().debug(f"plot y: {len(y)} {y}")
    # log().debug(f"plot clusters: {len(clusters)} {clusters}")
    # log().debug(f"plot labels: {len(labels)} {labels}")

    import plotly
    import plotly.graph_objs as go

    gap = max(x) - min(x)
    pix_width = 800
    pix_height = 800
    pix_avg = (pix_width + pix_height) / 2
    pix_per_unit = pix_avg / gap

    gap = max(x) - min(x)
    pix_width = 800
    pix_height = 800
    pix_avg = (pix_width + pix_height) / 2
    pix_per_unit = pix_avg / gap
    size = degrees
    sizemin = 1
    sizemode = "area"
    size_max = max(size)
    sizeref = 0.005 * size_max / pix_per_unit  # larger numbers give smaller bubbles
    colorscale = "Viridis"

    text = [str(i) + "/" + label + "/" + str(clusters[i]) for i, label in enumerate(labels)]
    trace1 = go.Scatter(x=x, y=y, text=text, mode='markers',
                        marker=dict(
                            size=size,
                            sizemode=sizemode,
                            sizemin=sizemin,
                            sizeref=sizeref,
                            color=clusters,
                            colorscale=colorscale,
                            line=dict(color="rgb(150, 150, 150)")
                        ))
    data = [trace1]
    title = "Graph"
    layout = go.Layout(height=800, width=800, title=title, hovermode="closest")

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)


def prune_node(graph, node, self_reference=False):
    """
    Prune a node from a graph: find neighbours, remove node, and reconnect neighbours
    """

    # This is a regular graph, not a DiGraphc (directional), hence neighbors
    # will be successors and predecessors
    neighbors = list(graph.neighbors(node))
    new_edges = [(x, y) for x in neighbors for y in neighbors if x != y and x != node and y != node]
    if self_reference:
        if len(new_edges) == 0:
            new_edges = [(x, y) for x in neighbors for y in neighbors]
    graph.remove_node(node)
    graph.add_edges_from(new_edges)


##########
#
# Utility Functions - Machine Learning
#
##########
def calc_cluster_distances(centers_dict, coordinates):
    """
    Return the distances between the input coordinate and the input cluster centers
    """
    xdistances = collections.defaultdict(list)
    center_distances = []
    for cluster in centers_dict.keys():
        center = centers_dict[cluster]
        for coordinate in coordinates:
            xdistance = calc_distance(coordinate, center)
            xdistances[cluster].append(xdistance)
            center_distances.append(xdistance)
    return center_distances


def calc_sentiment(text):
    """
    Return the sentiment of the input text.  The input text could be a word, sentence, or document
    """
    from textblob import TextBlob
    tb_analysis = TextBlob(text)

    info = {
        "polarity": tb_analysis.polarity,
        "subjectivity": tb_analysis.subjectivity,
        "negative": 0,
        "neutral": 0,
        "positive": 0,
        "compound": 0,
    }
    return info

    # Additional sentiment for each word
    # NOTE: this takes a material duration for each word and hence
    # has been commented out and not used

    # from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    # analyzer = SentimentIntensityAnalyzer()
    # vs_analysis = analyzer.polarity_scores(text)
    # info = {
    #     "polarity": tb_analysis.polarity,
    #     "subjectivity": tb_analysis.subjectivity,
    #     "negative": vs_analysis["neg"],
    #     "neutral": vs_analysis["neu"],
    #     "positive": vs_analysis["pos"],
    #     "compound": vs_analysis["compound"],
    # }
    # return info


def create_distance_model(labels, embeddings, degrees, centers_dict):
    """
    Return a list of items, with each items containing data for a label/cluster
    """

    # Calculate the maximum distances to use for probability calculations.
    log().info(f"Calculating distances")
    cluster_distances = calc_cluster_distances(centers_dict, embeddings)
    # min_distance = min(cluster_distances)
    max_distance = max(cluster_distances)
    # median_distance = statistics.median(cluster_distances)
    mean_distance = statistics.mean(cluster_distances)
    std_distance = statistics.stdev(cluster_distances)
    nmax_distance = (mean_distance + (3 * std_distance)) - (mean_distance - (3 * std_distance))
    print(f"max_distance: {max_distance} nmax_distance: {nmax_distance}")

    data = []
    yprobabilities = []
    wprobabilities = []
    nprobabilities = []
    for i, coordinate in enumerate(embeddings):
        label = labels[i]
        degree = degrees[i]
        xprobabilities = []
        wprobabilities = []
        for cluster in centers_dict.keys():
            center = centers_dict[cluster]

            # Calculate the actual distance and then normalize it
            # based upon the normalized max distance (mean +/- std)
            xdistance = calc_distance(coordinate, center)
            ndistance = (xdistance / max_distance) * nmax_distance
            # xprobability = calc_probability(xdistance)
            xprobability = calc_probability(ndistance)
            # print(f"xdistance: {xdistance}, ndistance: {ndistance} xprobability: {xprobability} nprobability: {nprobability}")
            wprobability = xprobability * degree

            item = {
                "label": label, "cluster": cluster, "degree": degree, "xprobability": xprobability
            }
            data.append(item)
            xprobabilities.append(xprobability)
            wprobabilities.append(wprobability)
        yprob = xprobabilities / np.sum(xprobabilities)
        yprobabilities.extend(yprob)
        nprob = wprobabilities / np.sum(wprobabilities)
        nprobabilities.extend(nprob)

    for i, item in enumerate(data):
        item["nprobability"] = nprobabilities[i]

    return data


def create_embeddings(graph, features, labels):
    """
    Return the embeddings for the input graph and features
    """

    df = pd.DataFrame(features, index=labels)

    # Create the model and generators
    Gs = sg.StellarGraph(graph, node_features=df)
    unsupervisedSamples = UnsupervisedSampler(Gs, nodes=graph.nodes(), length=5, number_of_walks=3, seed=RANDOM_SEED)
    train_gen = GraphSAGELinkGenerator(Gs, 50, [5, 5], seed=RANDOM_SEED).flow(unsupervisedSamples)
    graphsage = GraphSAGE(layer_sizes=[100, 100], generator=train_gen, bias=True, dropout=0.0, normalize="l2")
    x_inp_src, x_out_src = graphsage.node_model()
    x_inp_dst, x_out_dst = graphsage.node_model()

    x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
    x_out = [x_out_src, x_out_dst]
    prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)

    # Create and train the Keras model
    model = keras.Model(inputs=x_inp, outputs=prediction)
    learning_rate = 1e-2
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy])

    _ = model.fit_generator(train_gen, epochs=EPOCHS, verbose=2, use_multiprocessing=False, workers=1, shuffle=False)

    # Create a generator that serves up nodes for use in embedding prediction / creation
    node_gen = GraphSAGENodeGenerator(Gs, 50, [5, 5], seed=RANDOM_SEED).flow(labels)
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    embeddings = embedding_model.predict_generator(node_gen, workers=4, verbose=2)
    embeddings = embeddings[:, 0, :]

    return embeddings


def dbscan_hyperparameters(embeddings):
    """
    Return the best score cluster labels
    Run dbscan clustering greedily for different min_samples and e_eps and
    discover number of resulting clusters and noise points
    """

    # Establish lower and upper cluster sizes
    m_lower = int(len(embeddings) * 0.01)
    if m_lower <= 3:
        m_lower = 5
    m_upper = int(m_lower * 5)
    if m_upper <= (5 * m_lower):
        m_upper = int(m_lower * 5)
    m_increment = int((m_upper - m_lower) / 20)
    if m_increment <= 0:
        m_increment = 1

    x_clusters = []
    x_noises = []
    x_metrics = []
    x_sizes = []
    x_samples = []
    x_scores = []
    x_pct_noises = []
    x_pct_cleans = []

    x_total = len(embeddings)
    iteration = 0
    metric_types = [x for x in hdbscan.dist_metrics.METRIC_MAPPING.keys()]
    metric_types.sort()
    metric_types.remove("arccos")       # TBD
    metric_types.remove("canberra")     # Uses distance from origin which does not create useful clusters
    metric_types.remove("cosine")       # TBD
    metric_types.remove("haversine")    # TBD
    metric_types.remove("mahalanobis")  # TBD
    metric_types.remove("minkowski")    # TBD
    metric_types.remove("pyfunc")       # TBD
    metric_types.remove("seuclidean")   # TBD
    metric_types.remove("wminkowski")   # TBD

    metric_types = ["euclidean"]
    log().debug(f"HDBSCAN Metrics: {metric_types}")

    best_labels = []
    best_score = -100000
    for x_metric in metric_types:
        for x_size in np.arange(m_upper, m_lower, (-1 * abs(m_increment))):
            for x_sample in np.arange(1, m_lower):
                iteration += 1
                if iteration % 10 == 0:
                    log().debug(f"HDBSCAN iteration: {iteration} x_metric: {x_metric} x_size: {x_sample} x_sample: {x_sample}")

                dbscan = scan(embeddings, metric=x_metric, size=int(x_size), samples=int(x_sample))
                labels = dbscan.labels_
                x_cluster = len(set(labels)) - (1 if -1 in labels else 0)
                if x_cluster < 2:
                    continue

                x_noise = list(labels).count(-1)
                x_clean = x_total - x_noise
                pct_clean = x_clean / x_total
                pct_noise = 1 - pct_clean
                x_score = metrics.silhouette_score(embeddings, labels)

                if x_score > best_score:
                    best_labels = labels
                    best_score = x_score

                # tag = "--"
                if pct_clean >= PCT_CLUSTER_CLEAN_THRESHOLD and x_cluster >= 2:
                    x_clusters.append(x_cluster)
                    x_noises.append(x_noise)
                    x_metrics.append(x_metric)
                    x_sizes.append(x_size)
                    x_samples.append(x_sample)
                    x_scores.append(x_score)
                    x_pct_noises.append(pct_noise)
                    x_pct_cleans.append(pct_clean)
                    # tag = "++"

    xdf = pd.DataFrame({
        "x_scores": x_scores,
        "x_clusters": x_clusters,
        "x_sizes": x_sizes,
        "x_samples": x_samples,
        "x_noises": x_noises,
        "x_pct_noises": x_pct_noises,
        "x_pct_cleans": x_pct_cleans,
        "x_metrics": x_metrics,
    })
    xdf = xdf.sort_values(by=["x_scores"], ascending=False)[xdf.x_clusters >= 2]
    if xdf.shape[0] == 0:
        log().debug(f"No clusters found")
        return None

    # log().debug(f"--- Hyperparameter Scores \n{xdf.head(100)} ---")

    best = xdf.iloc[0]
    best_score = best["x_scores"]
    # best_metric = best["x_metrics"]
    # best_size = best["x_sizes"]
    # best_samples = best["x_samples"]
    # best_cluster = best["x_clusters"]
    best_noise = best["x_noises"]
    # best_clean = x_total - best_noise
    pct_noise = (best_noise + 1) / x_total
    # log().debug(f"HDBSCAN best scr: {best_score:0.5f} met: {best_metric} sz: {best_size} sm: {best_samples} cs: {best_cluster} cl: {best_clean} ns: {best_noise} pc: {pct_noise}")

    return best_labels


def encode(df, column, categories):
    """
    Return a data set with the specified column one-hot encoded
    """
    log().debug(f"Encoding column: {column}, categories: {categories}")
    ohe = OneHotEncoder(sparse=False, categories=[categories])

    encode_columns = [column]
    ohe.fit(df[encode_columns])
    x = ohe.transform(df[encode_columns])

    encoded = pd.DataFrame(x, columns=categories, dtype=int)
    encoded["label"] = df.index
    encoded = encoded.set_index("label")

    # Concatenate the encoded data to the original dataframe
    xdf = pd.concat([df.drop(encode_columns, 1), encoded], axis=1)

    # Add prefix based upon original column name to all new columns
    name_map = {x: column + "_" + str(x) for x in categories}
    xdf = xdf.rename(columns=name_map)

    return xdf


def kmeans_hyperparameters(embeddings):
    """
    Return the clustered labels with the best score.
    KMEANS is used across range of cluster sizes and the best score is returned.
    """
    x_clusters = []
    x_scores = []

    # x_total = len(embeddings)

    best_labels = None
    best_score = -100000
    for n_clusters in range(MIN_KMEANS_CLUSTERS, MAX_KMEANS_CLUSTERS):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        labels = kmeans.labels_
        # prediction = kmeans.predict([[0, 0], [12, 3]])

        # NOTE: Silhouette score: The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
        x_score = metrics.silhouette_score(embeddings, labels)
        x_cluster = len(set(labels)) - (1 if -1 in labels else 0)

        log().debug(f"KMEANS score: {x_score} clusters: {n_clusters} labels: {labels}")

        if x_score > best_score:
            best_labels = labels
            best_score = x_score

        x_clusters.append(x_cluster)
        x_scores.append(x_score)

    xdf = pd.DataFrame({
        "x_scores": x_scores,
        "x_clusters": x_clusters,
    })
    xdf = xdf.sort_values(by=["x_scores"], ascending=False)[xdf.x_clusters >= 2]
    if xdf.shape[0] == 0:
        log().debug(f"No clusters found")
        return None

    # log().debug(f"--- Hyperparameter Scores --- \n{xdf.head(100)} ")

    best = xdf.iloc[0]
    best_score = best["x_scores"]
    best_cluster = best["x_clusters"]
    log().debug(f"KMEANS best scr: {best_score:0.5f} cl: {best_cluster}")

    return best_labels


def predict_distance_model(y_prob):
    """
    Return the predicted cluster id
    """
    y_pred = y_prob.argmax(axis=1)
    return y_pred


def predict_distance_proba_model(data):
    """
    Return the predicted cluster probabilities
    """

    # log().debug(f"data: {len(data)} {data[0]}")

    flattened = {}
    max_cluster = 0
    labels = []
    for item in data:
        label = item["label"]
        cluster = int(item["cluster"])
        if label not in flattened:
            flattened[label] = {}
            labels.append(label)
        p = item["nprobability"]
        flattened[label][cluster] = p
        if cluster > max_cluster:
            max_cluster = cluster

    # log().debug(f"flattened: {len(flattened.keys())} {flattened['noun:innovation']}")

    x = []
    for label in labels:
        ps = []
        for cluster in range(max_cluster + 1):
            p = flattened[label][cluster]
            ps.append(p)
        x.append(ps)

    items = np.array(x)
    # log().debug(f"items: {items.shape} {items[0,:]}")
    return items


def redim_data(data, components=2, transformation="PCA"):
    """
    Return redimensioned input data
    """

    X = data

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # log().debug(f"Redimensioning data using {transformation}")

    # Note that TSNE provides much better cluster resolution.
    # reference: https://medium.com/@violante.andre/an-introduction-to-t-sne-with-python-example-47e6ae7dc58f
    if transformation == "TSNE":
        # TSNE performance degrades with higher than 50 dimensions.
        # Reduce dimensions using PCA if this is the situation
        # reference: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

        (n_rows, n_features) = X.shape
        max_features = 10000
        if n_features >= max_features:
            trans = PCA(n_components=max_features, random_state=RANDOM_SEED)
            X = trans.fit_transform(X)
        method = "exact"
        if components <= 4:
            method = "barnes_hut"

        # log().debug(f"Redim components: {components}")
        trans = TSNE(n_components=components, random_state=RANDOM_SEED, method=method)
        transformed = trans.fit_transform(X)
        # log().debug("Completed TSNE")
    elif transformation == "UMAP":
        import umap
        trans = umap.UMAP(random_state=RANDOM_SEED, transform_seed=RANDOM_SEED)
        transformed = trans.fit_transform(X)
        # log().debug("Completed UMAP")
    elif transformation == "PCA":
        # log().debug("Starting PCA")
        trans = PCA(n_components=components, random_state=RANDOM_SEED)
        transformed = trans.fit_transform(X)
        # log().debug(f"PCA: Explained variation per principal component: {trans.explained_variance_ratio_}")
        # log().debug(f"PCA: Cumulative variation: {np.sum(trans.explained_variance_ratio_):0.5f}")
        # log().debug("Completed PCA")
    else:
        raise Exception(f"Invalid transformation: {transformation}")

    return transformed


def assemble_data(documents, labels, degrees, y_preds, y_probs):
    """
    Return assembed (from input data) cluster, word, and document data details
    """

    # Assemble word data
    word_data = {}
    for i, label in enumerate(labels):
        info = {}
        info["degree"] = degrees[i]
        info["y_pred"] = y_preds[i]
        info["y_prob"] = y_probs[i]
        word_data[label] = info

    # Assemble cluster data
    c_probs = [y_probs[i][x] for i, x in enumerate(y_preds)]
    data = {
        "label": labels, "degree": degrees, "y_pred": y_preds, "c_prob": c_probs
    }
    df = pd.DataFrame(data)

    cluster_data = {}
    gdf = df.groupby(["y_pred"])
    for idx, group in gdf:
        idx = int(idx)

        group.sort_values(by=["c_prob"], ascending=[False], inplace=True)
        row_dict = group.to_dict(orient="records")
        cluster_data[idx] = row_dict

    document_data = {}
    for doc_idx in documents.keys():
        tokens = documents[doc_idx]
        for token in tokens:
            label = token["label"]
            if label not in word_data:
                continue
            word = word_data[label]
            y_pred = word["y_pred"]
            y_prob = word["y_prob"]

            item = {}
            item["label"] = label
            item["y_pred"] = y_pred
            item["y_prob"] = y_prob
            if doc_idx not in document_data:
                document_data[doc_idx] = []
            document_data[doc_idx].append(item)

    return cluster_data, word_data, document_data


##########
#
# Step 1:
# - load data and create graph and features
#
##########

# Initialize numpy, pandas, logger, and random state
init()
log().info(f"Initialization completed...")

# Create a graph
log().info(f"Loading data")

# Load data
loader = LoaderDocument()
loader.load(INPUT_FILE, OOV_FILE, FIELDS, DATA_FIELD, num_rows=NUM_ROWS, seed=RANDOM_SEED)

# Get loaded data (graph, features, tokens)
log().info(f"Creating Graph")
graph = loader.graph()
show_graph(graph, title="DOCUMENT GRAPH", show_details=True, rows=200)

log().info(f"Creating features")
features, labels, columns = loader.features()
log().info(f"Features completed, shape: {features.shape}, labels: {len(labels)}")

log().info(f"Getting document info")
filtered_words = loader.filtered_words()
document_tokens = loader.document_tokens()
documents = loader.documents()
show_document_tokens(document_tokens)
log().debug(f"Documents: {len(document_tokens.keys())}, Labels: {len(labels)}, Filtered Words: {len(filtered_words)}")

log().info(f"Creating graph")
degrees = [graph.degree(x) for x in graph.nodes]
edges = list(graph.edges())

##########
#
# Step 2:
# - create embeddings
# - identify clusters of words (topics)
# - plot the data to visualize the clusters
#
##########

log().info(f"Creating embeddings")
embeddings = create_embeddings(graph, features, labels)

log().info(f"Calculating clusters and (weighted centers)")
clusters = calc_clusters(embeddings)
centers_dict = calc_centers(embeddings, clusters, degrees)

model_file = "xmodel.pkl"
log().info(f"Saving model to file: {model_file}")
save_model(graph, labels, features, columns, embeddings, clusters, model_file)
# load_model(model_file)

# NOTE: this will take a long time with large datasets
log().info(f"Plotting data")
plot(embeddings, clusters, labels)
log().info(f"Plotting completed")

# Get the probability distribution for each node in each cluster
mod = create_distance_model(labels, embeddings, degrees, centers_dict)
y_prob = predict_distance_proba_model(mod)
y_pred = predict_distance_model(y_prob)

cluster_data, word_data, document_data = assemble_data(document_tokens, labels, degrees, y_pred, y_prob)

# Show word probabilities
show_words(word_data)

# Show cluster data
show_topics(cluster_data)

# Show document probabilities
show_document_topics(document_data, documents)
