import numpy as np
import pandas as pd

import collections

from sklearn.preprocessing import OneHotEncoder

import spacy
import uuid
import re
import json

import util


VOCABULARY = "en_core_web_lg"    # Spacy vocabulary to use
XGRAM_THRESHOLD = 3              # Max size of x-grams for phrases / noun-chunks
XGRAM_FREQUENCY_CUTOFF = 100     # Min occurrences of an phrase (xgram) for it to be considered relevant (ie. used in) processing
MAX_TOKENS_IN_SENTENCE = 300     # Maximum number of tokens for a valid sentence (over this is ignored due to bad puctuation)
MAX_TOKEN_LENGTH = 25            # Maximum characters in token text

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

# IMPORTANT_TAGS = ["noun", "propn", "x", "adj"]                # Spacy POS tags that are considered important
IMPORTANT_TAGS = ["noun", "propn", "x"]                         # Spacy POS tags that are considered important
CONSUMED_TAGS = IMPORTANT_TAGS                                  # Spacy POS tags that are considered useful (others are not included in analysis)
# CONSUMED_TAGS = ["noun", "adj", "verb", "adv", "propn", "x"]  # Spacy POS tags that are considered useful (others are not included in analysis)


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

    def load(self, input_spec, OOV_FILE, fields, data_field, num_rows=None, frac=1.0, seed=util.RANDOM_SEED):
        """
        Load documents from input specification (directory) using
        OOV (out of vocabulary) input file (oov.json) if it is present
        and create graph, features, and document-node mapping
        """

        util.log().debug(f"Reading filename: {input_spec}, fields: {fields} data_field: {data_field} num_rows: {num_rows} frac: {frac}")
        df = pd.read_csv(input_spec, engine="python", encoding="ISO-8859-1", names=fields, header=0)
        util.log().debug(f"Loaded dataframe, shape: {df.shape}: \n{df.head()}")

        # Sample the data (if num_rows exists then use it, otherwise use frac (default is 100% of data)
        if num_rows:
            total_rows = df.shape[0]
            if num_rows > total_rows:
                num_rows = total_rows
            df = df.sample(n=num_rows, random_state=seed)
        else:
            df = df.sample(frac=frac, random_state=seed)
        util.log().debug(f"Input (sampled) dataframe, shape: {df.shape}: \n{df.head()}")

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

        util.log().info("Loading spacy vocabulary: {}".format(VOCABULARY))
        nlp = spacy.load(VOCABULARY)
        util.log().info("Loading spacy vocabulary completed: {}".format(VOCABULARY))

        # Read OOV map file if it exists
        util.log().debug(f"Using OOV file: {OOV_FILE}")
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

        util.log().debug(f"OOV Mapping: {self._oov_map}")
        # util.log().debug(f"_oov_vectors: {self._oov_vectors}")

        # After much testing it appears that batch size has a strong impact on throughput.
        # Large batch sizes seem to hinder multi-threading and ultimately reduces CPU consumption
        # at least on smaller machines.  It seems that maximum thoughput and CPU usage
        # occurs with a very small batch size, hence it is set to 1 right now.
        # batch_size = int(len(data) / 1000)
        batch_size = 1

        # Add a test pipeline element (just to show how it can be done)
        nlp.add_pipe(self._pipeline, name="filter", last=True)

        util.log().info(f"Processing {len(data)} documents")
        from tqdm import tqdm
        docs = tqdm(nlp.pipe(data, batch_size=batch_size, n_threads=20))
        docs = list(docs)
        util.log().info(f"Processing {len(data)} documents completed")

        phrases = self._find_phrases(docs)
        phrases = sorted(phrases.items(), key=lambda kv: kv[1], reverse=True)
        # self._show_phrases(phrases)

        util.log().debug("Creating document tokens")
        documents = self._process_documents(docs)

        util.log().debug("Reconciling phrases")
        documents = self._reconcile_phrases(documents, phrases)

        util.log().debug("Reconciling children")
        documents = self._reconcile_children(documents)

        labels = [token["label"] for document in documents for token in document]
        token_count = collections.Counter(labels)
        # util.log().debug(f"Most common token labels: {token_count.most_common()[0:20]}")

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

        util.log().debug("Calculating filter words")
        filtered_words = self._find_filtered()
        self._filtered_words = filtered_words

        # util.log().debug(f"filtered_words: {len(filtered_words)} \n {filtered_words}")
        # util.log().debug(f"tokens_by_document: {[str(d) + ':' + t['lemma'] for d in tokens_by_document.keys() for t in tokens_by_document[d]]}")

    def features(self):
        """
        Return the features for the loaded/input data
        """

        util.log().info(f"Analysing features")

        if not self._graph:
            raise Exception("This loader requires that the graph be created before features - please call graph() first")

        util.log().info(f"Calculating metrics")
        ranks, hits, authorities, degree_centrality = util.calc_metrics(self._graph)
        util.log().info(f"Calculating metrics complete")

        util.log().info(f"Creating token features")
        features = []
        for node in self._graph.nodes:
            if node not in self._tokens_by_label.keys():
                util.log().debug(f"features node: {node} not in LABEL dictionary: {util.pretty(self._tokens_by_label.keys())}")
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

        util.log().info(f"Creating token features complete")

        df = self._create_feature_df(features)

        # for col in df.columns.values:
        #     uniques = df[col].unique().tolist()
        #     num_uniques = len(uniques)
        #     if num_uniques == 1:
        #         util.log().debug(f"Column {col} has a single ({num_uniques}) unique values")
        #     else:
        #         util.log().debug(f"Column {col} has {num_uniques} unique values")

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
            util.log().debug(f"nodes: {num_nodes} {nodes}")
            msg = f"Nodes and features length do not match, nodes: {num_nodes} features: {num_features}"
            raise Exception(msg)

        util.log().info(f"Analysing features complete")

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
        # util.log().debug(f"Document tokens: \n{pretty(token_lemmas)}")

        filtered_words = self._filtered_words
        util.log().debug("Calculating edges, docs: {}".format(len(documents)))
        edges = self._calc_edges(documents, filtered_words)

        util.log().debug("Creating graph, edges: {}".format(len(edges)))
        graph = util.create_graph(edges, graph_type="multigraph")

        node_degrees = list(graph.degree())
        distribution = collections.Counter([degree for (node, degree) in node_degrees])

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
        # util.log().debug(f"min_degree_threshold: {min_degree_threshold} pct: {pct} pruning: {sum_nodes} of total: {total_nodes}")

        for (node, degree) in node_degrees:
            if degree < min_degree_threshold:
                # util.log().debug(f"Pruning node: {node} with degree: {degree}")
                # util.log().debug(f"Pruning node: {node} with degree: {degree}")
                util.prune_node(graph, node)

        # Remove nodes which have no links (ie. degree == 0) which result from pruning
        node_degrees = list(graph.degree())
        for (node, degree) in node_degrees:
            if degree == 0:
                # util.log().debug(f"Removing node: {node} with degree: {degree}")
                graph.remove_node(node)

        node_degrees = list(graph.degree())
        distribution = collections.Counter([degree for (node, degree) in node_degrees])
        util.log().debug(f"Filtered degree distribution: {distribution.most_common()}")

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
        #     util.log().debug(f"doc {i}: {texts}")
        # util.log().debug(f"filtered_words: {filtered_words}")

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
            util.log().debug(f"Reducing dimensions from {nfeatures}to: {components}")
            vector = self._redim_vectors(features, components=components)

            # Step 2: Remove the old vector data
            util.log().debug(f"Removing old vector data")
            for feature in features:
                for col in list(feature.keys()):
                    if col.startswith("v-"):
                        del feature[col]

            # Step 3:Add the new vector data
            util.log().debug(f"Adding new vector data")
            for i, feature in enumerate(features):
                for j, item in enumerate(vector[i]):
                    cname = "v-" + str(j)
                    feature[cname] = item

        # Create the feature dataframe and: remove str idx and change booleans to integer
        df = pd.DataFrame(features)
        # util.log().debug(f"Creating feature dataframe using features: {features}")

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
            util.log().debug("Unusual DEP: {}, dep: {}, lemma: {}, pos: {}, tag: {}".format(tname, token["dep_"], lemma, token["pos_"], token["tag_"]))
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
        #     util.log().debug(f"Document sentiment idx not found: {didx} text: {token.doc.text[0:25]}")
        # dsentiment = self._dsentiments[didx]

        ssentiment = {
            "polarity": 0,
            "subjectivity": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "compound": 0
        }
        sidx = util.calc_hash(token.sent.text)
        if sidx not in self._ssentiments:
            util.log().debug(f"Sentence sentiment idx not found: {sidx} text: {token.sent.text[0:25]}")
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
        tidx = util.calc_hash(token.text)
        if tidx not in self._tsentiments:
            util.log().debug(f"Token sentiment idx not found: {tidx} text: {token.text[0:25]}")
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
                # util.log().debug(f"Substituting vector for label: {label} replacing with vector for word: {self._oov_map[label]}")
            else:
                x["is_oov"] = True
                util.log().debug(f"OOV (out of vocabulary, unknown) label: {label}")

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
        util.log().debug(output)

    def _encode(self, df, column, categories):
        util.log().debug(f"Encoding column: {column}, categories: {categories}")
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
                util.log().debug(f"filter idx: {idx} not in dictionary")
                util.log().debug(f"filter idx: {idx} not in dictionary")
                continue
            token = self._tokens_by_idx[idx]

            # util.log().debug(f"consuming token: {token['lemma']}")
            should_filter, reason = self._should_filter_token(token, filtered_words)
            if should_filter:
                self_reference = self._is_important(token)
                # util.log().debug(f"filtering word: {token['lemma']} self_reference: {self_reference} reason: {reason}")
                util.prune_node(graph, idx, self_reference=self_reference)

    def _filter_common(self, documents, keep=0.90):
        """
        Return a list of terms to filter which are considered less relevant based upon TFIDF analysis
        """
        tfidf = self._calc_tfidf(documents)
        tfidf_items = list(tfidf.items())
        tfidf_items.sort(key=lambda x: x[1], reverse=True)
        # util.log().debug(f"tfidf_items: {tfidf_items}")

        num = int(len(tfidf_items) * keep)
        filtered_scores = tfidf_items[num:]
        # util.log().debug(f"filtered_scores: {filtered_scores}")

        output = ""
        output += "\n --- Filtered Words (Common): (keep: {}, {} of total words: {}) ---".format(keep, len(filtered_scores), len(tfidf_items))
        for (word, score) in filtered_scores:
            output += "\n {:>30}:{:0.4f}".format(word, score)
        util.log().debug(output)

        filtered_words = [term for (term, score) in filtered_scores]

        return filtered_words, tfidf

    def _filter_extremes(self, documents):
        """
        Return list of terms in the document considered extreme/outliers (gensim functionality)
        """

        # util.log().debug("_filter_extremes documents: {}".format(documents))

        documents = [x.split() for document in documents for x in document]
        # util.log().debug("1. documents: {}".format(documents))
        # documents = documents.split()
        # util.log().debug("2. documents: {}".format(documents))

        from gensim import corpora
        original = corpora.Dictionary(documents)
        dct = corpora.Dictionary(documents)
        ntotal_words = len(dct.keys())

        # util.log().debug("1. dct: {}: {}".format(len(dct), dct))
        no_below = FILTER_PRESENT_MIN_DOCS  # HIGHER absolute number filter/remove MORE words
        no_above = FILTER_PRESENT_MAX_DOCS  # HIGHER pct filter/remove FEWER words
        dct.filter_extremes(no_below=no_below, no_above=no_above)
        # dct.filter_extremes(no_below=3, no_above=0.4)
        util.log().debug(f"Extreme Filter Dictionary: {len(dct)}: {dct}")

        filtered_words = [original[x] for x in original.keys() if x not in dct.keys()]
        if not filtered_words:
            util.log().debug("No extreme filtered words")
            return filtered_words

        nextreme_words = len(filtered_words)

        lens = [len(i) for i in filtered_words]
        avglens = np.mean(lens)
        stdlens = np.std(lens)
        util.log().debug(f"Candidate Filtered Words: num: {len(filtered_words)} length avg: {avglens} std: {stdlens}")

        filtered_words = [x for x in filtered_words if len(x) <= FILTER_BIG_WORD_LENGTH]
        lens = [len(i) for i in filtered_words]
        avglens = np.mean(lens)
        stdlens = np.std(lens)
        util.log().debug(f"Candidate Filtered Words (kept only bigger words): num: {len(filtered_words)} length avg: {avglens} std: {stdlens}")

        filtered_words = [x for x in filtered_words if x.split(":")[0] not in CONSUMED_TAGS]
        lens = [len(i) for i in filtered_words]
        avglens = np.mean(lens)
        stdlens = np.std(lens)
        util.log().debug(f"Final Filtered Words (keep consumable POS): num: {len(filtered_words)} length avg: {avglens} std: {stdlens}")

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
        util.log().debug(output)

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
        # util.log().debug(f"TFID documents: {len(documents)} {documents}")

        filtered_tfidf, tfidf = self._filter_common(documents, keep=TFIDF_KEEP_PCT)
        # util.log().debug(f"filtered_tfidf: {filtered_tfidf}")
        # util.log().debug(f"tfidf: {tfidf}")
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
        util.log().debug(output)

        exception_words = [word for word in candidate_words if word.split(":")[0] in CONSUMED_TAGS and len(word.split(":")[1]) >= MIN_FILTERED_WORD_LEN]
        output = ""
        output += "\n --- Exception Filtered Words: {} ---".format(len(exception_words))
        for word in exception_words:
            output += f"\n {word}"
        util.log().debug(output)

        filtered_words = [word for word in candidate_words if word not in exception_words]
        output = ""
        output += "\n --- Final Filtered Words (Common): {} ---".format(len(filtered_words))
        for word in filtered_words:
            output += f"\n {word}"
        util.log().debug(output)

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

        # util.log().debug(f"Parsing item, parent: {parent} token: {token}")
        if parent and token:
            edge = (parent["idx"], token["idx"])
            # util.log().debug(f"Creating edge: parent: {parent['label']}/{parent['text']}/{parent['idx']} token: {parent['label']}/{parent['text']}/{parent['idx']}")
            edges.append(edge)

        # if depth > 50:
        #     util.log().debug(f"Max depth, depth: {depth} edge: par: {parent['label']}/{parent['text']}/{parent['idx']} tok: {parent['label']}/{parent['text']}/{parent['idx']}")
        #     return False

        for idx in token["children"]:
            if idx not in self._tokens_by_idx:
                util.log().debug(f"Unknown child idx: {idx} for token: {token['label']}")
                continue
            child = self._tokens_by_idx[idx]

            if token["idx"] == child["idx"]:
                util.log().debug(f"token: {['idx']} same as child: {child['idx']}")
                # util.log().debug(f"token: {token['label']}/{token['text']}/{token['idx']}")
                # util.log().debug(f"children: {token['children']}")
                continue

            depth += 1
            self._parse_item(depth, edges, token, child)

    def _parse_sentence(self, root):
        """
        Return the graph for an individual sentence (defined by its root)
        """

        edges = []
        # util.log().debug(f"Parsing sentence, root: {root['label']}/{root['text']}/{root['idx']}")
        depth = 0
        parent = None

        self._parse_item(depth, edges, parent, root)

        graph = util.create_graph(edges)
        return graph

    def _pipeline(self, doc):
        # This is a test version of a pipeline - it gets called for every document
        # Examples can be found at: https://spacy.io/usage/processing-pipelines
        return doc

    def _process_documents(self, docs):
        """
        Return a list of processed documents where each document is composed of a list of tokens
        """

        util.log().info(f"Processing tokens from documents: {len(docs)}")

        # Capture the sentiments
        util.log().debug(f"Processing sentiments")
        tsentiments = {}
        ssentiments = {}
        dsentiments = {}
        for i, doc in enumerate(docs):
            document = []
            # Calculate the document sentiment

            didx = util.calc_hash(doc.text)
            if didx not in dsentiments.keys():
                dsentiment = util.calc_sentiment(doc.text)
                dsentiments[didx] = dsentiment
                # util.log().debug(f"doc text: {doc.text} dsentiment: {dsentiment}")

            if i % 100 == 0:
                util.log().debug(f"Processing sentiments for document: {i}")

            for j, sent in enumerate(doc.sents):

                # Calculate the sentence sentiment
                sidx = util.calc_hash(sent.text)
                if sidx not in ssentiments.keys():
                    ssentiment = util.calc_sentiment(sent.text)
                    ssentiments[sidx] = ssentiment
                    # util.log().debug(f"sent text: {sent.text} ssentiment: {ssentiment}")

                for k, token in enumerate(sent):
                    if token.text == "" or token.text == " ":
                        util.log().debug(f"Token text is empty: {token}")
                        continue

                    if len(token.text) >= MAX_TOKEN_LENGTH:
                        util.log().debug(f"Token too long: {token.text}, len: {len(token.text)}")
                        continue

                    tidx = util.calc_hash(token.text)
                    if tidx not in tsentiments.keys():
                        tsentiment = util.calc_sentiment(token.text)
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
                util.log().debug(f"Processing tokens for document: {i}")

            for j, sent in enumerate(doc.sents):

                if self._should_filter_sentence(sent):
                    util.log().debug(f"Invalid sentence: {sent}")
                    continue
                for k, token in enumerate(sent):
                    if token.text == "" or token.text == " ":
                        util.log().debug(f"Token text is empty: {token}")
                        continue

                    if len(token.text) >= 50:
                        util.log().debug(f"Token too long: {token.text}, len: {len(token.text)}")
                        continue

                    x = self._create_token(token)
                    document.append(x)
                    if x["is_oov"]:
                        oov_terms.append(x["label"])
            documents.append(document)

        # Create a token dictionary (by idx) to allow children indexing
        util.log().debug(f"Creating token dictionary")
        token_dict = {}
        for document in documents:
            for token in document:
                spacy_token = token["spacy_token"]
                spacy_doc = str(spacy_token.doc[0:25])
                key = spacy_doc + "." + str(spacy_token.idx)
                key = util.calc_hash(key)
                # util.log().debug(f"key: {key}")
                token_dict[key] = token

        # util.log().debug(f"Token dictionary: {token_dict}")

        # Assign children to tokens
        util.log().debug(f"Assigning children tokens, tokens: {len(token_dict.keys())}")
        for token_key in token_dict.keys():
            token = token_dict[token_key]
            spacy_token = token["spacy_token"]
            for child in spacy_token.children:
                child_doc = str(spacy_token.doc[0:25])
                key = child_doc + "." + str(child.idx)
                key = util.calc_hash(key)
                if key not in token_dict:
                    util.log().debug(f"Could not find key: {key} child doc: {child.doc[0:20]} child idx: {child.idx}")
                    continue
                child_token = token_dict[key]
                token["children"].append(child_token["idx"])
                # util.log().debug(f"token {token['idx']} has child_token: {child_token['idx']}")

        count = collections.Counter(oov_terms)
        util.log().debug(f"Remaining OOV terms: {count.most_common()}")

        util.log().info(f"Processing tokens complete")

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
            # util.log().debug("--- current")
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
                        # util.log().debug("1. rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))
                        rdocument.extend(beginning)
                        # util.log().debug("2. rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))
                        rdocument.append(phrase_token)
                        # util.log().debug("3. rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))
                        rdocument.extend(ending)
                        # util.log().debug("4. rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))

                        # util.log().debug("beginning: {}".format("|".join([x["lemma"] for x in beginning])))
                        # util.log().debug("phrase_token: {}".format(phrase_token["lemma"]))
                        # util.log().debug("ending: {}".format("|".join([x["lemma"] for x in ending])))
                        # util.log().debug("NEW rdocument: {}".format("|".join([x["lemma"] for x in rdocument])))
                        lemmas = [x["lemma"] for x in rdocument]

            # util.log().debug("--- revised")
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
        data = util.redim_data(data, components=components)
        return data

    def _show_phrases(self, phrases):
        """
        Log phrases in human readable format
        """
        util.log().debug(self._format_phrases(phrases))

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
