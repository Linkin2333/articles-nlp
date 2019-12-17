import random
import math
import statistics
import pickle
import pprint
import logging
import collections
import numpy as np
import pandas as pd

import hdbscan
import networkx as nx

import stellargraph as sg
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


import keras
import os
import tensorflow as tf
from keras import backend as K


RANDOM_SEED = 42

TRANSFORMATION = "UMAP"  # PCA is reproducible, TSNE and UMAP are fast but results vary each run, but clusters seem to align

BATCH_SIZE = 50
NUM_SAMPLES = [5, 5]
EPOCHS = 3

PCT_CLUSTER_CLEAN_THRESHOLD = 0.25  # HDBSCAN parameters
MIN_KMEANS_CLUSTERS = 3       # Minimum number of clusters for KMEANS
MAX_KMEANS_CLUSTERS = 15      # Maximum number of clusters for KMEANS


##########
#
# Utility Functions - General
#
##########
def init_random_state():
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
        max_len = 100
        if len(dstr) > max_len:
            dstr = dstr[0:max_len] + f"... ({len(dstr) - max_len} more chars)"
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
                linked_nodes_count = collections.Counter(linked_nodes)
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


def plot(embeddings, clusters, labels, degrees):
    """
    Plot the data on a 2D chart.  Note that raw embeddings were re-dimensioned to 2D.
    This will provide a general view of the clusters - note that the plotting
    is on the reduced dimensional data so the clustering visualization may be
    a bit inaccurate.
    """
    emb = redim_data(embeddings, components=2, transformation=TRANSFORMATION)
    x = [xx[0] for xx in emb]
    y = [yy[1] for yy in emb]
    plot_2D(x, y, clusters, labels, degrees)


def plot_2D(x, y, clusters, labels, degrees):
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
