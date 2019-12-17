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
import logging.config

import util
import Loader

# LOCAL IMPORTS
# import sys
# import_dir = os.path.expanduser("../src")
# import_dir = os.path.abspath(import_dir)
# sys.path.append(import_dir)  # insert at 1, 0 is the script path (or '' in REPL)
# import common as common

# Out of vocabulary words file
OOV_FILE = "/Users/ericbroda/Development/python/gnn/tests/oov.json"

# Number of rows statistics:
# Time to complete model: 1000: 75sec, 10000:500sec (~8 min), 100000: 55min
# - Size of model: 1000: 6MB, 2400: 10MB
NUM_ROWS = 100

# Data files
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
# Utility Functions - Initialization
#
##########
def init():

    # Establish logger, levels: "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
    level = "INFO"
    logging.config.dictConfig(util.configure_log(level))


##########
#
# Step 1:
# - load data and create graph and features
#
##########

# Initialize numpy, pandas, logger, and random state
init()
util.log().info(f"Initialization completed...")

# Create a graph
util.log().info(f"Loading data")

# Load data
loader = Loader.LoaderDocument()
loader.load(INPUT_FILE, OOV_FILE, FIELDS, DATA_FIELD, num_rows=NUM_ROWS, seed=util.RANDOM_SEED)

# Get loaded data (graph, features, tokens)
util.log().info(f"Creating Graph")
graph = loader.graph()
util.show_graph(graph, title="DOCUMENT GRAPH", show_details=True, rows=200)

util.log().info(f"Creating features")
features, labels, columns = loader.features()
util.log().info(f"Features completed, shape: {features.shape}, labels: {len(labels)}")

util.log().info(f"Getting document info")
filtered_words = loader.filtered_words()
document_tokens = loader.document_tokens()
documents = loader.documents()
util.show_document_tokens(document_tokens)
util.log().debug(f"Documents: {len(document_tokens.keys())}, Labels: {len(labels)}, Filtered Words: {len(filtered_words)}")

util.log().info(f"Creating graph")
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

util.log().info(f"Creating embeddings")
embeddings = util.create_embeddings(graph, features, labels)

util.log().info(f"Calculating clusters and (weighted centers)")
clusters = util.calc_clusters(embeddings)
centers_dict = util.calc_centers(embeddings, clusters, degrees)

model_file = "xmodel.pkl"
util.log().info(f"Saving model to file: {model_file}")
util.save_model(graph, labels, features, columns, embeddings, clusters, model_file)
# load_model(model_file)

# NOTE: this will take a long time with large datasets
util.log().info(f"Plotting data")
util.plot(embeddings, clusters, labels, degrees)
util.log().info(f"Plotting completed")

# Get the probability distribution for each node in each cluster
mod = util.create_distance_model(labels, embeddings, degrees, centers_dict)
y_prob = util.predict_distance_proba_model(mod)
y_pred = util.predict_distance_model(y_prob)

cluster_data, word_data, document_data = util.assemble_data(document_tokens, labels, degrees, y_pred, y_prob)

# Show word probabilities
util.show_words(word_data)

# Show cluster data
util.show_topics(cluster_data)

# Show document probabilities
util.show_document_topics(document_data, documents)
