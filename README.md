# articles-nlp

Source code for Medium.com articles discussing Natural Language Processing (NLP) Techniques
written by [Eric Broda](https://www.linkedin.com/in/ericbroda/)

## Graph-Neural-Networks-for-Topic-Modelling.py

Demonstrates how graph neural networks can be used to perform Topic Modelling.  This specific 
demo reads a corpus of documents (CSV file) to create a "word graph" and word features for 
processing in a graph neural network (GNN). The GNN takes into
account the relationship between words in the documents to creates word vectors that reflect
much more meanginful relationships between the words.  The word embeddings are clustered (using KMEANS)
into topics that aggregate related words.

This demo uses the following open source products:
- [Spacy](https://spacy.io/): a very flexible and industrial strength set of NLP functionality.  Spacy is used
to ingest and tokenize documents which are  used to create graphs
- [NetworkX](https://github.com/networkx/networkx): a very popular and powerful graph product.  NetworkX
is used to create the graphs used in the GNN
- [StellarGraph](https://github.com/stellargraph/stellargraph): an outstanding graph neural network product.  
StellarGraph offers many types of GNN - the GraphSAGE GNN is used in this demonstration

The full article for this demo is available at here (TBD).  The article provides a step-by-step
discussion for graph creation, GNN execution, and topic and document distribution analysis.  Example
outputs are also provided.

### Installation Instructions

1. Install the dependant packages:

```
pip install -r requirements.txt
```

2. Install the spacy vocabulary:

```
python -m spacy download en_core_web_lg
```

3.  Get the data:

```
[command to download the data]
```

4.  Run the python code:

```
python Graph-Neural-Networks-for-Topic-Modelling.py
```
