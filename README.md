# articles-nlp

*Under Construction*

Source code for Medium.com articles discussing Natural Language Processing (NLP)
Techniques written by [Eric Broda](https://www.linkedin.com/in/ericbroda/)

## Topic-Modeling-using-Graph-Neural-Networks.py

Article: [Topic Modeling using Graph Neural Networks](medium.com)

This program demonstrates how graph neural networks can be used to perform
Topic Modeling. This specific demo reads a corpus of documents (CSV file)
to create a "word graph" and word features for processing in a graph neural
network (GNN). The GNN takes into account the relationship between words
in the documents to creates word vectors that reflect much more meanginful
relationships between the words.  The word embeddings are clustered (using KMEANS)
into topics that aggregate related words.

This demo uses the following open source products:
- [Spacy](https://spacy.io/): a very flexible and industrial strength
set of NLP functionality.  Spacy is used to ingest and tokenize documents which are
used to create graphs
- [NetworkX](https://github.com/networkx/networkx): a very popular and powerful
graph product.  NetworkX is used to create the graphs used in the GNN
- [StellarGraph](https://github.com/stellargraph/stellargraph): an outstanding
graph neural network product. StellarGraph offers many types of GNN - the
GraphSAGE GNN is used in this demonstration

Note: Instructions are available in source code file

## Topic-Modeling-using-Graph-Neural-Network-Embeddings.py

Article: [Topic-Modeling using Graph Neural Network Embeddings](medium.com)

## Generating-Word-Embeddings-using-Graph-Neural-Networks.py

Article: [Generating Word Embeddings using Graph Neural Networks](medium.com)

## Topic-Modeling-Comparison-of-Graph-Neural-Network-and-LDA.py

Article: [Topic Modeling Comparison of Graph-Neural-Network and LDA](medium.com)

