<h2 align="center">
    Beyond Vector Search
    <br>
    Hybrid Vector/Knowledge Graph Query Engine for Enhanced Data Retrieval
</h2>

<p align="center">
  <img src="./notebooks/plotCosSim/zoom/k1000gk500.png" alt="Hybrid Vector/Knowledge Graph Query Engine for Enhanced Data Retrieval">
<!-- ![image](./notebooks/plotCosSim/zoom/k1000gk500.png "Hybrid Vector/Knowledge Graph Query Engine for Enhanced Data Retrieval") -->

# Installation

We used Python==3.9.18, and we recommend using a virtual environment to install the required packages.

```
pip install -r requirements.txt
```

### Creating filtered data

- `notebooks/parsing_json.ipynb`: filter data for [arxiv](https://www.kaggle.com/Cornell-University/arxiv)
- `notebooks/parsing_cnn_news.ipynb`: filter data for [CNN news](https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning/data)
- `notebooks/parsing_wiki_movies.ipynb`: filter data for [wiki movies](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)

# Dataset

We used a subset of the ArXiv dataset from Kaggle which contains 12,926 samples, and the the pickle file is provided at "beyond_vector_search/data/filtered_data.pickle".

# Source Code

```
make_vectordb.py: a script to build a vector database from a "data/filtered_data.pickle"

utils/
    - build_graph.py: a script containing helper functions for building the knowledge graph
    - parse_arxiv.py: a script containing helper functions for parsing the arxiv dataset
vector_graph/
    - bipartite_graph_dict.py: A custom implementation of the bipartite graph
    - bipartite_graph_networkx.py: An experimental implementation of the bipartite graph using networkx
    - embedding_models.py: A custom implementation of the embedding models for generating the text embeddings
workloads
    - keyword_extractor.py
    - query_gen.py: A script for generating the text queries given paper data points
    - workload_gen.sh: This is the script for generating the workloads we described in the report
testing
    - inference.py: A script for executing our various search query engines on the generated workloads
zy_testing
    - compute_metrics_cos.py: A script for computing the accuracy of our results utilizing various performance compute_metrics_cos
```
