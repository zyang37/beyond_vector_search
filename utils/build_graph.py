from utils.parse_arxiv import (
    make_keyword_id,
    parse_authors,
    parse_categories,
    parse_journal,
    parse_year,
)
import pickle
import pandas as pd
import sys

sys.path.append("../")
from vector_graph.bipartite_graph_dict import BipartiteGraphDict


def graph_extend_node_edge(
    idx, target_infor, k_id_name, keyword_nodes, document_id, edges
):
    """
    This function extends the graph by adding new nodes and edges

    args:
        - idx: index of the row in the dataframe
        - target_infor: target information to be parsed (could be authors, keywords, categories, etc)
        - k_id_name: keyword name used to make keyword id
        - keyword_nodes: list of keyword nodes
        - document_id: id of the document
        - edges: list of edges
    """

    if type(target_infor.iloc[0]) is list:
        target_infor_dim = 2
    else:
        target_infor_dim = 1

    if target_infor_dim == 1:
        keyword_ids = make_keyword_id(k_id_name, target_infor.iloc[idx])
        keyword_nodes.append(keyword_ids)
        edges.append((document_id, keyword_ids))
    elif target_infor_dim == 2:
        keyword_ids = [make_keyword_id(k_id_name, x) for x in target_infor.iloc[idx]]
        keyword_nodes.extend(keyword_ids)
        edges.extend([(document_id, k) for k in keyword_ids])
    else:
        raise NotImplementedError
    return keyword_nodes, edges


def build_graph(df, G=None):
    # add document nodes

    if not G:
        G = BipartiteGraphDict()
    author_keywords = []
    category_keywords = []
    journal_keywords = []
    year_keywords = []

    author_edges = []
    category_edges = []
    journal_edges = []
    year_edges = []

    authors = df["authors"].map(parse_authors)
    categories = df["categories"].map(parse_categories)
    journals = df["journal-ref"].map(parse_journal)
    years = df["update_date"].map(parse_year)

    # df.drop_duplicates(subset=['id'], inplace=True)
    df.drop_duplicates(subset=["id"], inplace=True)
    df["id"] = df["id"].astype("string")
    data_ids = set(df["id"].tolist())

    for idx in range(df.shape[0]):
        document_id = df["id"].iloc[idx]
        keyword_author_ids = [make_keyword_id("author", x) for x in authors.iloc[idx]]
        author_keywords.extend(keyword_author_ids)
        author_edges.extend([(document_id, k) for k in keyword_author_ids])

        keyword_category_ids = [
            make_keyword_id("category", x) for x in categories.iloc[idx]
        ]
        category_keywords.extend(keyword_category_ids)
        category_edges.extend([(document_id, k) for k in keyword_category_ids])

        journal_id = make_keyword_id("journal", journals.iloc[idx])
        journal_keywords.append(journal_id)
        journal_edges.append((document_id, journal_id))

        year_id = make_keyword_id("year", years.iloc[idx])
        year_keywords.append(year_id)
        year_edges.append((document_id, year_id))

    author_keywords = set(author_keywords)
    author_edges = set(author_edges)
    category_keywords = set(category_keywords)
    category_edges = set(category_edges)
    journal_keywords = set(journal_keywords)
    journal_edges = set(journal_edges)
    year_keywords = set(year_keywords)
    year_edges = set(year_edges)

    G.add_data_nodes(data_ids)
    G.add_keyword_nodes(author_keywords)
    G.add_keyword_nodes(category_keywords)
    G.add_keyword_nodes(journal_keywords)
    G.add_keyword_nodes(year_keywords)
    G.add_raw_edges(author_edges)
    G.add_raw_edges(category_edges)
    G.add_raw_edges(journal_edges)
    G.add_raw_edges(year_edges)
    return G
