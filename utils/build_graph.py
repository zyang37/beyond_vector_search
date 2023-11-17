from parse_arxiv import (
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
from vector_graph.bipartite_graph import BipartiteGraph


def build_graph(df):
    # add document nodes
    G = BipartiteGraph()
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


if __name__ == "__main__":
    filtered_data_path = "../data/filtered_data.pickle"
    with open(filtered_data_path, "rb") as f:
        filtered_data = pickle.load(f)
    graph = build_graph(filtered_data.iloc[:100])
    graph.draw_graph()
