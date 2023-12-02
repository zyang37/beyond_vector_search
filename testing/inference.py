"""
This script is used to run inference on a workload csv file.
The workload csv file should have a column named "query" that contains the queries.
The script will save 3 results (GT, vector search, hybrid search) to the csv file.
The name of the columns are the same as the collection names (in chroma db).
"""

import pandas as pd
import chromadb
import argparse
import pickle
from tqdm.auto import tqdm
import os
import sys
import math
from pathlib import Path

sys.path.append("../")
from utils.build_graph import build_graph

HYBRID_COL = "hybrid"
WEIGHTED_HYBRID_COL = "weighted_hybrid"


def create_paper_id_to_title_dict(filtered_data):
    return dict(zip(filtered_data.id.astype("string"), filtered_data.title))


def create_id_to_abstract_dict(filtered_data):
    return dict(zip(filtered_data.id.astype("string"), filtered_data.abstract))


def get_query_col(df, id2abstract_dict):
    return df["query"]


def get_abstract_col(df, id2abstract_dict):
    return df["paper_id"].astype("string").map(id2abstract_dict)


def vector_search(
    df, coll_name, client, k, get_query_func, id2abstract_dict, batch_size=50
):
    collection = client.get_collection(name=coll_name)
    query_col = get_query_func(df, id2abstract_dict)
    search_results = []
    for idx in tqdm(range(0, df.shape[0], batch_size)):
        queries = query_col.iloc[idx : idx + batch_size].values.tolist()
        results = collection.query(query_texts=queries, n_results=k)
        search_results.extend(results["ids"])
    return search_results


def hybrid_search(
    df,
    coll_name,
    client,
    graph,
    vector_k,
    graph_k,
    get_query_func,
    id2abstract_dict,
    batch_size=50,
):
    vector_search_results = vector_search(
        df, coll_name, client, vector_k, get_query_func, id2abstract_dict, batch_size
    )

    graph_search_results = []
    for single_query_results in vector_search_results:
        graph_search_results.append(graph.find_relevant(single_query_results, graph_k))

    return [
        sublist1 + sublist2
        for sublist1, sublist2 in zip(vector_search_results, graph_search_results)
    ]


def weighted_hybrid_search(
    df,
    coll_name,
    client,
    graph,
    vector_k,
    graph_k,
    get_query_func,
    id2abstract_dict,
    keyword_to_edge_weights,
    hop_penalty,
    batch_size=50,
):
    vector_search_results = vector_search(
        df, coll_name, client, vector_k, get_query_func, id2abstract_dict, batch_size
    )

    graph.define_edge_weight_by_keyword_and_hop_penalty(
        keyword_to_edge_weights, hop_penalty
    )
    graph_search_results = []
    for single_query_results in vector_search_results:
        graph_search_results.append(
            graph.find_relevant_weighted(single_query_results, graph_k)
        )
        if len(graph_search_results[-1]) != graph_k:
            print("WARNING: graph search results not enough")

    return [
        sublist1 + sublist2
        for sublist1, sublist2 in zip(vector_search_results, graph_search_results)
    ]


def infer(
    chroma_path,
    graph,
    workload_csv,
    title_col,
    abstract_col,
    id2abstract_dict,
    k,
    graph_k,
):
    df = pd.read_csv(workload_csv)
    df = df.iloc[:20]
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    vector_search_results = vector_search(
        df, title_col, chroma_client, k, get_query_col, id2abstract_dict
    )
    ground_truths = vector_search(
        df, abstract_col, chroma_client, k, get_abstract_col, id2abstract_dict
    )
    hybrid_search_results = hybrid_search(
        df,
        title_col,
        chroma_client,
        graph,
        k - graph_k,
        graph_k,
        get_query_col,
        id2abstract_dict,
    )
    keyword_to_edge_weights = {
        "author": 1,
        "category": 4,
        "journal": 1,
        "year": 1,
    }
    hop_penalty = 1
    weighted_hybrid_search_results = weighted_hybrid_search(
        df,
        title_col,
        chroma_client,
        graph,
        k - graph_k,
        graph_k,
        get_query_col,
        id2abstract_dict,
        keyword_to_edge_weights,
        hop_penalty,
    )
    df[title_col] = pd.Series(vector_search_results)
    df[abstract_col] = pd.Series(ground_truths)
    df[HYBRID_COL] = pd.Series(hybrid_search_results)
    df[WEIGHTED_HYBRID_COL] = pd.Series(weighted_hybrid_search_results)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="number of k to retrieve for each query",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default="inference_results",
        help="folder path to save the results",
    )
    parser.add_argument(
        "-c",
        "--chroma",
        type=str,
        default="../data/chroma_dbs/",
        help="path to load chroma db collections",
    )
    parser.add_argument(
        "-w",
        "--workloads",
        type=str,
        default="workloads",
        help="workload csv folder",
    )
    parser.add_argument(
        "-a",
        "--abstracts",
        type=str,
        default="abstracts",
        help="ChromaDB collection name that stores the embeddings of abstracts",
    )
    parser.add_argument(
        "-t",
        "--titles",
        type=str,
        default="arxiv_vector",
        help="ChromaDB collection name that stores the embeddings of paper titles",
    )
    parser.add_argument(
        "-f",
        "--filtered-data-path",
        type=str,
        default="../data/filtered_data.pickle",
        help="path to filtered_data pickle file",
    )
    parser.add_argument(
        "-g",
        "--graph",
        type=str,
        default="../data/graph.pickle",
        help="path to graph pickle file",
    )
    parser.add_argument(
        "-l",
        "--graph-k",
        # required=True,
        type=int,
        help="number of k to retrieve for each query from graph",
    )

    args = parser.parse_args()
    assert args.graph_k < args.k
    with open(args.filtered_data_path, "rb") as f:
        filtered_data = pickle.load(f)
    id2title_dict = create_paper_id_to_title_dict(filtered_data)
    id2abstract_dict = create_id_to_abstract_dict(filtered_data)

    if not os.path.exists(args.graph):
        print("Building graph...")
        graph = build_graph(filtered_data)
        with open(args.graph, "wb") as f:
            pickle.dump(graph, f)

    graph = pickle.load(open(args.graph, "rb"))
    print(
        f"Graph has {len(graph.get_data_ids_sorted_by_num_edges())} data points attached to {len(graph.get_keyword_ids_sorted_by_num_edges())} keywords"
    )

    workload_folder = Path(args.workloads)
    for f in workload_folder.iterdir():
        if f.suffix != ".csv":
            continue
        print(f"Processing {f.name}...")
        result_df = infer(
            args.chroma,
            graph,
            workload_folder / f.name,
            args.titles,
            args.abstracts,
            id2abstract_dict,
            args.k,
            args.graph_k,
        )

        # Save the results
        save_folder = Path(args.save)
        if not save_folder.exists():
            save_folder.mkdir()
        result_df.to_csv(save_folder / f.name, index=False)
