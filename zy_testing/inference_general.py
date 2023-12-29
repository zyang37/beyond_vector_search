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
from pprint import pprint
import time
import random
import numpy as np
import json

sys.path.append("../")
from utils.build_graph import build_graph
from utils.cnn_news import CnnNewsParser
from utils.wiki_movie import WikiMoviesParser

# fix random seed for reproducibility
random.seed(1)
np.random.seed(1)


global RUNTIME_pickle

VECTOR_TIME = 0.0
HYBRID_TIME = 0.0
WEIGHTED_HYBRID_TIME = 0.0
RUNTIME_pickle = "runtime.pickle"


HYBRID_COL = "hybrid"
WEIGHTED_HYBRID_COL = "weighted_hybrid"


def log_runtime(args):
    """
    Note: The order of running search method matters. since ChromaDB will cache the results.

    check if runtime.pickle exists
    if not, create one and save the runtime
    if yes, load the runtime and add runtime
    Save k and gk to runtime.pickle
    """
    if not os.path.exists(RUNTIME_pickle):
        runtime = [
            {
                "k": args.k,
                "gk": args.graph_k,
                "VECTOR_TIME": VECTOR_TIME,
                "HYBRID_TIME": HYBRID_TIME,
                "WEIGHTED_HYBRID_TIME": WEIGHTED_HYBRID_TIME,
            }
        ]
        with open(RUNTIME_pickle, "wb") as f:
            pickle.dump(runtime, f)
    else:
        with open(RUNTIME_pickle, "rb") as f:
            runtime = pickle.load(f)
        runtime.append(
            {
                "k": args.k,
                "gk": args.graph_k,
                "VECTOR_TIME": VECTOR_TIME,
                "HYBRID_TIME": HYBRID_TIME,
                "WEIGHTED_HYBRID_TIME": WEIGHTED_HYBRID_TIME,
            }
        )
        with open(RUNTIME_pickle, "wb") as f:
            pickle.dump(runtime, f)


def create_id_to_gt_dict(filtered_data, id_col, gt_col):
    return dict(zip(filtered_data[id_col].astype("string"), filtered_data[gt_col]))


def get_query_col(df, id2gt_dict):
    return df["query"]


def get_gt_col(df, id2gt_dict, id_col):
    return df[id_col].astype("string").map(id2gt_dict)


def vector_search(
    df, coll_name, client, k, get_query_func, id2gt_dict, batch_size=50, id_col=None
):
    collection = client.get_collection(name=coll_name)
    query_col = get_query_func(df, id2gt_dict, id_col)
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
    id2gt_dict,
    batch_size=50,
):
    vector_search_results = vector_search(
        df, coll_name, client, vector_k, get_query_func, id2gt_dict, batch_size
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
    id2gt_dict,
    keyword_to_edge_weights,
    hop_penalty,
    batch_size=50,
):
    vector_search_results = vector_search(
        df, coll_name, client, vector_k, get_query_func, id2gt_dict, batch_size
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


def weighted_hybrid_search_cut_off(
    df,
    coll_name,
    client,
    graph,
    k,
    graph_k,
    get_query_func,
    id2gt_dict,
    keyword_to_edge_weights,
    hop_penalty,
    batch_size=50,
    cut_off=0,
):
    vector_search_results = vector_search(
        df, coll_name, client, k, get_query_func, id2gt_dict, batch_size
    )

    graph.define_edge_weight_by_keyword_and_hop_penalty(
        keyword_to_edge_weights, hop_penalty
    )
    graph_search_results = []

    for single_query_results in vector_search_results:
        # print(single_query_results)
        graph_search_results.append(
            graph.find_relevant_weighted_ranked(
                single_query_results[: k - graph_k], graph_k, cut_off
            )
        )

    return [
        sublist1[: k - len(sublist2)] + sublist2
        for sublist1, sublist2 in zip(vector_search_results, graph_search_results)
    ]


def list_to_str(l):
    # given a python list obj: [1,2,3]
    # return a string: "[1,2,3]"
    # pandas can not handle list obj on a single cell...
    return str(l).replace(" ", "")


def reslist2liststr(reslist):
    # reslist 2d list
    # return a 1d list of string_list
    return [list_to_str(x) for x in reslist]


def infer(
    chroma_path,
    graph,
    workload_csv,
    id_col,
    embed_col,
    gt_col,
    id2gt_dict,
    k,
    graph_k,
    keyword_weights,
    should_sample=False,
):
    df = pd.read_csv(workload_csv)
    if should_sample:
        # df = df.iloc[:150]
        df = df.sample(100)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    # time the inference
    start = time.time()
    vector_search_results = vector_search(
        df, embed_col, chroma_client, k, get_query_col, id2gt_dict
    )
    end = time.time()
    global VECTOR_TIME
    VECTOR_TIME = end - start

    ground_truths = vector_search(
        df, gt_col, chroma_client, k, get_gt_col, id2gt_dict, id_col
    )

    # time the inference
    start = time.time()
    hybrid_search_results = hybrid_search(
        df,
        embed_col,
        chroma_client,
        graph,
        k - graph_k,
        graph_k,
        get_query_col,
        id2gt_dict,
    )
    end = time.time()
    global HYBRID_TIME
    HYBRID_TIME = end - start

    hop_penalty = 1

    # time the inference
    start = time.time()
    weighted_hybrid_search_results = weighted_hybrid_search_cut_off(
        df,
        embed_col,
        chroma_client,
        graph,
        k,
        graph_k,
        get_query_col,
        id2gt_dict,
        keyword_weights,
        hop_penalty,
        cut_off=args.cut_off,
    )
    end = time.time()
    global WEIGHTED_HYBRID_TIME
    WEIGHTED_HYBRID_TIME = end - start

    vector_search_results = reslist2liststr(vector_search_results)
    ground_truths = reslist2liststr(ground_truths)
    hybrid_search_results = reslist2liststr(hybrid_search_results)
    weighted_hybrid_search_results = reslist2liststr(weighted_hybrid_search_results)

    df[embed_col] = vector_search_results
    df[gt_col] = ground_truths
    df[HYBRID_COL] = hybrid_search_results
    df[WEIGHTED_HYBRID_COL] = weighted_hybrid_search_results

    # df[title_col] = pd.Series(vector_search_results)
    # df[abstract_col] = pd.Series(ground_truths)
    # df[HYBRID_COL] = pd.Series(hybrid_search_results)
    # df[WEIGHTED_HYBRID_COL] = pd.Series(weighted_hybrid_search_results)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="axriv",
        help="type of dataset to run inference on (arxiv, cnn, wiki)",
    )
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
        "-id",
        type=str,
        default="Title",
        help="Column name that represents the id of the data frame",
    )
    parser.add_argument(
        "-gt",
        "--ground_truths",
        type=str,
        default="abstracts",
        help="ChromaDB collection name that stores the embeddings of the ground truths",
    )
    parser.add_argument(
        "-emb",
        "--embed",
        type=str,
        default="arxiv_vector",
        help="ChromaDB collection name that stores the embeddings of the dataset",
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
        # required=True,
        default="../data/graph.pickle",
        help="path to graph.pickle",
    )
    parser.add_argument(
        "-l",
        "--graph-k",
        # required=True,
        type=int,
        help="number of k to retrieve for each query from graph",
    )
    parser.add_argument(
        "-co",
        "--cut-off",
        default=-3,
        type=int,
        help="Priority cutoff, weighted ranked result will only include data whose priority is lower (better) than cutoff",
    )
    parser.add_argument(
        "-ss",
        "--should_sample",
        action="store_true",
        help="should sample (will only infer first 20 from each input file)",
    )
    parser.add_argument(
        "-kw",
        "--keyword_weights",
        default="../data/wiki_movies/keyword_weights.json",
        type=str,
        help="json file that stores the weights of the keywords",
    )

    args = parser.parse_args()
    pprint(vars(args))

    assert args.graph_k < args.k
    with open(args.filtered_data_path, "rb") as f:
        filtered_data = pickle.load(f)
    id2gt_dict = create_id_to_gt_dict(filtered_data, args.id, args.ground_truths)

    with open(args.keyword_weights, "rb") as f:
        keyword_weights = json.load(f)

    # adjust the root folder of the graph path, so user can just specify the filter data path,
    # and the graph path will be automatically adjusted, if it is not specified
    if args.graph.split("/")[:-1] != args.filtered_data_path.split("/")[:-1]:
        print(
            "Adjusting graph path to match filtered_data_path: {}".format(
                args.filtered_data_path.split("/")[:-1]
            )
        )
        args.graph = os.path.join(
            "/".join(args.filtered_data_path.split("/")[:-1]), args.graph.split("/")[-1]
        )
        print("New graph path: {}".format(args.graph))

    if not os.path.exists(args.graph):
        print("Building graph...")
        if args.type == "arxiv":
            graph = build_graph(filtered_data)
        elif args.type == "cnn":
            cnn_news_parser = CnnNewsParser(filtered_data)
            graph = cnn_news_parser.G
        elif args.type == "wiki":
            wiki_movies_parser = WikiMoviesParser(filtered_data)
            graph = wiki_movies_parser.G
        else:
            raise ValueError(f"Unknown type {args.type}")

        with open(args.graph, "wb") as f:
            pickle.dump(graph, f)

    graph = pickle.load(open(args.graph, "rb"))
    print(
        f"Graph has {len(graph.get_data_ids_sorted_by_num_edges())} data points attached to {len(graph.get_keyword_ids_sorted_by_num_edges())} keywords"
    )

    workload_folder = Path(args.workloads)
    for f in workload_folder.iterdir():
        for i in range(1, args.k, 4):
            graph_k = i
            if args.graph_k != -1:
                graph_k = args.graph_k
            if f.suffix != ".csv":
                continue
            print(f"Processing {f.name}...")
            result_df = infer(
                args.chroma,
                graph,
                workload_folder / f.name,
                args.id,
                args.embed,
                args.ground_truths,
                id2gt_dict,
                args.k,
                graph_k,
                keyword_weights,
                args.should_sample,
            )

            # Save the results
            save_folder = Path(args.save)
            if not save_folder.exists():
                save_folder.mkdir()
            # result_df.to_csv(save_folder / Path(f"{graph_k}_{f.name}"), index=False)
            result_df.to_csv(
                save_folder / Path(f"k{args.k}_gk{graph_k}_{f.name}"), index=False
            )
            print(f"Saved to {save_folder / Path(f'k{args.k}_gk{graph_k}_{f.name}')}\n")

            if args.graph_k != -1:
                break

    print(f"VECTOR_TIME: {VECTOR_TIME}")
    print(f"HYBRID_TIME: {HYBRID_TIME}")
    print(f"WEIGHTED_HYBRID_TIME: {WEIGHTED_HYBRID_TIME}")
    print()

    log_runtime(args)
