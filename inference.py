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

# sys.path.append("../")
from utils.parse_arxiv import load_json, save_json
# from utils.build_graph import build_graph
# from utils.cnn_news import CnnNewsParser
# from utils.wiki_movie import WikiMoviesParser

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

'''
saving df:   2683.71s user 22.82s system 228% cpu 19:43.88 total, 339M
saving json: slow weighted search, 388M
'''

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

def get_query_col(df, id2gt_dict, id_col):
    return df["query"]

def get_gt_col(df, id2gt_dict, id_col):
    # print(df.head(3))
    # return df[id_col].astype("string").map(id2gt_dict)
    return df["query"]

def vector_search(df, cfg, client, k, get_query_func, id2gt_dict, batch_size=50, id_col="paper_id"):
    coll_name = cfg['collection_name']
    collection = client.get_collection(name=coll_name)
    query_col = get_query_func(df, id2gt_dict, id_col)
    search_results = []
    # print(df.shape[0], batch_size)
    for idx in tqdm(range(0, df.shape[0], batch_size)):
        queries = query_col.iloc[idx : idx + batch_size].values.tolist()
        # try:
        results = collection.query(query_texts=queries, n_results=k)
        # except:
        #     for q in queries:
        #         if type(q) != str: print(q)
        #     exit()
        search_results.extend(results["ids"])
    return search_results


def hybrid_search(
    df,
    cfg,
    client,
    graph,
    vector_k,
    graph_k,
    get_query_func,
    id2gt_dict,
    batch_size=50,
):
    vector_search_results = vector_search(
        df, cfg, client, vector_k, get_query_func, id2gt_dict, batch_size
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
    cfg,
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
        df, cfg, client, vector_k, get_query_func, id2gt_dict, batch_size
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
    cfg,
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
        df, cfg, client, k, get_query_func, id2gt_dict, batch_size
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

def infer(
    cfg,
    graph,
    workload_csv,
    id2gt_dict,
    k,
    graph_k,
    keyword_weights,
    should_sample=False,
):

    chroma_path = cfg['vectorDB']['root']
    vector_coll = cfg['vectorDB']['collection_name']
    gt_coll = cfg['vectorDBGT']['collection_name']
    # id_col = cfg['vectorDB']['id_field']

    df = pd.read_csv(workload_csv)
    if should_sample:
        # df = df.iloc[:150]
        df = df.sample(100)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    # time the inference
    start = time.time()
    vector_search_results = vector_search(
        df, cfg['vectorDB'], chroma_client, k, get_query_col, id2gt_dict
    )
    end = time.time()
    global VECTOR_TIME
    VECTOR_TIME = end - start

    ground_truths = vector_search(
        df, cfg['vectorDBGT'], chroma_client, k, get_gt_col, id2gt_dict
    )

    # time the inference
    start = time.time()
    hybrid_search_results = hybrid_search(
        df,
        cfg['vectorDB'],
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
        cfg['vectorDB'],
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
    
    print("returing df")
    # return df

    df.reset_index(inplace=True) # reset index, keep a copy of the original index
    res_dict = df.to_dict(orient="index")
    for idx, row in res_dict.items():
        row[vector_coll] = vector_search_results[idx]
        row[gt_coll] = ground_truths[idx]
        row[HYBRID_COL] = hybrid_search_results[idx]
        row[WEIGHTED_HYBRID_COL] = weighted_hybrid_search_results[idx]
    return res_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", metavar="", type=str, required=True, help="path to the config file")
    parser.add_argument("-w","--workloads", type=str, required=True, help="workload csv folder")
    parser.add_argument("-k", type=int, required=True, help="number of k to retrieve for each query")
    parser.add_argument("-gk", "--graph-k", required=True, type=int, help="number of k to retrieve for each query from graph",)
    parser.add_argument("-s", "--save", type=str, default="inference_results", help="folder path to save the results")
    parser.add_argument("-ss", "--should_sample", action="store_true", 
                        help="should sample (will only infer first few from each input file)")
    parser.add_argument("-co", "--cut-off",default=-3,type=int,
        help="Priority cutoff, weighted ranked result will only include data whose priority is lower (better) than cutoff")
    parser.add_argument("-kw", "--keyword_weights", default="data/cnn_news/keyword_weights.json", type=str,
        help="json file that stores the weights of the keywords")
    args = parser.parse_args()
    pprint(vars(args))

    # load the config file
    cfg = load_json(args.cfg)
    pprint(cfg)

    chromadb_root = cfg['vectorDB']['root']
    filtered_data_path = cfg['data']['path']
    graph_path = cfg['graphDB']['path']
    gt_id_col = cfg['vectorDBGT']['id_field']
    gt_embed_col = cfg['vectorDBGT']['embed_field']

    assert args.graph_k < args.k

    with open(filtered_data_path, "rb") as f:
        filtered_data = pickle.load(f)
    id2gt_dict = create_id_to_gt_dict(filtered_data, gt_id_col, gt_embed_col)

    # with open(args.keyword_weights, "rb") as f:
    #     keyword_weights = json.load(f)
    
    #TODO: NEED TO CHANGE THIS!!!
    if args.keyword_weights == "data/cnn_news/keyword_weights.json":
        keyword_weights = load_json(args.keyword_weights)
    else:
        keyword_weights = {
            "author": 4,
            "category": 4,
            "journal": 2,
            "year": 1,
        }

    graph = pickle.load(open(graph_path, "rb"))
    print(
        f"Graph has {len(graph.get_data_ids_sorted_by_num_edges())} data points attached to {len(graph.get_keyword_ids_sorted_by_num_edges())} keywords"
    )

    workload_folder = Path(args.workloads)
    for f in workload_folder.iterdir():
        if f.suffix != ".csv":
            continue
        print(f"Processing {f.name}...")
        result_df = infer(
            cfg,
            graph,
            workload_folder / f.name,
            id2gt_dict,
            args.k,
            args.graph_k,
            keyword_weights,
            args.should_sample,
        )
        
        print("saving to json...")
        # Save the results
        save_folder = Path(args.save)
        if not save_folder.exists():
            save_folder.mkdir()
        
        fn = f.name
        fn = fn.replace(".csv", ".json")
        save_json(result_df, save_folder / Path(f"k{args.k}_gk{args.graph_k}_{fn}"), verbose=True)
        # print(f"Saved to {save_folder / Path(f'k{args.k}_gk{graph_k}_{fn}')}\n")
        if args.graph_k != -1:
            break

    # print(f"VECTOR_TIME: {VECTOR_TIME}")
    # print(f"HYBRID_TIME: {HYBRID_TIME}")
    # print(f"WEIGHTED_HYBRID_TIME: {WEIGHTED_HYBRID_TIME}")
    # print()

    # log_runtime(args)
