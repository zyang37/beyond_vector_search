"""
This script should take in a csv file and user selected two columns, and compute some metric

Example: python compute_metrics.py -c test.csv -gt gt_col -pred pred_col
"""

import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import multiprocessing
from sklearn.metrics import accuracy_score
import chromadb


def compute_distance_metrics(gt, pred, abstract_collection):
    gt_results = abstract_collection.get(ids=gt, include=["embeddings"])
    pred_results = abstract_collection.get(ids=pred, include=["embeddings"])
    gt_results = np.array(gt_results["embeddings"])
    pred_results = np.array(pred_results["embeddings"])
    distances = np.linalg.norm((gt_results - pred_results), axis=1)
    return np.mean(distances)


def batch_compute_distance_metrics(gt_list, pred_list, abstract_collection):
    """
    Compute the percentage of ground truth that is included in the prediction for a batch of data, return a list of percentages
    """

    return [
        compute_distance_metrics(gt, pred, abstract_collection)
        for gt, pred in zip(gt_list, pred_list)
    ]


def mproc_batch_compute_distance_metrics(args):
    """
    Compute the percentage of ground truth that is included in the prediction for a batch of data, multiprocess version
    """
    gt_list, pred_list, distance_list, normalize, abstract_collection = args
    distance_list.extend(
        batch_compute_distance_metrics(gt_list, pred_list, abstract_collection)
    )


# TODO: double check compute_percent_include
def compute_percent_include(gt, pred):
    """
    Compute the percentage of ground truth that is included in the prediction
    """
    counter = 0
    for p in pred:
        if p in gt:
            counter += 1
    # print(counter)
    # print(len(gt))
    return counter / len(gt)


def batch_compute_percent_include(gt_list, pred_list):
    """
    Compute the percentage of ground truth that is included in the prediction for a batch of data, return a list of percentages
    """
    return [compute_percent_include(gt, pred) for gt, pred in zip(gt_list, pred_list)]


def mproc_batch_compute_percent_include(args):
    """
    Compute the percentage of ground truth that is included in the prediction for a batch of data, multiprocess version
    """
    gt_list, pred_list, percent_include_list, normalize = args
    percent_include_list.extend(batch_compute_percent_include(gt_list, pred_list))


def compute_accuracy(gt, pred, normalize=True):
    """
    Compute accuracy
    """
    return accuracy_score(gt, pred, normalize=normalize)


def batch_compute_accuracy(gt_list, pred_list, normalize=True):
    """
    Compute accuracy for a batch of data, return a list of accuracies
    """
    return [
        accuracy_score(gt, pred, normalize=normalize)
        for gt, pred in zip(gt_list, pred_list)
    ]


def mproc_batch_compute_accuracy(args):
    """
    Compute accuracy for a batch of data, multiprocess version
    """
    gt_list, pred_list, acc_list, normalize = args
    acc_list.extend(batch_compute_accuracy(gt_list, pred_list, normalize=normalize))


if __name__ == "__main__":
    # get a number from the command line
    parser = argparse.ArgumentParser(description="Generate a workload")
    parser.add_argument("-c", "--csv", type=str, required=True, help="csv file to read")
    parser.add_argument(
        "-gt",
        "--ground_true",
        type=str,
        required=True,
        help="the name of the ground true column",
    )
    parser.add_argument(
        "-pd",
        "--pred",
        type=str,
        required=True,
        help="the name of the prediction column",
    )
    parser.add_argument(
        "-p", "--proc", type=int, default=1, help="number of processes to use"
    )
    parser.add_argument(
        "-hd",
        "--hybrid_pred",
        type=str,
        required=True,
        help="the name of the hybrid prediction column",
    )
    parser.add_argument(
        "-whd",
        "--weighted_hybrid_pred",
        type=str,
        required=True,
        help="the name of the weighted hybrid prediction column",
    )
    parser.add_argument(
        "-ch",
        "--chroma",
        type=str,
        default="../data/chroma_dbs/",
        help="path to load chroma db collections",
    )
    parser.add_argument(
        "-a",
        "--abstracts",
        type=str,
        default="abstracts",
        help="ChromaDB collection name that stores the embeddings of abstracts",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        default="distances",
        help="Specify the metrics to use. Currently supporting accuracy, percent_include and distances",
    )
    args = parser.parse_args()

    num_processes = args.proc
    csv_file = args.csv
    df = pd.read_csv(csv_file)

    potential_args_to_eval = [
        args.ground_true,
        args.pred,
        args.hybrid_pred,
        args.weighted_hybrid_pred,
    ]
    cols_to_eval = []
    for arg in potential_args_to_eval:
        if arg is not None and arg in df.columns:
            cols_to_eval.append(arg)

    # compute metrics
    manager = multiprocessing.Manager()

    lists_to_eval = []
    for col in cols_to_eval:
        lists_to_eval.append(df[col].apply(ast.literal_eval).tolist())

    # 1. Accuracy
    acc_normalize = True
    accuracy_lists = []
    args_lists = []
    for i in range(len(lists_to_eval)):
        accuracy_lists.append(manager.list())
        args_lists.append([])

    # loop through the data a batch at a time
    batch_size = 10000
    abstract_collection = None
    if args.metrics == "distances":
        client = chromadb.PersistentClient(path=args.chroma)
        abstract_collection = client.get_collection(name=args.abstracts)

    for i in tqdm(range(0, len(lists_to_eval[0]), batch_size)):
        batches = []
        for j in range(len(lists_to_eval)):
            batches.append(lists_to_eval[j][i : i + batch_size])

        for j in range(1, len(lists_to_eval)):
            if args.metrics == "distances":
                args_lists[j].append(
                    (
                        batches[0],  # We assume the first batch is the ground truth
                        batches[j],
                        accuracy_lists[j],
                        acc_normalize,
                        abstract_collection,
                    )
                )
            else:
                args_lists[j].append(
                    (
                        batches[0],  # We assume the first batch is the ground truth
                        batches[j],
                        accuracy_lists[j],
                        acc_normalize,
                    )
                )

    # Actually compute the metrics
    if args.metrics == "distances":
        for i in range(1, len(args_lists)):
            for arg in args_lists[i]:
                mproc_batch_compute_distance_metrics(arg)
    elif args.metrics == "accuracy":
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(1, len(args_lists)):
                for arg in args_lists[i]:
                    pool.map(mproc_batch_compute_accuracy, arg)
    else:  # percent_include
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(1, len(args_lists)):
                for arg in args_lists[i]:
                    pool.map(mproc_batch_compute_percent_include, arg)

    """
    for arg in args_list_vector:
        mproc_batch_compute_distance_metrics(arg)

    for arg in args_list_hybrid:
        mproc_batch_compute_distance_metrics(arg)
    """

    for i in range(1, len(accuracy_lists)):
        accuracy = np.array(accuracy_lists[i])
        print(f"{cols_to_eval[i]} ACCURACY:")
        print(f"Avg accuracy: {accuracy.mean()}")
        print(f"Accuracy std: {accuracy.std()}")
        print(f"Accuracy max: {accuracy.max()}")
        print(f"Accuracy min: {accuracy.min()}\n")
