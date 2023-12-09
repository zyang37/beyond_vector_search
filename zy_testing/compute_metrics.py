"""
This script should take in a csv file and user selected two columns, and compute some metric

Example: python compute_metrics.py -c test.csv -gt gt_col -pred pred_col
"""

import ast
import argparse
import numpy as np
import pandas as pd
from requests import get
from tqdm.auto import tqdm
from pprint import pprint
import multiprocessing
from sklearn.metrics import accuracy_score
import chromadb
from pathlib import Path


def compute_distance_metrics(gt, pred, abstract_collection):
    gt_results = abstract_collection.get(ids=gt, include=["embeddings"])
    pred_results = abstract_collection.get(ids=pred, include=["embeddings"])
    gt_results = np.array(gt_results["embeddings"])
    pred_results = np.array(pred_results["embeddings"])
    gt_results = gt_results[: pred_results.shape[0], :]
    distances = np.linalg.norm((gt_results - pred_results), axis=1)
    return np.mean(distances)


def batch_compute_distance_metrics(gt_list, pred_list, abstract_collection):
    """
    Compute the percentage of ground truth that is included in the prediction for a batch of data, return a list of percentages
    """
    dists = []
    for gt, pred in zip(gt_list, pred_list):
        if len(gt) != len(pred):
            print("size mismatch, add cuur mean to the list")
            dists.append(np.mean(dists))
            continue
        dists.append( compute_distance_metrics(gt, pred, abstract_collection) )
    
    return dists
    # return [
    #     compute_distance_metrics(gt, pred, abstract_collection)
    #     for gt, pred in zip(gt_list, pred_list)
    # ]


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
    accs = []
    for gt, pred in zip(gt_list, pred_list):
       if len(gt) != len(pred):
           print("size mismatch, add cuur mean to the list")
           accs.append(np.mean(accs))
           continue
       accs.append( accuracy_score(gt, pred, normalize=normalize) )
    return accs


def mproc_batch_compute_accuracy(args):
    """
    Compute accuracy for a batch of data, multiprocess version
    """
    gt_list, pred_list, acc_list, normalize = args

    # DEBUG
    # conter = 0
    # for i in range(len(gt_list)):
    #     if len(gt_list[i]) != len(pred_list[i]): 
    #         conter += 1
            # print(i)
            # print(len(gt_list[i]))
            # print(len(pred_list[i]))
            # print()
    # print("counter: ", conter)

    acc_list.extend(batch_compute_accuracy(gt_list, pred_list, normalize=normalize))


FILENAME_COL = "filename"


def get_col_name(metric_name, query_name, stat_name):
    return f"{metric_name}_{query_name}_{stat_name}"


def append_results(results_df, accuracy_lists, cols_to_eval, metric_name):
    cur_col = 1

    for accuracy_list in accuracy_lists:
        accuracy = np.array(accuracy_list)
        query_name = cols_to_eval[cur_col]
        row_loc = results_df.index[-1]

        results_df.loc[
            row_loc, get_col_name(metric_name, query_name, "mean")
        ] = accuracy.mean()
        results_df.loc[
            row_loc, get_col_name(metric_name, query_name, "std")
        ] = accuracy.std()
        results_df.loc[
            row_loc, get_col_name(metric_name, query_name, "max")
        ] = accuracy.max()
        results_df.loc[
            row_loc, get_col_name(metric_name, query_name, "min")
        ] = accuracy.min()

        cur_col += 1


def get_cols_to_eval(args, df):
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
    return cols_to_eval


def get_metrics_to_eval(args):
    metrics_to_eval = []
    if args.metrics == "distances":
        metrics_to_eval.append("distances")
    elif args.metrics == "accuracy":
        metrics_to_eval.append("accuracy")
    elif args.metrics == "include":
        metrics_to_eval.append("percent_include")
    else:
        metrics_to_eval.append("distances")
        metrics_to_eval.append("accuracy")
        metrics_to_eval.append("percent_include")
    return metrics_to_eval


def process_metric(args, df, results_df, metric_name):
    print(f"Processing {metric_name}")
    cols_to_eval = get_cols_to_eval(args, df)

    # compute metrics
    manager = multiprocessing.Manager()

    lists_to_eval = []
    for col in cols_to_eval:
        # convert the string to a list
        output_list = df[col].apply(ast.literal_eval).tolist()
        lists_to_eval.append(output_list)

    # 1. Accuracy
    acc_normalize = True
    accuracy_lists = []
    args_lists = []
    for i in range(1, len(lists_to_eval)):
        accuracy_lists.append(manager.list())
        args_lists.append([])

    # loop through the data a batch at a time
    batch_size = 3000
    abstract_collection = None
    if metric_name == "distances":
        client = chromadb.PersistentClient(path=args.chroma)
        abstract_collection = client.get_collection(name=args.abstracts)

    for i in tqdm(range(0, len(lists_to_eval[0]), batch_size)):
        batches = []
        for j in range(len(lists_to_eval)):
            batches.append(lists_to_eval[j][i : i + batch_size])

        for j in range(len(args_lists)):
            if metric_name == "distances":
                args_lists[j].append(
                    (
                        batches[0],  # We assume the first batch is the ground truth
                        batches[j + 1],
                        accuracy_lists[j],
                        acc_normalize,
                        abstract_collection,
                    )
                )
            else:
                args_lists[j].append(
                    (
                        batches[0],  # We assume the first batch is the ground truth
                        batches[j + 1],
                        accuracy_lists[j],
                        acc_normalize,
                    )
                )

    if metric_name == "distances":
        for args_list in args_lists:
            for arg in args_list:
                mproc_batch_compute_distance_metrics(arg)
        append_results(results_df, accuracy_lists, cols_to_eval, "distances")
    elif metric_name == "accuracy":
        with multiprocessing.Pool(processes=num_processes) as pool:
            for a_list in args_lists:
                for arg in a_list:
                    pool.map(mproc_batch_compute_accuracy, [arg])
        append_results(results_df, accuracy_lists, cols_to_eval, "accuracy")
    elif metric_name == "percent_include":
        with multiprocessing.Pool(processes=num_processes) as pool:
            for args_list in args_lists:
                for arg in args_list:
                    pool.map(mproc_batch_compute_percent_include, [arg])
        append_results(results_df, accuracy_lists, cols_to_eval, "percent_include")


if __name__ == "__main__":
    # get a number from the command line
    parser = argparse.ArgumentParser(description="Generate a workload")
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="folder to read csv file from",
    )
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
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default="stats.csv",
        help="file to save the results to",
    )
    args = parser.parse_args()
    num_processes = args.proc
    metrics_to_eval = get_metrics_to_eval(args)
    results_df = pd.DataFrame()
    inference_res_folder = Path(args.folder)

    for f in inference_res_folder.iterdir():
        if f.suffix != ".csv":
            continue
        print(f"Processing {f.name}")
        df = pd.read_csv(f)

        # Add the column names
        if results_df.shape[1] == 0:
            columns = ["filename"]
            cols_to_eval = get_cols_to_eval(args, df)
            for metric in metrics_to_eval:
                for col in cols_to_eval[1:]:
                    columns.append(get_col_name(metric, col, "mean"))
                    columns.append(get_col_name(metric, col, "std"))
                    columns.append(get_col_name(metric, col, "max"))
                    columns.append(get_col_name(metric, col, "min"))
            results_df = pd.DataFrame(columns=columns)
        row = [f.name]
        for i in range(results_df.shape[1] - 1):
            row.append(np.nan)
        
        # results_df = results_df.append(
        #     pd.Series(row, index=results_df.columns), ignore_index=True
        # )
        new_row = pd.Series(row, index=results_df.columns)
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


        for metric in metrics_to_eval:
            process_metric(args, df, results_df, metric)

    # Save the results
    results_df.to_csv(args.save, index=False)
