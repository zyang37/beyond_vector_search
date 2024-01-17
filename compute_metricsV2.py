""" IN PROGRESS!!!
This script should take in a csv file and user selected two columns, and compute some metric

Example: python compute_metrics.py -c test.csv -gt gt_col -pred pred_col
"""

import chromadb
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from pprint import pprint
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine as cosine_similarity

from utils.parse_arxiv import load_json, save_json

if __name__ == "__main__":
    # get a number from the command line
    parser = argparse.ArgumentParser(description="Compute metrics")
    parser.add_argument("-c", "--cfg", metavar="", type=str, default=None, help="path to the config file")
    parser.add_argument("-f", "--folder", type=str, required=True, help="folder to read csv file from")
    parser.add_argument("-p", "--proc", type=int, default=1, help="number of processes to use")
    parser.add_argument("-hd", "--hybrid_pred", type=str, default="hybrid", help="the name of the hybrid prediction column")
    parser.add_argument("-whd", "--weighted_hybrid_pred", type=str, default="weighted_hybrid", 
        help="the name of the weighted hybrid prediction column")
    parser.add_argument("-m", "--metrics", type=str, default="all",
        help="Specify the metrics to use. Currently supporting accuracy, percent_include and distances")
    parser.add_argument( "-s", "--save", type=str, default="stats.csv", help="file to save the results to")
    args = parser.parse_args()

    cfg = load_json(args.cfg)

    num_processes = args.proc
    metrics_to_eval = get_metrics_to_eval(args)
    results_df = pd.DataFrame()
    inference_res_folder = Path(args.folder)
    