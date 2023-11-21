'''
This script should take in a csv file and user selected two columns, and compute some metric

Example: python compute_metrics.py -c test.csv -gt gt_col -pred pred_col
'''

import ast
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import multiprocessing
from sklearn.metrics import accuracy_score

# TODO: double check compute_percent_include
def compute_percent_include(gt, pred):
    '''
    Compute the percentage of ground truth that is included in the prediction
    '''
    counter = 0
    for p in pred:
        if p in gt: counter += 1
    # print(counter)
    # print(len(gt))
    return counter / len(gt)

def batch_compute_percent_include(gt_list, pred_list):
    '''
    Compute the percentage of ground truth that is included in the prediction for a batch of data, return a list of percentages
    '''
    return [compute_percent_include(gt, pred) for gt, pred in zip(gt_list, pred_list)]

def mproc_batch_compute_percent_include(args):
    '''
    Compute the percentage of ground truth that is included in the prediction for a batch of data, multiprocess version
    '''
    gt_list, pred_list, percent_include_list, normalize = args
    percent_include_list.extend(batch_compute_percent_include(gt_list, pred_list))

def compute_accuracy(gt, pred, normalize=True):
    '''
    Compute accuracy
    '''
    return accuracy_score(gt, pred, normalize=normalize)

def batch_compute_accuracy(gt_list, pred_list, normalize=True):
    '''
    Compute accuracy for a batch of data, return a list of accuracies
    '''
    return [accuracy_score(gt, pred, normalize=normalize) for gt, pred in zip(gt_list, pred_list)]

def mproc_batch_compute_accuracy(args):
    '''
    Compute accuracy for a batch of data, multiprocess version
    '''
    gt_list, pred_list, acc_list, normalize = args
    acc_list.extend(batch_compute_accuracy(gt_list, pred_list, normalize=normalize))


if __name__ == '__main__':
    # get a number from the command line
    parser = argparse.ArgumentParser(description="Generate a workload")    
    parser.add_argument('-c', '--csv', type=str, required=True, help='csv file to read')
    parser.add_argument('-gt', '--ground_true', type=str, required=True, help='the name of the ground true column')
    parser.add_argument('-pd', '--pred', type=str, required=True, help='the name of the prediction column')
    parser.add_argument('-p', '--proc', type=int, default=1, help='number of processes to use')
    args = parser.parse_args()

    csv_file = args.csv
    gt_col = args.ground_true
    pred_col = args.pred
    num_processes = args.proc

    df = pd.read_csv(csv_file)

    # compute metrics
    manager = multiprocessing.Manager()

    gt_list = df[gt_col].apply(ast.literal_eval).tolist()
    pred_list = df[pred_col].apply(ast.literal_eval).tolist()
    
    # 1. Accuracy
    acc_normalize = True
    accuracy_list = manager.list()
    # loop through the data a batch at a time
    batch_size = 10000
    args_list = []
    for i in tqdm(range(0, len(gt_list), batch_size)):
        batch_gt = gt_list[i:i+batch_size]
        batch_pred = pred_list[i:i+batch_size]
        # batch_accuracy = batch_compute_accuracy(batch_gt, batch_pred)
        # accuracies.extend(batch_accuracy)
        args_list.append( (batch_gt, batch_pred, accuracy_list, acc_normalize) )
    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.map(mproc_batch_compute_accuracy, args_list)
        pool.map(mproc_batch_compute_percent_include, args_list)

    # accuracies to numpy array
    accuracies = np.array(accuracy_list)
    print(f'Avg accuracy: {accuracies.mean()}')
    print(f'Accuracy std: {accuracies.std()}')
    print(f'Accuracy max: {accuracies.max()}')
    print(f'Accuracy min: {accuracies.min()}')
