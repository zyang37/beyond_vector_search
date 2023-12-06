'''
cd zy_testing
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 5 --graph-k 3 -w ../data/workloads \
                          -s ../data/results/k5_gk3_outputs -ss

python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid \
                          -f ../data/results/k5_gk3_outputs -a abs_arxiv_vector -m "all" \
                          -s ../data/results/k5_gk3_outputs/stats.res
'''

import os
import sys

# adjust the following parameters as needed
abstract_db_name = "abs_arxiv_vector"
title_db_name = "arxiv_vector"
workload_folder = "../data/workloads/"
save_folder_root = "../data/results/"

testing_dic = "zy_testing/"

gk_ratios = [0.1, 0.3, 0.5, 0.7]
k_gk_dict = {5: [3], 10: [3, 5, 7], 
             50: [int(r*50) for r in gk_ratios], 
             100: [int(r*100) for r in gk_ratios], 
             500: [int(r*500) for r in gk_ratios], 
             1000: [int(r*1000) for r in gk_ratios]}

if __name__ == "__main__":
    '''
    print the bash commands for running inference.py and compute_metrics.py, for many different k and graph_k values
    '''
    print("#!/bin/bash")
    print()
    print("cd " + testing_dic)
    exp_counter = 1

    echo_msg = []
    inference_cmds = []
    compute_metrics_cmds = []

    for k, gk_list in k_gk_dict.items():
        for gk in gk_list:
            save_folder = os.path.join(save_folder_root, f"k{k}_gk{gk}_outputs")

            echo_str = "echo \"k=" + str(k) + ", gk=" + str(gk) + "\""
            echo_msg.append(echo_str)
            # construct command 1st, use "+"
            # to run all queries in the workload, remove "-ss"
            inference_cmd = "python inference.py -a " + abstract_db_name + \
                            " -t " + title_db_name + " -k " + str(k) + " --graph-k " + str(gk) + \
                            " -w " + workload_folder + " -s " + save_folder + " -ss "
            
            compute_metrics_cmd = "python compute_metrics.py -pd " + title_db_name + \
                                  " -gt " + abstract_db_name + " -hd hybrid -whd weighted_hybrid -f " + \
                                  save_folder + " -a " + abstract_db_name + " -m \"all\" -s " + save_folder + "/stats.res" + \
                                  " --proc 4 "

            inference_cmds.append(inference_cmd)
            compute_metrics_cmds.append(compute_metrics_cmd)
            # print("# Experiment " + str(exp_counter))
            # exp_counter += 1
            # print(inference_cmd)
            # print(compute_metrics_cmd)
            # print()

    for cmd, msg in zip(inference_cmds, echo_msg):
        print("# Experiment " + str(exp_counter))
        exp_counter += 1
        # echo "k/gk"
        print(msg)
        print(cmd)
        print()
    
    print("#", "="*100)
    # eval sometimes hang, so we run them separately
    exp_counter = 1
    for cmd, msg in zip(compute_metrics_cmds, echo_msg):
        print("# Evaluation " + str(exp_counter))
        exp_counter += 1
        # echo "k/gk"
        print(msg)
        print(cmd)
        print()