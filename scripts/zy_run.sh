#!/bin/bash

# python make_vectordb.py -c title_vector --emb title
# python make_vectordb.py -c abstract_vector --emb abstract

# cd workloads
# python query_gen.py -pn 20 -n 2 -s workload.csv
# python inference.py -a abs_arxiv_vector -t arxiv_vector -k 10 -s res.csv
# python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -c res.csv

# new 
cd zy_testing
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 5 -l -1 -w ../data/workloads -s ../data/workloads/inference_results
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid \
                          -f ../data/workloads/inference_results -a abs_arxiv_vector -m "all" \
                          -s ../data/workloads/inference_results_stats.csv