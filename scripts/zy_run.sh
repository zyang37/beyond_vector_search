#!/bin/bash

# python make_vectordb.py -c title_vector --emb title
# python make_vectordb.py -c abstract_vector --emb abstract

# cd workloads
# python query_gen.py -pn 20 -n 2 -s workload.csv
# python inference.py -a abs_arxiv_vector -t arxiv_vector -k 10 -s res.csv
# python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -c res.csv

# new 
cd zy_testing
# python inference.py -a abs_arxiv_vector -t arxiv_vector -k 500 --graph-k 250 -w ../data/workloads/ -s ../tmp/k500_gk250_outputs -ss 
python compute_metrics_cos.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid \
                          -f ../tmp/k500_gk250_outputs -a abs_arxiv_vector -m "distances" \
                          -s ../tmp/k500_gk250_outputs/stats.res