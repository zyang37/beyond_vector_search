#!/bin/bash

# python make_vectordb.py -c title_vector --emb title
# python make_vectordb.py -c abstract_vector --emb abstract

mkdir data/example
cd workloads
python query_gen.py -pn 20 -n 2 -s ../data/example/tmp_workload.csv
cd ../testing
python inference.py -a abstract_vector -t title_vector -k 5 -l -1 -w ../data/example -ss -s ../data/example/inference_results
cd ../zy_testing
python compute_metrics_cos.py -pd title_vector -gt abstract_vector -hd hybrid -whd weighted_hybrid -f ../data/example/inference_results -a abstract_vector -m "distances"
