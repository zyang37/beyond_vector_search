#!/bin/bash

python make_vectordb.py -c title_vector --emb title
python make_vectordb.py -c abstract_vector --emb abstract

cd workloads
python query_gen.py -pn 20 -n 2 -s workload.csv
python inference.py -a abstract_vector -t title_vector -k 5 -l -1 -w ../data/workloads -ss
python compute_metrics.py -pd title_vector -gt abstract_vector -hd hybrid -whd weighted_hybrid -f ./inference_results -a abstract_vector -m "all"
