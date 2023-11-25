#!/bin/bash

# python make_vectordb.py -c title_vector --emb title
# python make_vectordb.py -c abstract_vector --emb abstract

cd workloads
python query_gen.py -pn 20 -n 2 -s workload.csv
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 10 -s res.csv
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -c res.csv
