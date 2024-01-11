#!/bin/bash
python query_gen.py -c ../config/cfg_base.json -pn 50 -n 20 -s ../data/arxiv/arxiv_workloads/pn50_n20.csv
python query_gen.py -c ../config/cfg_base.json -pn 100 -n 10 -s ../data/arxiv/arxiv_workloads/pn100_n10.csv
python query_gen.py -c ../config/cfg_base.json -pn 200 -n 5 -s ../data/arxiv/arxiv_workloads/pn200_n5.csv
python query_gen.py -c ../config/cfg_base.json -pn 1000 -n 1 -s ../data/arxiv/arxiv_workloads/pn1000_n1.csv
