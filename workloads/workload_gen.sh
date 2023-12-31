#!/bin/bash
python query_gen.py -pn 646 -n 20 -s ../data/workloads/cv0_05_num20_prob0_1.csv --prob cfgs/prob_0.1.json
python query_gen.py -pn 646 -n 20 -s ../data/workloads/cv0_05_num20_prob0_3.csv --prob cfgs/prob_0.3.json
python query_gen.py -pn 646 -n 20 -s ../data/workloads/cv0_05_num20_prob0_5.csv --prob cfgs/prob_0.5.json
python query_gen.py -pn 646 -n 20 -s ../data/workloads/cv0_05_num20_prob1_0.csv --prob cfgs/prob_1.0.json
python query_gen.py -pn 1292 -n 10 -s ../data/workloads/cv0_1_num10_prob0_1.csv --prob cfgs/prob_0.1.json
python query_gen.py -pn 1292 -n 10 -s ../data/workloads/cv0_1_num10_prob0_3.csv --prob cfgs/prob_0.3.json
python query_gen.py -pn 1292 -n 10 -s ../data/workloads/cv0_1_num10_prob0_5.csv --prob cfgs/prob_0.5.json
python query_gen.py -pn 1292 -n 10 -s ../data/workloads/cv0_1_num10_prob1_0.csv --prob cfgs/prob_1.0.json
python query_gen.py -pn 3877 -n 5 -s ../data/workloads/cv0_3_num5_prob0_1.csv --prob cfgs/prob_0.1.json
python query_gen.py -pn 3877 -n 5 -s ../data/workloads/cv0_3_num5_prob0_3.csv --prob cfgs/prob_0.3.json
python query_gen.py -pn 3877 -n 5 -s ../data/workloads/cv0_3_num5_prob0_5.csv --prob cfgs/prob_0.5.json
python query_gen.py -pn 3877 -n 5 -s ../data/workloads/cv0_3_num5_prob1_0.csv --prob cfgs/prob_1.0.json
python query_gen.py -pn 6463 -n 3 -s ../data/workloads/cv0_5_num3_prob0_1.csv --prob cfgs/prob_0.1.json
python query_gen.py -pn 6463 -n 3 -s ../data/workloads/cv0_5_num3_prob0_3.csv --prob cfgs/prob_0.3.json
python query_gen.py -pn 6463 -n 3 -s ../data/workloads/cv0_5_num3_prob0_5.csv --prob cfgs/prob_0.5.json
python query_gen.py -pn 6463 -n 3 -s ../data/workloads/cv0_5_num3_prob1_0.csv --prob cfgs/prob_1.0.json
python query_gen.py -pn 9048 -n 2 -s ../data/workloads/cv0_7_num2_prob0_1.csv --prob cfgs/prob_0.1.json
python query_gen.py -pn 9048 -n 2 -s ../data/workloads/cv0_7_num2_prob0_3.csv --prob cfgs/prob_0.3.json
python query_gen.py -pn 9048 -n 2 -s ../data/workloads/cv0_7_num2_prob0_5.csv --prob cfgs/prob_0.5.json
python query_gen.py -pn 9048 -n 2 -s ../data/workloads/cv0_7_num2_prob1_0.csv --prob cfgs/prob_1.0.json
