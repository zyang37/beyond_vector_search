#!/bin/bash

cd zy_testing/
# Experiment 1
echo "k=5, gk=3"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 5 --graph-k 3 -w ../data/workloads/ -s ../data/results/k5_gk3_outputs -ss 

# Experiment 2
echo "k=10, gk=3"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 10 --graph-k 3 -w ../data/workloads/ -s ../data/results/k10_gk3_outputs -ss 

# Experiment 3
echo "k=10, gk=5"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 10 --graph-k 5 -w ../data/workloads/ -s ../data/results/k10_gk5_outputs -ss 

# Experiment 4
echo "k=10, gk=7"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 10 --graph-k 7 -w ../data/workloads/ -s ../data/results/k10_gk7_outputs -ss 

# Experiment 5
echo "k=50, gk=5"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 50 --graph-k 5 -w ../data/workloads/ -s ../data/results/k50_gk5_outputs -ss 

# Experiment 6
echo "k=50, gk=15"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 50 --graph-k 15 -w ../data/workloads/ -s ../data/results/k50_gk15_outputs -ss 

# Experiment 7
echo "k=50, gk=25"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 50 --graph-k 25 -w ../data/workloads/ -s ../data/results/k50_gk25_outputs -ss 

# Experiment 8
echo "k=50, gk=35"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 50 --graph-k 35 -w ../data/workloads/ -s ../data/results/k50_gk35_outputs -ss 

# Experiment 9
echo "k=100, gk=10"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 100 --graph-k 10 -w ../data/workloads/ -s ../data/results/k100_gk10_outputs -ss 

# Experiment 10
echo "k=100, gk=30"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 100 --graph-k 30 -w ../data/workloads/ -s ../data/results/k100_gk30_outputs -ss 

# Experiment 11
echo "k=100, gk=50"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 100 --graph-k 50 -w ../data/workloads/ -s ../data/results/k100_gk50_outputs -ss 

# Experiment 12
echo "k=100, gk=70"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 100 --graph-k 70 -w ../data/workloads/ -s ../data/results/k100_gk70_outputs -ss 

# Experiment 13
echo "k=500, gk=50"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 500 --graph-k 50 -w ../data/workloads/ -s ../data/results/k500_gk50_outputs -ss 

# Experiment 14
echo "k=500, gk=150"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 500 --graph-k 150 -w ../data/workloads/ -s ../data/results/k500_gk150_outputs -ss 

# Experiment 15
echo "k=500, gk=250"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 500 --graph-k 250 -w ../data/workloads/ -s ../data/results/k500_gk250_outputs -ss 

# Experiment 16
echo "k=500, gk=350"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 500 --graph-k 350 -w ../data/workloads/ -s ../data/results/k500_gk350_outputs -ss 

# Experiment 17
echo "k=1000, gk=100"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 1000 --graph-k 100 -w ../data/workloads/ -s ../data/results/k1000_gk100_outputs -ss 

# Experiment 18
echo "k=1000, gk=300"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 1000 --graph-k 300 -w ../data/workloads/ -s ../data/results/k1000_gk300_outputs -ss 

# Experiment 19
echo "k=1000, gk=500"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 1000 --graph-k 500 -w ../data/workloads/ -s ../data/results/k1000_gk500_outputs -ss 

# Experiment 20
echo "k=1000, gk=700"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 1000 --graph-k 700 -w ../data/workloads/ -s ../data/results/k1000_gk700_outputs -ss 

# ====================================================================================================
# Evaluation 1
echo "k=5, gk=3"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k5_gk3_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k5_gk3_outputs/stats.res --proc 4 

# Evaluation 2
echo "k=10, gk=3"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k10_gk3_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k10_gk3_outputs/stats.res --proc 4 

# Evaluation 3
echo "k=10, gk=5"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k10_gk5_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k10_gk5_outputs/stats.res --proc 4 

# Evaluation 4
echo "k=10, gk=7"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k10_gk7_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k10_gk7_outputs/stats.res --proc 4 

# Evaluation 5
echo "k=50, gk=5"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k50_gk5_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k50_gk5_outputs/stats.res --proc 4 

# Evaluation 6
echo "k=50, gk=15"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k50_gk15_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k50_gk15_outputs/stats.res --proc 4 

# Evaluation 7
echo "k=50, gk=25"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k50_gk25_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k50_gk25_outputs/stats.res --proc 4 

# Evaluation 8
echo "k=50, gk=35"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k50_gk35_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k50_gk35_outputs/stats.res --proc 4 

# Evaluation 9
echo "k=100, gk=10"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k100_gk10_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k100_gk10_outputs/stats.res --proc 4 

# Evaluation 10
echo "k=100, gk=30"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k100_gk30_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k100_gk30_outputs/stats.res --proc 4 

# Evaluation 11
echo "k=100, gk=50"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k100_gk50_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k100_gk50_outputs/stats.res --proc 4 

# Evaluation 12
echo "k=100, gk=70"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k100_gk70_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k100_gk70_outputs/stats.res --proc 4 

# Evaluation 13
echo "k=500, gk=50"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k500_gk50_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k500_gk50_outputs/stats.res --proc 4 

# Evaluation 14
echo "k=500, gk=150"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k500_gk150_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k500_gk150_outputs/stats.res --proc 4 

# Evaluation 15
echo "k=500, gk=250"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k500_gk250_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k500_gk250_outputs/stats.res --proc 4 

# Evaluation 16
echo "k=500, gk=350"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k500_gk350_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k500_gk350_outputs/stats.res --proc 4 

# Evaluation 17
echo "k=1000, gk=100"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k1000_gk100_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k1000_gk100_outputs/stats.res --proc 4 

# Evaluation 18
echo "k=1000, gk=300"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k1000_gk300_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k1000_gk300_outputs/stats.res --proc 4 

# Evaluation 19
echo "k=1000, gk=500"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k1000_gk500_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k1000_gk500_outputs/stats.res --proc 4 

# Evaluation 20
echo "k=1000, gk=700"
python compute_metrics.py -pd arxiv_vector -gt abs_arxiv_vector -hd hybrid -whd weighted_hybrid -f ../data/results/k1000_gk700_outputs -a abs_arxiv_vector -m "all" -s ../data/results/k1000_gk700_outputs/stats.res --proc 4 

