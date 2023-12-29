#!/bin/bash

cd zy_testing

# python inference_cnn.py -k 100 -l 10 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
# python inference_cnn.py -k 100 -l 50 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
# python inference_cnn.py -k 300 -l 50 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
# python inference_cnn.py -k 300 -l 150 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
# python inference_cnn.py -k 500 -l 100 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
# python inference_cnn.py -k 500 -l 150 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
# python inference_cnn.py -k 500 -l 250 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
# python inference_cnn.py -k 1000 -l 100 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
# python inference_cnn.py -k 1000 -l 250 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss
python inference_cnn.py -k 1000 -l 500 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss --cut-off -20
# python inference_cnn.py -k 1000 -l 700 -w ../data/cnn_workloads -s ../data/cnn_news/res -ss

python compute_metrics_cos.py -f ../data/cnn_news/res -gt cnn_news_smallGT -pd cnn_news_small -hd hybrid -whd weighted_hybrid -a cnn_news_smallGT -s ../data/cnn_news/res/stats.res -m "all"

cp ../data/cnn_news/res/stats.res ../data/cnn_news/stats.csv