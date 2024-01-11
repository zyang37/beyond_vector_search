#!/bin/bash

# cd zy_testing

python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -k 10 -gk 5 -s data/cnn_news/cnn_workloads/res
python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -k 100 -gk 25 -s data/cnn_news/cnn_workloads/res
python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -k 100 -gk 50 -s data/cnn_news/cnn_workloads/res
python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -k 500 -gk 100 -s data/cnn_news/cnn_workloads/res
python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -k 500 -gk 250 -s data/cnn_news/cnn_workloads/res
python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -k 1000 -gk 100 -s data/cnn_news/cnn_workloads/re
python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -k 1000 -gk 250 -s data/cnn_news/cnn_workloads/re
python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -k 1000 -gk 500 -s data/cnn_news/cnn_workloads/re

# python compute_metrics.py -f ../data/cnn_news/res -gt cnn_news_smallGT -pd cnn_news_small -hd hybrid -whd weighted_hybrid -a cnn_news_smallGT -s ../data/cnn_news/res/stats.res -m "all"

# cp ../data/cnn_news/res/stats.res ../data/cnn_news/stats.csv