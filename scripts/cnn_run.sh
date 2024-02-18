#!/bin/bash


python inference.py -c config/cnn_cfg.json -w data/cnn_news/cnn_workloads -kw cnn -k 100 -gk 25 -s data/cnn_news/cnn_workloads/res