#!/bin/bash

python inference.py -c config/cfg_base.json -w data/arxiv/arxiv_workloads -s data/arxiv/arxiv_workloads/res -k 100 -gk 20
python inference.py -c config/cfg_base.json -w data/arxiv/arxiv_workloads -s data/arxiv/arxiv_workloads/res -k 100 -gk 50

python inference.py -c config/cfg_base.json -w data/arxiv/arxiv_workloads -s data/arxiv/arxiv_workloads/res -k 500 -gk 100
python inference.py -c config/cfg_base.json -w data/arxiv/arxiv_workloads -s data/arxiv/arxiv_workloads/res -k 500 -gk 250

python inference.py -c config/cfg_base.json -w data/arxiv/arxiv_workloads -s data/arxiv/arxiv_workloads/res -k 1000 -gk 100
python inference.py -c config/cfg_base.json -w data/arxiv/arxiv_workloads -s data/arxiv/arxiv_workloads/res -k 1000 -gk 300
python inference.py -c config/cfg_base.json -w data/arxiv/arxiv_workloads -s data/arxiv/arxiv_workloads/res -k 1000 -gk 500
