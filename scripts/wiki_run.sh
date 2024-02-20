#!/bin/bash

python inference.py -c config/wiki_cfg.json -w data/wiki_movies/wiki_workloads -kw wiki -k 100 -gk 25 -s data/wiki_movies/wiki_workloads/res
