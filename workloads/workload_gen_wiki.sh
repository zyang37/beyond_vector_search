#!/bin/bash
cd ../utils

# python wiki_movie.py -nn 50 -n 20 -s ../data/arxiv/arxiv_workloads/pn50_n20.csv
# python wiki_movie.py -nn 100 -n 10 -s ../data/arxiv/arxiv_workloads/pn100_n10.csv
# python wiki_movie.py -nn 200 -n 5 -s ../data/arxiv/arxiv_workloads/pn200_n5.csv
python wiki_movie.py -nn 1000 -n 1 -s ../data/wiki_movies/wiki_workloads/nn1000_n1.csv
