#!/bin/bash

cd zy_testing/

echo "k=5, gk=3"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 5 --graph-k 3 -w ../tmp/w -s ../tmp/tmp_out

echo "k=10, gk=3"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 10 --graph-k 3 -w ../tmp/w -s ../tmp/tmp_out

echo "k=10, gk=7"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 10 --graph-k 7 -w ../tmp/w -s ../tmp/tmp_out

echo "k=50, gk=15"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 50 --graph-k 15 -w ../tmp/w -s ../tmp/tmp_out

echo "k=50, gk=35"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 50 --graph-k 35 -w ../tmp/w -s ../tmp/tmp_out

echo "k=100, gk=30"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 100 --graph-k 30 -w ../tmp/w -s ../tmp/tmp_out

echo "k=100, gk=70"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 100 --graph-k 70 -w ../tmp/w -s ../tmp/tmp_out

echo "k=500, gk=150"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 500 --graph-k 150 -w ../tmp/w -s ../tmp/tmp_out

echo "k=500, gk=350"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 500 --graph-k 350 -w ../tmp/w -s ../tmp/tmp_out

echo "k=1000, gk=300"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 1000 --graph-k 300 -w ../tmp/w -s ../tmp/tmp_out

echo "k=1000, gk=700"
python inference.py -a abs_arxiv_vector -t arxiv_vector -k 1000 --graph-k 700 -w ../tmp/w -s ../tmp/tmp_out
