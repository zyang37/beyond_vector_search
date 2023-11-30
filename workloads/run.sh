# python query_gen.py -pn 100 -n 10 -s not_title.csv

python inference.py -k 100 -s not_title_res.csv -c ../data/chroma_dbs -w not_title.csv -a abs_arxiv_vector_at -t arxiv_vector_at

python compute_metrics.py -c not_title_res.csv -gt abs_arxiv_vector_at -pd arxiv_vector_at --proc 2
