'''
This scipt is used to generate the base config file for the making vectorDB, inference, and compute metrics
'''

import sys
import json

sys.path.append("../")
from utils.parse_arxiv import load_json, save_json

def base_arxiv():
    cfg = {}
    # infor can be obtained from running "stat <file name>"
    # 
    cfg['data'] = {
        'path': 'data/arxiv/filtered_data.pickle',
        'size': '22440040', 
        'modify': '2023-12-25 15:43:42'
    }
    # available models: https://docs.trychroma.com/embeddings
    # vectorDB should contain all information needed for both the testing and GT DBs. 
    cfg['vectorDB'] = {
        'root': 'data/chroma_dbs/',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk': False,
        'collection_name': 'arxiv_title',
        'id_field': 'id',
        'embed_field': 'title',
        'metadata_fields': ['id', 'title', 'abstract', 'authors', 'journal-ref', 'categories', 'comments', 'update_date']
    }
    # metadata in GT should have id that maps chunk data to the original data id used in the vectorDB
    cfg['vectorDBGT'] = {
        'root': 'data/chroma_dbs/',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk': False,
        'collection_name': 'arxiv_abstract',
        'id_field': 'id',
        'embed_field': 'abstract',
        'metadata_fields': ['id', 'title', 'abstract', 'authors', 'journal-ref', 'categories', 'comments', 'update_date'], 
    }
    # currently, the graphDB is hard coded to handle arxiv, cnn and wiki
    cfg['graphDB'] = {
        'path': 'data/arxiv/graph.pickle', 
        'size': '', 
        'modify': '', 
        'dataset_name': 'arxiv',
        'keyword_fields': ["this is unused"], 
        'data_modify': '',
    }

    # dump the config file 
    # with open('cfg_base.json', 'w', encoding='utf-8') as f:
    #     json.dump(cfg, f, ensure_ascii=False, indent=4)
    save_json(cfg, 'cfg_base.json', verbose=True)

def base_cnn():
    cfg = {}
    # infor can be obtained from running "stat <file name>"
    cfg['data'] = {
        'path': 'data/cnn_news/filtered_dataCNN.pickle',
        'size': '180363105', 
        'modify': '2023-12-28 16:52:13'
    }
    # available models: https://docs.trychroma.com/embeddings
    cfg['vectorDB'] = {
        'root': 'data/chroma_dbs/',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk': False,
        'collection_name': 'cnn_headline',
        'id_field': 'Url',
        'embed_field': 'Headline',
        'metadata_fields': ['Index', 'Author', 'Date published', 'Category', 'Section', 'Url', 'Headline', 'Description', 'Keywords', 'Second headline', 'Article text']
    }
    # metadata in GT should have id that maps chunk data to the original data id used in the vectorDB
    cfg['vectorDBGT'] = {
        'root': 'data/chroma_dbs/',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk': True,
        'collection_name': 'cnn_article',
        'id_field': 'Url',
        'embed_field': 'Article text',
        'metadata_fields': ['Index', 'Author', 'Date published', 'Category', 'Section', 'Url', 'Headline', 'Description', 'Keywords', 'Second headline', 'Article text'],
    }
    # currently, the graphDB is hard coded to handle arxiv, cnn and wiki
    cfg['graphDB'] = {
        'path': 'data/cnn_news/graph.pickle', 
        'size': '', 
        'modify': '', 
        'dataset_name': 'cnn',
        'keyword_fields': ["this is unused"], 
        'data_modify': '',
    }

    # dump the config file 
    # with open('cfg_base.json', 'w', encoding='utf-8') as f:
    #     json.dump(cfg, f, ensure_ascii=False, indent=4)
    save_json(cfg, 'cnn_cfg_base.json', verbose=True)

if __name__ == '__main__':
    # base_arxiv()
    base_cnn()
    