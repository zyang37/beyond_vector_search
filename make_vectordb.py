"""
Assume you have ran the notebook `notebook`

This script is used to make a presistent vector databases:
    - ground truth vector database
    - subset of the arxiv dataset
    - subset of the arxiv dataset (hybrid)

args: 
    - path to the arxiv dataset
    - vectorize feilds: title or abstract
    - Save path
"""

import os
import pickle
import chromadb
import argparse
import numpy as np
from tqdm.auto import tqdm
from pprint import pprint
from datetime import datetime
from chromadb.utils import embedding_functions

from utils.build_graph import build_graph
from utils.cnn_news import CnnNewsParser
from utils.wiki_movie import WikiMoviesParser
from utils.parse_arxiv import load_json, save_json, get_metadta_str
from utils.nlp_tools import token_limited_sentences

# Inline debugging
# import code
# code.interact(local=locals())

def aggregate_embeddings_from_chunks(long_text_list, embedding_model, agg_func=np.mean):
    '''
    This function takes a list of long text. 
    For each text, splits it into chunks, and then get chunk embeds, then aggregate the embeddings of the chunks => embed for long text
    
    return: a list of embeddings, size = len(long_text_list)
    '''
    agg_embeddings = []
    for text in tqdm(long_text_list):
        text_chunks = token_limited_sentences(text)
        text_embeds = np.array(embedding_model(text_chunks))
        agg_embeddings.append(list(agg_func(text_embeds, axis=0)))
    return agg_embeddings

def get_embedding_model(vectordb_cfg):
    '''
    this function parses args and return the embedding model from the vectorDB cfg
    '''
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=vectordb_cfg['embedding_model'])
        return ef
    except:
        print("Invalid embedding model: {}".format(vectordb_cfg['embedding_model']))
        print("Using default model: all-MiniLM-L6-v2")
    return embedding_functions.DefaultEmbeddingFunction()

def create_vector_database_chunk(data, db_cfg_dict):
    '''
    This should create the vector database for long texts data
    '''
    db_root = db_cfg_dict['root']
    coll_name = db_cfg_dict['collection_name']
    id_col = db_cfg_dict['id_field']
    embed_col = db_cfg_dict['embed_field']
    metadata_cols = db_cfg_dict['metadata_fields']

    data.drop_duplicates(subset=id_col, inplace=True)
    data.drop_duplicates(subset=embed_col, inplace=True)
    data.reset_index(inplace=True)

    ids = list(data[id_col].astype("str").values)
    data = data[metadata_cols]

    documents = list(data[embed_col].values)
    metedata = list(data.to_dict(orient="records"))
    embed_func = get_embedding_model(db_cfg_dict)

    chroma_client = chromadb.PersistentClient(path=db_root)
    # load data into the database
    batch_size = 500
    collection = chroma_client.create_collection(name=coll_name, embedding_function=embed_func)
    for i in tqdm(range(0, len(ids), batch_size)):
        # this line split the long text into chunks, and then aggregate the embeddings of the chunks
        batch_embeddings = aggregate_embeddings_from_chunks(documents[i : i + batch_size], embed_func)

        collection.add(
            embeddings=batch_embeddings,
            documents=documents[i : i + batch_size],
            metadatas=metedata[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
    print("Saved <{}> at {}".format(coll_name, db_root))

def create_vector_database(data, db_cfg_dict):
    
    db_root = db_cfg_dict['root']
    coll_name = db_cfg_dict['collection_name']
    id_col = db_cfg_dict['id_field']
    embed_col = db_cfg_dict['embed_field']
    metadata_cols = db_cfg_dict['metadata_fields']

    data.drop_duplicates(subset=id_col, inplace=True)
    data.drop_duplicates(subset=embed_col, inplace=True)
    data.reset_index(inplace=True)

    ids = list(data[id_col].astype("str").values)
    data = data[metadata_cols]

    documents = list(data[embed_col].values)
    if "metadata_append" in db_cfg_dict:
        print("append metadata")
        for index, row in data.iterrows():
            # Note: index is the index of the dataframe, not the index of the document in the vector database
            # print(index)
            documents[index] = documents[index] + " " + get_metadta_str(row, db_cfg_dict["metadata_append"])
            # print(documents[index])
            # break
            # exit()

    metedata = list(data.to_dict(orient="records"))
    embed_func = get_embedding_model(db_cfg_dict)

    chroma_client = chromadb.PersistentClient(path=db_root)
    # load data into the database
    batch_size = 500
    collection = chroma_client.create_collection(name=coll_name, embedding_function=embed_func)
    for i in tqdm(range(0, len(ids), batch_size)):
        collection.add(
            # embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]], # could add embed if they are already computed!
            documents=documents[i : i + batch_size],
            metadatas=metedata[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
    print("Saved <{}> at {}".format(coll_name, db_root))

def create_graph_database(data, db_cfg_dict):
    '''
    This should create the graph pickle file, and update the cfg file: "size and modify" fields
    '''
    graph_path = db_cfg_dict['path']

    print("Building graph...")
    if db_cfg_dict['dataset_name'].lower() == "arxiv":
        print("for arxiv dataset")
        graph = build_graph(data)
    elif db_cfg_dict['dataset_name'].lower() == "cnn":
        print("for cnn dataset")
        parser = CnnNewsParser(df=data)
        parser.build_graph()
        graph = parser.G
    elif db_cfg_dict['dataset_name'].lower() == "wiki":
        print("for wiki dataset")
        parser = WikiMoviesParser(df=data)
        parser.build_graph()
        graph = parser.G
    else:
        raise ValueError("Invalid dataset name: {}".format(db_cfg_dict['dataset_name']))
    
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

    graph = pickle.load(open(graph_path, "rb"))
    print(
        f"Graph has {len(graph.get_data_ids_sorted_by_num_edges())} data points attached to {len(graph.get_keyword_ids_sorted_by_num_edges())} keywords"
    )

    print("Saved graph at {}".format(graph_path))

    # Update the cfg file: "size and modify" fields
    file_stats = os.stat(graph_path)
    # Access the last modified timestamp
    last_modified = file_stats.st_mtime
    last_modified_str = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")
    # Access the size of the file in bytes
    file_size = file_stats.st_size

    print("Graph size: {} bytes".format(file_size))
    print("Graph last modified: {}".format(last_modified_str))

    db_cfg_dict['size'] = file_size
    db_cfg_dict['modify'] = last_modified_str
    return db_cfg_dict


# def create_vector_database(data, args, ids):
#     chroma_client = chromadb.PersistentClient(path=args.save)

#     # ids = list(data["id"].astype("str").values)
#     documents = list(data[args.emb].values)
#     metedata = list(data.to_dict(orient="records"))

#     # load data into the database
#     batch_size = 500
#     collection = chroma_client.create_collection(name=args.collection)
#     for i in tqdm(range(0, len(ids), batch_size)):
#         collection.add(
#             # embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]], # could add embed if they are already computed!
#             documents=documents[i : i + batch_size],
#             metadatas=metedata[i : i + batch_size],
#             ids=ids[i : i + batch_size],
#         )
#     print("Saved <{}> at {}".format(args.collection, args.save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vector-graph database, and the ground truth vector database")
    parser.add_argument("-c", "--cfg", metavar="", type=str, default=None, help="path to the config file")
    parser.add_argument("-r", "--root", metavar="", type=str, default="data/chroma_dbs/", help="path to the chroma root")
    # parser.add_argument("-d", "--dataset", metavar="", type=str, default="data/filtered_data.pickle", help="Dataset to use (pickle file))")
    # parser.add_argument("-c", "--collection", metavar="", type=str, default=None, help="Name of the collection")
    # parser.add_argument("--emb", metavar="", type=str, default="title", help="Field to vectorize (title or abstract)")
    # parser.add_argument("-t", "--type", metavar="", type=str, default="v", help="Type of vector database (arxiv, cnn, wiki)")
    # parser.add_argument("-s", "--save", metavar="", type=str, default="data/chroma_dbs/", help="Path to save the vector database")
    parser.add_argument("-bg", "--build_graph_only", action="store_true", help="Build the graph database only")
    parser.add_argument("-l", "--list", action="store_true", help="DO Nothing and List all the vector databases")
    args = parser.parse_args()
    pprint(args.__dict__)

    chromadb_root = args.root

    if args.list:
        chroma_client = chromadb.PersistentClient(path=chromadb_root)
        dbs = chroma_client.list_collections()
        print()
        print("Vector databases under < {} >".format(chromadb_root))
        for i, db in enumerate(dbs):
            print("{:3d}. {}".format(i + 1, db))
        print()
        exit()

    # load the config file
    cfg = load_json(args.cfg)
    pprint(cfg)
    chromadb_root = cfg['vectorDB']['root']

    # Load the dataset
    file = open(cfg['data']['path'], "rb")
    data = pickle.load(file)
    file.close()
    data.fillna("",inplace=True)

    if not args.build_graph_only:
        # create the text vector database
        print("Creating the test vector database...")
        create_vector_database(data, cfg['vectorDB'])

        # create the ground truth vector database
        print("Creating the GT vector database...")
        if 'chunk' in cfg['vectorDBGT'] and cfg['vectorDBGT']['chunk']['use']:
            print("Creating the GT vector database with chunk...")
            create_vector_database_chunk(data, cfg['vectorDBGT'])
        else:
            create_vector_database(data, cfg['vectorDBGT'])

    # create the graph database
    updated_graph_cfg = create_graph_database(data, cfg['graphDB'])
    cfg['graphDB'] = updated_graph_cfg
    cfg['graphDB']['data_modify'] = cfg['data']['modify']

    # save the updated config file
    save_json(cfg, args.cfg, verbose=True)
    print("Saved updated config file at {}".format(args.cfg))

    # old
    # data preprocessing
    # if args.type == "arxiv":
    #     data.fillna("",inplace=True)
    #     data.drop_duplicates(subset='id', inplace=True)
    #     data.drop_duplicates(subset='title', inplace=True)
    #     # for each abstract, append the title text
    #     # data['abstract'] = data['title'] + " " + data['abstract']
    #     # print(data['abstract'].head())
    #     ids = list(data["id"].astype("str").values)
    #     data = data[['id', 'title', 'abstract', 'authors', 'journal-ref', 
    #             'categories', 'comments', 'update_date']]
    # elif args.type == "cnn":
    #     data.drop_duplicates(subset='Url', inplace=True)
    #     ids = list(data["Url"].astype("str").values)
    #     data = data[['Url', 'Headline', 'Author', 'Date published', 'Category', 
    #             'Section', 'Description', 'Keywords', 'Article text']]
    # elif args.type == "wiki":
    #     # for c in data.columns:
    #     #     print(c)
    #     # exit()
    #     ids = list(data["Title"].astype("str").values)
    #     data.drop_duplicates(subset='Title', inplace=True)
    #     data = data[["Title", "Origin", "Director", "Cast", "Genre", "Year", "Plot Summary", "Plot", "Wiki Page"]]
    # else:
    #     raise ValueError("Invalid type: {}".format(args.type))

    # # Create the vector database
    # create_vector_database(data, args, ids)
