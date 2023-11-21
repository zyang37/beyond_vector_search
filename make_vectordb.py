'''
Assume you have ran the notebook `notebook`

This script is used to make a presistent vector databases:
    - ground truth vector database
    - subset of the arxiv dataset
    - subset of the arxiv dataset (hybrid)

args: 
    - path to the arxiv dataset
    - vectorize feilds: title or abstract
    - Save path
'''

import pickle
import chromadb
import argparse
import numpy as np
from tqdm.auto import tqdm
from pprint import pprint


def create_vector_arxiv(data, args):
    chroma_client = chromadb.PersistentClient(path=args.save)

    documents = list(data[args.emb].values)
    ids = list(data['id'].astype('str').values)
    metedata = list(data.to_dict(orient='records'))

    # load data into the database
    batch_size = 500
    collection = chroma_client.create_collection(name=args.collection)
    for i in tqdm(range(0, len(ids), batch_size)):
        collection.add(
            # embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]], # could add embed if they are already computed!
            documents = documents[i:i+batch_size],
            metadatas = metedata[i:i+batch_size],
            ids = ids[i:i+batch_size]
        )
    print("Saved <{}> at {}".format(args.collection, args.save))


def create_hybrid_arxiv(data, args):
    raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a vector database")
    parser.add_argument("-d", "--dataset", metavar="", type=str, default="data/filtered_data.pickle", help="Dataset to use (pickle file))")
    parser.add_argument("-c", "--collection", metavar="", type=str, default="arxiv_vector", help="Name of the collection")
    parser.add_argument("--emb", metavar="", type=str, default="title", help="Field to vectorize (title or abstract)")
    parser.add_argument("-t", "--type", metavar="", type=str, default="v", help="Type of vector database (v or vg)")
    parser.add_argument("-s", "--save", metavar="", type=str, default="data/chroma_dbs/", help="Path to save the vector database")
    parser.add_argument("-l", "--list", action="store_true", help="DO Nothing and List all the vector databases")
    args = parser.parse_args()

    if args.list:
        chroma_client = chromadb.PersistentClient(path=args.save)
        dbs = chroma_client.list_collections()
        print()
        print("Vector databases in {}".format(args.save))
        for i, db in enumerate(dbs):
            print("{:3d}. {}".format(i+1, db))
        print()
        exit()

    # Load the arxiv dataset
    file = open(args.dataset, "rb")
    data = pickle.load(file)
    file.close()

    # data preprocessing
    data.fillna("",inplace=True)
    data.drop_duplicates(subset='id', inplace=True)
    data.drop_duplicates(subset='title', inplace=True)

    # for each abstract, append the title text
    data['abstract'] = data['title'] + " " + data['abstract']
    # print(data['abstract'].head())

    data = data[['id', 'title', 'abstract', 'authors', 'journal-ref', 
             'categories', 'comments', 'update_date']]

    # Create the vector database
    if args.type == "v":
        create_vector_arxiv(data, args)
    elif args.type == "vg":
        create_hybrid_arxiv(data, args)
    else:
        raise ValueError("Invalid type: {}".format(args.type))
