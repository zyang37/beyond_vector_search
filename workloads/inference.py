import pandas as pd
import chromadb
import argparse
import pickle
from tqdm import tqdm


def create_paper_id_to_title_dict(filtered_data):
    return dict(zip(filtered_data.id.astype("string"), filtered_data.title))

def create_id_to_abstract_dict(filtered_data):
    return dict(zip(filtered_data.id.astype("string"), filtered_data.abstract))


def get_query_col(df, id2abstract_dict):
    return df["query"]


def get_abstract_col(df, id2abstract_dict):
    return df["paper_id"].astype("string").map(id2abstract_dict)


def vector_search(df, coll_name, client, k, get_query_func, id2abstract_dict, batch_size=50):
    collection = client.get_collection(name=coll_name)
    query_col = get_query_func(df, id2abstract_dict)
    search_results = []
    for idx in tqdm(range(0, df.shape[0], batch_size)):
        queries = query_col.iloc[idx : idx + batch_size].values.tolist()
        results = collection.query(query_texts=queries, n_results=k)
        search_results.extend(results["ids"])
    return search_results


def infer(chroma_path, workload_csv, title_col, abstract_col, id2abstract_dict, k):
    df = pd.read_csv(workload_csv)
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    vector_search_results = vector_search(
        df, title_col, chroma_client, k, get_query_col, id2abstract_dict
    )
    ground_truths = vector_search(df, abstract_col, chroma_client, k, get_abstract_col, id2abstract_dict)
    df[title_col] = pd.Series(vector_search_results)
    df[abstract_col] = pd.Series(ground_truths)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="number of k to retrieve for each query",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default="../data/inference_results.csv",
        help="csv path to save the results",
    )
    parser.add_argument(
        "-c",
        "--chroma",
        type=str,
        default="../data/chroma_dbs/",
        help="path to load chroma db collections",
    )
    parser.add_argument(
        "-w",
        "--workload-csv",
        type=str,
        default="workload.csv",
        help="workload csv file",
    )
    parser.add_argument(
        "-a",
        "--abstracts",
        type=str,
        default="abstracts",
        help="ChromaDB collection name that stores the embeddings of abstracts",
    )
    parser.add_argument(
        "-t",
        "--titles",
        type=str,
        default="arxiv_vector",
        help="ChromaDB collection name that stores the embeddings of paper titles",
    )
    parser.add_argument(
        "-f",
        "--filtered-data-path",
        type=str,
        default="../data/filtered_data.pickle",
        help="path to filtered_data pickle file",
    )

    args = parser.parse_args()
    with open(args.filtered_data_path, "rb") as f:
        filtered_data = pickle.load(f)
    id2title_dict = create_paper_id_to_title_dict(filtered_data)
    id2abstract_dict = create_id_to_abstract_dict(filtered_data)
    result_df = infer(
        args.chroma,
        args.workload_csv,
        args.titles,
        args.abstracts,
        id2abstract_dict,
        args.k,
    )
    print("Papers found by titles:")
    for x in result_df[args.titles]:
        x = [id2title_dict[y] for y in x]
        print(x)
    print()
    print("Papers found by abstracts:")
    for x in result_df[args.abstracts]:
        x = [id2title_dict[y] for y in x]
        print(x)
    # print(result_df[args.titles].map(id2title_dict))
    # print(result_df[args.abstracts].map(id2abstract_dict))
    result_df.to_csv(args.save)
