'''
This file contains some utility functions for vector search that are used for experiments and analysis.
'''

from tqdm.auto import tqdm

def vector_search(coll, text_queries, k, batch_size=100):
    collection = coll
    search_results = []
    # Split the queries into batches and search for each batch
    for idx in tqdm(range(0, len(text_queries), batch_size)):
        batch_queries = text_queries[idx : idx + batch_size]
        results = collection.query(query_texts=batch_queries, n_results=k)
        search_results.extend(results["ids"])
    return search_results

