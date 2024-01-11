from bipartite_graph_dict import BipartiteGraphDict
from bipartite_graph_networkx import BipartiteGraphNetworkx

import pickle
import sys

sys.path.append("../")
from utils.build_graph import build_graph


def test():
    print("Building graph using custom method...")
    graph_pickle = "../data/graph.pickle"
    with open(graph_pickle, "rb") as f:
        bgd = pickle.load(f)

    bgd_data_dict = bgd.data_dict
    data_ids = list(bgd_data_dict.keys())
    bgd_data_retrieved = bgd.get_data_edges(data_ids[0])

    print("Building graph using networkx...")
    filtered_data_path = "../data/filtered_data.pickle"
    with open(filtered_data_path, "rb") as f:
        filtered_data = pickle.load(f)
    bgn = BipartiteGraphNetworkx()
    bgn = build_graph(filtered_data, bgn)

    bgn_data_retrieved = bgn.get_data_edges(data_ids[0])
    for data in bgn_data_retrieved:
        assert data in bgd_data_retrieved
    print("Passed assertion")

    # test find_relevant
    graph_k = 500
    relevant_bgd = bgd.find_relevant([data_ids[0]], graph_k)
    print(relevant_bgd)
    relevant_bgn = bgn.find_relevant([data_ids[0]], graph_k)
    print(relevant_bgn)
    cnt = 0
    for id in relevant_bgd:
        if id not in relevant_bgn:
            cnt += 1
    print("Total ids not in networkx result:", cnt)


if __name__ == "__main__":
    test()
