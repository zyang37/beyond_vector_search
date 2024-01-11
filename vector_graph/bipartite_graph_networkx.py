import networkx as nx
from networkx.algorithms import bipartite
from collections import Counter, defaultdict
from typing import Any


class BipartiteGraphNetworkx:
    def __init__(self):
        self.G = nx.Graph()
        self.data_ids = []

    def add_data_nodes(self, data_id_list):
        self.G.add_nodes_from(data_id_list, bipartite=0)
        self.data_ids += data_id_list

    def add_data_node(self, data_id):
        self.add_data_nodes([data_id])

    def add_keyword_nodes(self, keyword_id_list):
        self.G.add_nodes_from(keyword_id_list, bipartite=1)

    def add_keyword_node(self, keyword_id):
        self.add_keyword_nodes(keyword_id)

    def add_edges(self, data_id_list, keyword_id_list):
        edge_list = list(zip(data_id_list, keyword_id_list))
        print(edge_list)
        self.G.add_edges_from(edge_list)

    def get_data_edges(self, data_id):
        nodes = [x[1] for x in self.G.edges(data_id)]
        return set(nodes)

    def get_keyword_edges(self, keyword_id):
        nodes = [x[1] for x in self.G.edges(keyword_id)]
        return set(nodes)

    def is_bipartite(self):
        return bipartite.is_bipartite(self.G)

    def draw_graph(self):
        nx.draw_networkx(
            self.G,
            pos=nx.drawing.layout.bipartite_layout(self.G, self.data_ids),
            width=2,
        )

    def add_raw_edges(self, edge_list):
        self.G.add_edges_from(edge_list)

    def find_relevant(
        self, input_ids_list: list[str], k: int, method: str = ""
    ) -> list[str]:
        if method == "":
            return self.find_relevant_default(input_ids_list, k)
        elif method == "debug":
            return self.find_relevant_default(input_ids_list, k, True)
        else:
            return []

    def find_relevant_default(
        self, input_ids_list: list[str], k: int, debug: bool = False
    ) -> list[Any]:
        res = []
        ids_to_explore = input_ids_list
        explored_keyword_ids = set()
        round_num = 0
        debug_node_map = {}

        while len(res) < k and len(ids_to_explore) != 0:
            # Get the next set of keywords to explore
            cur_relevant_data_ids = []
            keywords_to_explore = set()
            for cur_id in ids_to_explore:
                for keyword_id in self.G[cur_id]:
                    if keyword_id not in explored_keyword_ids:
                        keywords_to_explore.add(keyword_id)

                        if debug:
                            debug_node_map[keyword_id] = cur_id

            # Go through all of the keywords to find the resultant data ids
            for keyword_id in keywords_to_explore:
                explored_keyword_ids.add(keyword_id)
                cur_relevant_data_ids += self.G[keyword_id]

                if debug:
                    for data_id in self.G[keyword_id]:
                        if data_id not in input_ids_list and data_id not in res:
                            debug_node_map[data_id] = keyword_id

            # Sort this round of ids by the number of keywords explored in this round they are associated with
            c = Counter(cur_relevant_data_ids)
            common = c.most_common(k + len(input_ids_list))

            # Filter redundant ids
            ids_to_explore = []
            for x in common:
                if x[0] not in input_ids_list and x[0] not in res:
                    ids_to_explore.append(x[0])

            res += ids_to_explore
            round_num += 1
        if debug:
            self.log(f"num search rounds: {round_num}")
        res = res[:k]

        if debug:
            for id in res:
                # Build the id chain
                id_add_chain = [id]
                cur_id_in_chain = id
                while cur_id_in_chain in debug_node_map.keys():
                    cur_id_in_chain = debug_node_map[cur_id_in_chain]
                    id_add_chain.append(cur_id_in_chain)

                tmp_c = Counter()
                id_keywords = set(self.data_dict[id])
                for input_id in input_ids_list:
                    shared_keywords = id_keywords.intersection(
                        set(self.data_dict[input_id])
                    )
                    tmp_c.update(shared_keywords)

                self.log(
                    f"{id}, keywords shared with input:{tmp_c.most_common()} discovery chain: {id_add_chain}"
                )
        return res
