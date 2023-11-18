from collections import Counter
from typing import Any


class BipartiteGraphDict:
    def __init__(self):
        self.data_dict = {}
        self.keyword_dict = {}

    def add_data_nodes(self, data_id_list: set[str]):
        for id in data_id_list:
            self.add_data_node(id)

    def add_data_node(self, data_id: str):
        if data_id not in self.data_dict.keys():
            self.data_dict[data_id] = set()
        else:
            self.log(f"Data node {data_id} already in data_dict")

    def add_keyword_nodes(self, keyword_id_list: set[str]):
        for id in keyword_id_list:
            self.add_keyword_node(id)

    def add_keyword_node(self, keyword_id):
        if keyword_id not in self.keyword_dict.keys():
            self.keyword_dict[keyword_id] = set()
        else:
            self.log(f"Keyword node {keyword_id} already in keyword_dict")

    def add_edges(self, data_id_list, keyword_id_list):
        edge_list = list(zip(data_id_list, keyword_id_list))
        self.add_raw_edges(edge_list)

    def add_raw_edges(self, edge_list):
        for edge in edge_list:
            if edge[0] not in self.data_dict.keys():
                assert False, f"Data node {edge[0]} found in data_dict"
            if edge[1] not in self.keyword_dict.keys():
                assert False, f"Keyword node {edge[1]} found in keyword_dict"

            self.data_dict[edge[0]].add(edge[1])
            self.keyword_dict[edge[1]].add(edge[0])

    def get_data_edges(self, data_id):
        return self.data_dict[data_id]

    def get_keyword_edges(self, keyword_id):
        return self.keyword_dict[keyword_id]

    def is_bipartite(self):
        return True

    def draw_graph(self):
        return

    def find_relevant(
        self, input_ids_list: list[str], k: int, method: str = ""
    ) -> list[Any]:
        if method == "":
            return self.find_relevant_default(input_ids_list, k)
        elif method == "debug":
            return self.find_relevant_default(input_ids_list, k, True)
        else:
            return []

    def get_data_ids_sorted_by_num_edges(self):
        return sorted(
            self.data_dict.keys(), key=lambda x: len(self.data_dict[x]), reverse=True
        )

    def get_keyword_ids_sorted_by_num_edges(self):
        return sorted(
            self.keyword_dict.keys(),
            key=lambda x: len(self.keyword_dict[x]),
            reverse=True,
        )

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
            # print("IDS: ", ids_to_explore)
            for cur_id in ids_to_explore:
                # print("CUR ID: ", self.data_dict[cur_id])
                for keyword_id in self.data_dict[cur_id]:
                    if keyword_id not in explored_keyword_ids:
                        keywords_to_explore.add(keyword_id)

                        if debug:
                            debug_node_map[keyword_id] = cur_id

            # Go through all of the keywords to find the resultant data ids
            for keyword_id in keywords_to_explore:
                # # print("KEYWORD ID: ", keyword_id)
                explored_keyword_ids.add(keyword_id)
                cur_relevant_data_ids += self.keyword_dict[keyword_id]

                if debug:
                    for data_id in self.keyword_dict[keyword_id]:
                        if data_id not in input_ids_list and data_id not in res:
                            debug_node_map[data_id] = keyword_id

            # Sort this round of ids by the number of keywords explored in this round they are associated with
            c = Counter(cur_relevant_data_ids)
            common = c.most_common(k + len(input_ids_list))
            # # print("Common", common)

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

    def get_keyword_totals_of_id_list(self, id_list) -> list[Any]:
        keyword_list = []
        for id in id_list:
            keyword_list += self.data_dict[id]
        return Counter(keyword_list).most_common()

    def log(self, statement: str):
        print(statement)
