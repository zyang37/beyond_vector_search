from ast import keyword
from collections import Counter, defaultdict
from typing import Any
import heapq


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
        elif method == "weighted":
            return self.find_relevant_weighted(input_ids_list, k)
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
            for cur_id in ids_to_explore:
                for keyword_id in self.data_dict[cur_id]:
                    if keyword_id not in explored_keyword_ids:
                        keywords_to_explore.add(keyword_id)

                        if debug:
                            debug_node_map[keyword_id] = cur_id

            # Go through all of the keywords to find the resultant data ids
            for keyword_id in keywords_to_explore:
                explored_keyword_ids.add(keyword_id)
                cur_relevant_data_ids += self.keyword_dict[keyword_id]

                if debug:
                    for data_id in self.keyword_dict[keyword_id]:
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

    def define_edge_weight_by_keyword_and_hop_penalty(
        self, keyword_map, hop_penalty_per_round
    ):
        self.hop_penalty_per_round = hop_penalty_per_round
        self.keyword_map = keyword_map

    def get_edge_weight(self, keyword_id):
        for keyword, value in self.keyword_map.items():
            if keyword in keyword_id:
                return -1 * value
        exit("Keyword not found in keyword_map")

    def get_edge_weight_by_degree(self, keyword_id):
        return -1 * len(self.keyword_dict[keyword_id])

    def add_edges_to_pq(
        self, ids_to_explore, explored_keyword_ids, edge_pq, hop_penalty
    ):
        for id in ids_to_explore:
            for keyword_id in self.data_dict[id]:
                if keyword_id not in explored_keyword_ids:
                    explored_keyword_ids.add(keyword_id)
                    heapq.heappush(
                        edge_pq,
                        (
                            min(self.get_edge_weight(keyword_id) + hop_penalty, 0),
                            keyword_id,
                        ),
                    )

    def find_relevant_weighted(self, input_ids_list: list[str], k: int) -> list[Any]:
        # Build a Counter object maybe associating each paper with the cost of the edges that discovered it
        # While we have not found k relevant documents and we still have edges to explore
        # Given the current ids_to_explore, construct a priority queue of all edges to explore
        # Explore all of the edges that have the same priority and add the edges to an explored list and add the nodes from these edges to the input_ids_to_explore list as well as the relevant Counter list with the cost of the edge
        # Go through the input ids to explore list and add all of the edges to the priority queue minus some constant as long as they aren't already in the explored list
        # Repeat
        relevant_ids = defaultdict(int)
        edge_pq = []
        explored_keyword_ids = set()
        hop_penalty = 0

        for id in input_ids_list:
            relevant_ids[id] = 0
        # Construct the initial list of edges to explore
        self.add_edges_to_pq(input_ids_list, explored_keyword_ids, edge_pq, hop_penalty)
        hop_penalty += self.hop_penalty_per_round

        # Loop until we find all of the ids or run out of edges
        while len(relevant_ids) < k + len(input_ids_list) and len(edge_pq) != 0:
            # Remove all of the edges with the same priority from the heap
            ids_to_explore = []
            top_priority, top_id = heapq.heappop(edge_pq)
            for data_id in self.keyword_dict[top_id]:
                relevant_ids[data_id] += top_priority
                ids_to_explore.append(data_id)
            while len(edge_pq) != 0 and edge_pq[0][0] == top_priority:
                top_priority, top_id = heapq.heappop(edge_pq)
                for data_id in self.keyword_dict[top_id]:
                    relevant_ids[data_id] += top_priority
                    ids_to_explore.append(data_id)

            # Explore all of the edges that have the same priority
            self.add_edges_to_pq(
                ids_to_explore, explored_keyword_ids, edge_pq, hop_penalty
            )

            hop_penalty += self.hop_penalty_per_round

        # Sort ids by importance and filter out the initial ids
        sorted_ids = sorted(
            relevant_ids.items(), key=lambda item: item[1], reverse=False
        )
        filtered_ids = []
        for x in sorted_ids:
            if x[0] not in input_ids_list:
                filtered_ids.append(x[0])

        return filtered_ids[:k]

    def add_edges_to_pq_ranked(
        self, ids_to_explore, explored_keyword_ids, edge_pq, hop_penalty, relevant_ids
    ):
        for id in ids_to_explore:
            for keyword_id in self.data_dict[id]:
                if keyword_id not in explored_keyword_ids.keys():
                    explored_keyword_ids[keyword_id] = 1
                    heapq.heappush(
                        edge_pq,
                        (
                            min(
                                self.get_edge_weight(keyword_id) + hop_penalty,
                                0,
                            ),
                            keyword_id,
                        ),
                    )
                else:
                    # decrement the priority (make more important) of data that are directly connected to
                    # the explored keywords
                    for data_id in self.keyword_dict[keyword_id]:
                        relevant_ids[data_id] -= explored_keyword_ids[keyword_id]
                    explored_keyword_ids[keyword_id] += 1

    def find_relevant_weighted_ranked(
        self, input_ids_list: list[str], k: int, cut_off: int
    ) -> list[Any]:
        # Build a Counter object maybe associating each paper with the cost of the edges that discovered it
        # While we have not found k relevant documents and we still have edges to explore
        # Given the current ids_to_explore, construct a priority queue of all edges to explore
        # Explore all of the edges that have the same priority and add the edges to an explored list and add the nodes from these edges to the input_ids_to_explore list as well as the relevant Counter list with the cost of the edge
        # Go through the input ids to explore list and add all of the edges to the priority queue minus some constant as long as they aren't already in the explored list
        # Repeat
        relevant_ids = defaultdict(int)
        edge_pq = []
        explored_keyword_ids = {}
        hop_penalty = 0

        for id in input_ids_list:
            relevant_ids[id] = 0
        # Construct the initial list of edges to explore
        self.add_edges_to_pq_ranked(
            input_ids_list, explored_keyword_ids, edge_pq, hop_penalty, relevant_ids
        )
        hop_penalty += self.hop_penalty_per_round

        # Loop until we find all of the ids or run out of edges
        while len(relevant_ids) < k + len(input_ids_list) and len(edge_pq) != 0:
            # Remove all of the edges with the same priority from the heap
            ids_to_explore = []
            top_priority, top_id = heapq.heappop(edge_pq)
            for data_id in self.keyword_dict[top_id]:
                relevant_ids[data_id] += top_priority
                ids_to_explore.append(data_id)
            while len(edge_pq) != 0 and edge_pq[0][0] == top_priority:
                top_priority, top_id = heapq.heappop(edge_pq)
                for data_id in self.keyword_dict[top_id]:
                    relevant_ids[data_id] += top_priority
                    ids_to_explore.append(data_id)

            # Explore all of the edges that have the same priority
            self.add_edges_to_pq_ranked(
                ids_to_explore, explored_keyword_ids, edge_pq, hop_penalty, relevant_ids
            )

            hop_penalty += self.hop_penalty_per_round

        # Sort ids by importance and filter out the initial ids
        sorted_ids = sorted(
            relevant_ids.items(), key=lambda item: item[1], reverse=False
        )
        filtered_ids = []
        for x in sorted_ids:
            # print(x[1])
            if x[0] not in input_ids_list and x[1] <= cut_off:
                filtered_ids.append(x[0])

        return filtered_ids[:k]

    def get_keyword_totals_of_id_list(self, id_list) -> list[Any]:
        keyword_list = []
        for id in id_list:
            keyword_list += self.data_dict[id]
        return Counter(keyword_list).most_common()

    def log(self, statement: str):
        print(statement)
