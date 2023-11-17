import networkx as nx
from networkx.algorithms import bipartite


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
        return

    def get_keyword_edges(self, keyword_id):
        return

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

    def find_relevant(self, input_ids_list: list[str], method: str = None) -> list[str]:
        return
