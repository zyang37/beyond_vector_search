
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class MetadataGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.metadata_nodes = []

    def update_one(self, metadata_set):
        self.add_data_nodes(metadata_set)
        self.connect_edges(metadata_set)

    def add_data_nodes(self, metadata_set):
        metadata_list = list(metadata_set)
        # color hex code random generator
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        # construct a list of tuples, each tuple is a node and its attributes: (node, {attr1: val1, attr2: val2, ...})
        metadata_nodes_with_attrs = []
        for md in metadata_list:
            metadata_nodes_with_attrs.append((md, {"color": random_color}))

        self.G.add_nodes_from(metadata_nodes_with_attrs)
        self.metadata_nodes.extend(metadata_list)

    def connect_edges(self, metadata_set):
        metadata_list = list(metadata_set)
        edge_list = get_pair_combinations(metadata_list)
        self.G.add_edges_from(edge_list)

    def get_adjacency_matrix(self):
        return nx.to_numpy_matrix(self.G)
    
    def get_laplacian_matrix(self):
        return nx.laplacian_matrix(self.G)
    
    def draw(self, with_labels=False, node_size=15):
        colors = [self.G.nodes[n]['color'] for n in self.G.nodes]
        nx.draw(self.G, with_labels=with_labels, node_color=colors, node_size=node_size, pos=nx.spring_layout(self.G), alpha=0.85)

# a func take in a list and return all combinations of 2 elements, do not use list comprehension
def get_pair_combinations(l):
    result = []
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            result.append((l[i], l[j]))
    return result

def get_metadata_matrix(graph_obj):
    '''
    Input: graph_obj (custom graph object)
    Output: metadata matrix (numpy matrix)

    Use class var to get metadata matrix from graph object
        - graph_obj.data_dict: data_id: [metadata1, metadata2, ...]
        - graph_obj.keyword_dict: metadata1: [data_id1, data_id2, ...]

    return metadata matrix: MxN matrix where M is the number of metadata and N is the number of data. 
    1 if data_id has metadata, 0 otherwise
    '''
    metadata_matrix = []
    for metadata in graph_obj.keyword_dict:
        metadata_matrix.append([1 if data_id in graph_obj.keyword_dict[metadata] else 0 for data_id in graph_obj.data_dict])

    return np.array(metadata_matrix)

def get_data_embed(graph_obj, emb_func):
    '''
    Input: graph_obj (custom data object), emb_func (function, embed model)
    Output: data_embed (numpy matrix)

    Use emb_func to get data_embed from data_obj
    return data_embed: NxD tensor where N is the number of data and D is the dimension of the embedding
    '''
    data_embed = []
    pass

def get_centroid_embed(metadata_matrix, data_embed):
    pass

def get_transformed_data(data_embed, centroid_embed):
    pass
