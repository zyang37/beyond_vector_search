import random
import pickle
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

def get_pair_combinations(l):
    '''
    A func take in a list and return all combinations of 2 elements, do not use list comprehension
    '''
    result = []
    for i in range(len(l)):
        # ring haha
        # try: 
        #     result.append((l[i], l[i+1]))
        # except:
        #     result.append((l[i], l[0]))
        for j in range(i+1, len(l)):
            result.append((l[i], l[j]))
    return result

class MetadataGraph:
    '''
    A class to represent a graph of metadata nodes and their connections.
    '''
    def __init__(self):
        self.G = nx.Graph()
        self.metadata_nodes = []

    def build_from_metadata_set_list(self, metadata_set_list):
        '''
        Build the graph from a list of metadata sets. Each metadata set is a set of metadata strings.
        '''
        for metadata_set in metadata_set_list:
            self.add_data_nodes(metadata_set)
            self.connect_edges(metadata_set)

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
        return nx.adjacency_matrix(self.G).todense()
    
    def get_laplacian_matrix(self, normalized=False):
        if normalized:
            return nx.normalized_laplacian_matrix(self.G).todense()
        
        return nx.laplacian_matrix(self.G).todense()
    
    def draw(self, with_labels=False, node_size=15, alpha=0.85, subplot_ax=None, color_scheme=None):
        if color_scheme==None:
            colors = [self.G.nodes[n]['color'] for n in self.G.nodes]
        else:
            colors = color_scheme
        nx.draw(self.G, with_labels=with_labels, node_color=colors, node_size=node_size, pos=nx.spring_layout(self.G), alpha=alpha, ax=subplot_ax)


def get_cat_from_keyword(keyword):
    return keyword.split(':')[0]

def color_scheme_gen(num):
    '''
    Generate a list of random color hex codes, each code is a string. 
    '''
    color_list = []
    for i in range(num):
        color_list.append("#{:06x}".format(random.randint(0, 0xFFFFFF)))
    return color_list

def plot_color_scheme(color_list, text_list=None):
    '''
    Plot the color scheme, each color is a hex code string. horizontal color bars with text after each bar.

    Args:
        - color_list: list of color hex codes
        - text_list: list of text strings, each string is the name of the color
    '''
    height = 0.9
    width = 0.6

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, c in enumerate(color_list):
        ax.add_patch(plt.Rectangle((0, i), width, height, color=c))
    
    if text_list is not None:
        for i, t in enumerate(text_list):
            ax.text(width*1.02, i+0.5*height, t, ha='left', va='center', fontsize=9, color='black')
        
    ax.set_ylim(0, len(color_list))
    # ax.set_xlim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
 
