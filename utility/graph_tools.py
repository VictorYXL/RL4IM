import matplotlib.pyplot as plt
import networkx as nx

import os

def load_graph(node_file, edge_file):
    graph = nx.read_weighted_edgelist(edge_file, delimiter=",", )
    for line in open(node_file).readlines()[1:]:
        data = line.strip().split(',')
        graph.add_node(data[0], threshold=float(data[1]))
    return graph

def draw(graph, seed_list):
    node_labels = {u: u + ":" + str(v) for u, v in nx.get_node_attributes(graph, 'threshold').items()}
    edge_labels = nx.get_edge_attributes(graph, "weight")
    pos = nx.spring_layout(graph, seed=7)

    # nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="b")
    nx.draw_networkx_nodes(graph, pos, node_size=700, nodelist=seed_list, node_color="r")
    # edges
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, width=2)
    # node labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif", labels=node_labels)
    # edge weight labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)

    ax = plt.gca()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graph = load_graph(os.path.join("data", "graph1", "node.csv"), os.path.join("data", "graph1", "graph.csv"))

    draw(graph, ['0', '1'])