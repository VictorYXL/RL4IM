import matplotlib.pyplot as plt
import networkx as nx

import os

def load_graph(node_file, edge_file):
    graph = nx.DiGraph()
    for line in open(edge_file).readlines()[1:]:
        data = line.strip().split(',')
        graph.add_edge(int(data[0]), int(data[1]), weight=float(data[2]))
    for line in open(node_file).readlines()[1:]:
        data = line.strip().split(',')
        graph.add_node(int(data[0]), threshold=float(data[1]))
    return graph

def draw(graph, influenced=[], new_incluenced=[], action_edges=[], seed=0, output_file=None):
    node_labels = {u: str(u) + ":" + str(v) for u, v in nx.get_node_attributes(graph, 'threshold').items()}
    edge_labels = nx.get_edge_attributes(graph, "weight")
    pos = nx.spring_layout(graph, seed=seed)

    # nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="gray")
    nx.draw_networkx_nodes(graph, pos, node_size=700, nodelist=influenced, node_color="r")
    nx.draw_networkx_nodes(graph, pos, node_size=700, nodelist=new_incluenced, node_color="g")
    # node labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif", labels=node_labels)
    # edges
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, width=2)
    nx.draw_networkx_edges(graph, pos, edgelist=action_edges, width=2, edge_color="b")
    
    # edge weight labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8)

    ax = plt.gca()
    plt.tight_layout()
    if output_file == None:
        plt.show()
    else:
        plt.savefig(output_file)

if __name__ == "__main__":
    graph = load_graph(os.path.join("data", "graph1", "node.csv"), os.path.join("data", "graph1", "graph.csv"))

    draw(graph)