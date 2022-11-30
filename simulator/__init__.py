from simulator.env.influence_env import InfluenceEnv
from simulator.wrappers.default_wrappers import DefaultWrapper
from utility.graph_tools import load_graph
import os

all = ["make_env"]

def make_env(task_name, output_dir):
    tasks = {
        "demo": ["graph1", [0], 3, "DefaultWrapper"]
    }
    assert(task_name in tasks)
    graph_name      = tasks[task_name][0]
    init_seeds      = tasks[task_name][1]
    max_step        = tasks[task_name][2]
    wrapper_name    = tasks[task_name][3]

    graph_dir = os.path.join("data", graph_name)
    graph = load_graph(os.path.join(graph_dir, "node.csv"), os.path.join(graph_dir, "graph.csv"))
    env = InfluenceEnv(graph, init_seeds, max_step, output_dir)
    if wrapper_name == "DefaultWrapper":
        env = DefaultWrapper(env)
    else:
        raise NotImplementedError
    return env