import numpy as np
import random

class InfluenceMaximizationEnvironment:
    def __init__(self, adjacency_matrix_file, max_step=10):
        self.num_nodes, self.adjacency_matrix = self.load_adjacency_matrix(adjacency_matrix_file)
        self.seed_nodes = []
        self.activated_nodes = []
        self.max_step = max_step
        self.current_step = 0

    def load_adjacency_matrix(self, adjacency_matrix_file):
        with open(adjacency_matrix_file, 'r') as file:
            lines = file.readlines()
            num_nodes = int(lines[0])
            matrix = np.zeros((num_nodes, num_nodes))
            for line in lines[1:]:
                node1, node2, probability = line.strip().split(',')
                node1 = int(node1)
                node2 = int(node2)
                probability = float(probability)
                matrix[node1][node2] = probability
        return num_nodes, matrix

    def reset(self):
        # Reset the environment to its initial state
        self.seed_nodes = []
        self.activated_nodes = []

    def step(self, action):
        # Take a step in the environment given an action
        # action: Node to activate

        new_activated_nodes = self.simulate(action)

        reward = len(new_activated_nodes)
        done = self.current_step >= self.max_step
        self.current_step += 1

        return new_activated_nodes, reward, done

    def simulate(self, action):
        # Simulate the Independent Cascade (IC) model given the chosen action
        # action: Node to activate

        self.seed_nodes.append(action)

        activated_nodes = set(self.activated_nodes)  # Track the nodes that are activated
        activated_nodes.add(action)

        # Perform the Independent Cascade model simulation
        new_nodes = set()  # Track the newly activated nodes in each step
        new_nodes.add(action)
        while new_nodes:
            next_nodes = set()
            for node in new_nodes:
                for neighbor in range(self.num_nodes):
                    if (
                        self.adjacency_matrix[node][neighbor] > 0
                        and neighbor not in activated_nodes
                        and random.random() < self.adjacency_matrix[node][neighbor]
                    ):
                        activated_nodes.add(neighbor)
                        next_nodes.add(neighbor)
            new_nodes = next_nodes

        new_activated_nodes = list(activated_nodes - set(self.activated_nodes))
        self.activated_nodes = list(activated_nodes)
        return new_activated_nodes

if __name__ == "__main__":
    env = InfluenceMaximizationEnvironment("graph.csv")
    print(env.activated_nodes)
    env.step(1)
    print(env.activated_nodes)