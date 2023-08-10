import numpy as np
import random

class Simulator:
    def __init__(self, adjacency_matrix_file, max_step=10):
        self.num_nodes, self.adjacency_matrix = self.load_adjacency_matrix(adjacency_matrix_file)
        self.max_step = max_step
        
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
        self.state = np.zeros((self.num_nodes, 1))
        self.current_step = 0
        return self.state


    def step(self, action):
        # Take a step in the environment given an action
        # action: Node to activate

        next_state = self.simulate(action)
        reward = sum(next_state) - sum(self.state)

        self.state = next_state

        done = self.current_step >= self.max_step
        self.current_step += 1

        return next_state, reward, done

    def simulate(self, action):
        next_state = self.state.copy()
        if self.state[action] == 0:
            next_state[action][0] = 1  # Activate the selected node

            # Perform influence propagation using the IC model
            active_nodes = [action]
            while active_nodes:
                current_node = active_nodes.pop(0)
                for neighbor in range(len(self.adjacency_matrix[current_node])):
                    if (
                        self.state[neighbor][0] == 0
                        and random.random() < self.adjacency_matrix[current_node][neighbor]
                    ):
                        next_state[neighbor][0] = 1
                        active_nodes.append(neighbor)

        return next_state