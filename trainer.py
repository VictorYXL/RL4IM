import torch
import numpy as np
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class RLTrainer:  
    def __init__(self, policy, simulator, replay_buffer, optimizer, criterion, batch_size=32):  
        self.policy = policy 
        self.simulator = simulator  
        self.replay_buffer = replay_buffer  
        self.optimizer = optimizer  
        self.criterion = criterion  
        self.batch_size = batch_size  
  
    def train(self, num_episodes):  
        for episode in range(num_episodes):  
            state = self.simulator.reset()  
            done = False  
            total_reward = 0  
            print(".")
            while not done:  
                action = self.policy.choose_action(torch.tensor(state, dtype=torch.float32)) 
                print(action)
                next_state, reward, done = self.simulator.step(action)  
                total_reward += reward  
  
                # Store the transition in the replay buffer  
                self.replay_buffer.add(Transition(state, action, reward, next_state))  
                state = next_state  
  
                # Train the QNetwork using a mini-batch of experiences  
                if len(self.replay_buffer.buffer) >= self.batch_size:  
                    self.update_policy()  
  
            print(f"Episode {episode + 1}: Total reward = {total_reward}")  
  
    def update_policy(self):  
        # Sample a mini-batch of experiences from the replay buffer  
        transitions = self.replay_buffer.sample(self.batch_size)  
        batch = Transition(*zip(*transitions))  
  
        state_batch = torch.tensor(batch.state, dtype=torch.float32)  
        action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1)  
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).view(-1, 1)  
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)  
  
        # Compute the Q-values of the current states and actions  
        q_values = self.policy.q_network(state_batch, torch.tensor(self.simulator.adjacency_matrix, dtype=torch.float32)) 
        q_values = q_values.gather(1, action_batch.unsqueeze(2))  
  
        # Compute the target Q-values using the next states  
        with torch.no_grad():  
            target_q_values = self.policy.q_network(next_state_batch, torch.tensor(self.simulator.adjacency_matrix, dtype=torch.float32))  
            max_target_q_values, _ = target_q_values.max(dim=1, keepdim=True)  
        target_q_values = reward_batch + max_target_q_values  
  
        # Compute the loss and update the QNetwork's parameters  
        loss = self.criterion(q_values, target_q_values)  
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step()  

