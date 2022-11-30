import gym

class DefaultWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = env
    
    '''
    actions = {
        node1: target_node,
        node2: target_node,
        ...
    }

    states = {
        influenced_node1: {uninfluenced_nbr1: {weight, threshold}, uninfluenced_nbr2: {}, ...}
        influenced_node2: {uninfluenced_nbr1: {weight, threshold}, uninfluenced_nbr2: {}, ...}
    }
    '''
    def step(self, actions: dict) -> tuple[dict, list, bool, dict]:
        states, rewards, done, infos = self.env.step(actions)
        states = self.add_weight_threshold_info(states)
        return states, rewards, done, infos

    def reset(self) -> None:
        states = self.env.reset()
        states = self.add_weight_threshold_info(states)
        return states
    
    def add_weight_threshold_info(self, states):
        for node in states:
            for nbr in states[node]:
                weight = self.env.graph[node][nbr]['weight']
                nbr_threshold = self.env.graph.nodes[nbr]['threshold']
                states[node][nbr] = [weight, nbr_threshold]
        return states

