import os
import random
from simulator import make_env

def random_actions(task_name, output_dir):
    env = make_env(task_name, output_dir)
    done = False
    states = env.reset()
    env.render()
    while not done:
        actions = {}
        for node in states:
            uninfluenced = list(states[node].keys())
            target = uninfluenced[random.randrange(0, len(uninfluenced))]
            actions[node] = target
        states, rewards, done, infos = env.step(actions)
        env.render()
    return infos

infos = random_actions("demo", os.path.join("output", "demo_random_action"))
print(infos)