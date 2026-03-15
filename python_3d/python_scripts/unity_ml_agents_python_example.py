from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import time

env = UnityEnvironment(file_name=None)
env.reset()

behavior_name = list(env.behavior_specs.keys())[0]
print("Behavior:", behavior_name)

for step in range(500):

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    if len(decision_steps) > 0:
        n_agents = len(decision_steps)

        actions = np.random.uniform(-1,1,(n_agents,2)).astype(np.float32)

        action_tuple = ActionTuple(continuous=actions)
        env.set_actions(behavior_name, action_tuple)

    env.step()
    time.sleep(0.05)

env.close()