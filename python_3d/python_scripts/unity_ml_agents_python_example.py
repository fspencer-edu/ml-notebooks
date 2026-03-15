# Python side example for Unity ML-Agents
# Build the scene in Unity, then connect from Python for training or inference.

from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(file_name=None)  # replace with built environment path when needed
env.reset()

behavior_names = list(env.behavior_specs.keys())
print("Behaviors:", behavior_names)

# Typical CLI training:
# mlagents-learn config/ppo/3DBall.yaml --run-id=my_run
