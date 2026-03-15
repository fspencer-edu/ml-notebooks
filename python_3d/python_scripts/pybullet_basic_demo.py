import pybullet as p
import pybullet_data
import time

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
p.loadURDF("r2d2.urdf", [0, 0, 1])

for _ in range(2400):
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
