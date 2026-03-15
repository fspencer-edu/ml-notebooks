from ursina import *

app = Ursina()
cube = Entity(model='cube', color=color.azure, scale=2)
ground = Entity(model='plane', scale=10, y=-2, color=color.light_gray)
camera.position = (0, 2, -10)

def update():
    cube.rotation_y += 20 * time.dt

app.run()
