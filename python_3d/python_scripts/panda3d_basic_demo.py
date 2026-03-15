from direct.showbase.ShowBase import ShowBase

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        model = self.loader.loadModel("models/environment")
        model.reparentTo(self.render)
        model.setScale(0.1)
        model.setPos(-8, 42, 0)

app = MyApp()
app.run()
