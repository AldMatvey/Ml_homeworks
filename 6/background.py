import constants as const

class Background:
    velocity = 1
    width = const.BACKGROUND_IMAGE.get_width()
    image = const.BACKGROUND_IMAGE

    def __init__(self):
        self.y = 0
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.velocity
        self.x2 -= self.velocity

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width

        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, window):
        window.blit(self.image, (self.x1, self.y))
        window.blit(self.image, (self.x2, self.y))