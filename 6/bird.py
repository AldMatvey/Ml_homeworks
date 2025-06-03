import pygame
import random
import constants as const


class Bird:
    max_rotation = 25
    rotation_velocity = 20
    animation_time = 5

    def __init__ (self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.acceleration = 0.3
        self.height = self.y
        self.img_count = 0
        self.images = random.choice(
            [const.BLUE_BIRD_IMAGES,
            const.RED_BIRD_IMAGES,
            const.YELLOW_BIRD_IMAGES]
        )
        self.img = self.images[0]

    def jump (self):
        self.velocity = -8.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        self.velocity += self.acceleration * self.tick_count

        d = self.velocity * self.tick_count

        if d >= 16:
            d = 16

        if d < 0: 
            d -= 2

        self.y += d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation
        else:
            if self.tilt > -90:
                self.tilt-= self.rotation_velocity

    def draw(self, window):
        self.img_count += 1

        if self.img_count < self.animation_time:
            self.img = self.images[0]
        elif self.img_count < self.animation_time * 2:
            self.img = self.images[1]
        elif self.img_count < self.animation_time * 3:
            self.img = self.images[2]
        elif self.img_count < self.animation_time * 4:
            self.img = self.images[1]
        elif self.img_count == self.animation_time * 4 + 1:
            self.img = self.images[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.images[1]
            self.img_count = self.animation_time * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rectangle = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        window.blit(rotated_image, new_rectangle.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)