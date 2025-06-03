import pygame
import os

pygame.mixer.init()

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

GEN = 0

death_sound = pygame.mixer.Sound('sounds/death_cut.mp3')
win_sound = pygame.mixer.Sound('sounds/win_cut.mp3')

NUM_IMAGES = {}

NUM_IMAGES['numbers'] = (
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "0.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "3.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "4.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "5.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "6.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "7.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "8.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "9.png"))),
)

LETTER_G = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "G.png")))
LETTER_G = pygame.transform.flip(LETTER_G, True, True)
LETTER_E = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "E.png")))
LETTER_N = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "N.png")))

BLUE_BIRD_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "bluebird-downflap.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "bluebird-midflap.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "bluebird-upflap.png")))
]

RED_BIRD_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "redbird-downflap.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "redbird-midflap.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "redbird-upflap.png")))
]

YELLOW_BIRD_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "yellowbird-downflap.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "yellowbird-midflap.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "yellowbird-upflap.png")))
]

PIPE_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "pipe-green.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "pipe-red.png")))
]

BASE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "base.png")))

BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "sprites", "background-day.png")))