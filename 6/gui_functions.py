import constants as const
import pygame

pygame.mixer.init()

def show_score(window, score):
    scoreDigits = [int(x) for x in list(str(score))]
    total_width = 0

    for digit in scoreDigits:
        total_width += const.NUM_IMAGES['numbers'][digit].get_width()

    x_off_set = (const.WINDOW_WIDTH - total_width) / 2

    for digit in scoreDigits:
        window.blit(const.NUM_IMAGES['numbers'][digit], (x_off_set, const.WINDOW_HEIGHT * 0.1))
        x_off_set += const.NUM_IMAGES['numbers'][digit].get_width()

def show_gen_stat(window, generation):
    scoreDigits = [int(x) for x in list(str(generation))]
    total_width = 0

    for digit in scoreDigits:
        total_width += const.NUM_IMAGES['numbers'][digit].get_width()

    x_off_set = -220 + (const.WINDOW_WIDTH - total_width) / 2

    window.blit(const.LETTER_G, (x_off_set, const.WINDOW_HEIGHT - 100))
    x_off_set += const.LETTER_G.get_width()
    window.blit(const.LETTER_E, (x_off_set, const.WINDOW_HEIGHT - 100))
    x_off_set += const.LETTER_E.get_width()
    window.blit(const.LETTER_N, (x_off_set, const.WINDOW_HEIGHT - 100))
    x_off_set += const.LETTER_N.get_width()
    x_off_set += 30

    for digit in scoreDigits:
        window.blit(const.NUM_IMAGES['numbers'][digit], (x_off_set, const.WINDOW_HEIGHT - 100))
        x_off_set += const.NUM_IMAGES['numbers'][digit].get_width()

def draw_window(window, birds, pipes, base, background, score, gen_count):
    background.draw(window)
    for pipe in pipes:
        pipe.draw(window)

    show_score(window, score)

    base.draw(window)
    show_gen_stat(window, gen_count)
    for bird in birds:
        bird.draw(window)
    pygame.display.update()   

def load_music():
    pygame.mixer.music.load('sounds/background_music.mp3')
    
    pygame.mixer.music.set_volume(0.1)
    pygame.mixer.music.play(1000)