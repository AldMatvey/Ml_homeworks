import pygame 
import neat
import os 
from activation_func import activation_func
from gui_functions import load_music

pygame.mixer.init() 

def main(genomes, config):
    activation_func(genomes, config)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, 
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, 
                                neat.DefaultStagnation,
                                config_path
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main, 100)

if __name__ == "__main__":
    
    load_music()
    local_directory = os.path.dirname(__file__)
    config_path = os.path.join(local_directory, "config_file.txt")
    run(config_path)