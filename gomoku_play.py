import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from PIL import Image, ImageDraw
import copy
from gomoku import Game, cnn, best_game, tf_model_v

size = 9
model = tf_model_v()

g = Game()
for i in range(size*size):
    if i % 2 is 0:
        g.input_put()
    else:
        g = best_game(g, model)
        g.aout()
        if g.end_game() != 0:
            break
