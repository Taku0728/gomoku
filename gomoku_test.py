import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from PIL import Image, ImageDraw
import copy
from gomoku import Game
from gomoku import cnn
from gomoku import best_game
from gomoku import tf_model

size = 8
model = tf_model()

g = Game()
for i in range(size*size):
    g = best_game(g, model)
    g.aout()
    gs = g.game_set(0)
    if gs != 0:
        break
