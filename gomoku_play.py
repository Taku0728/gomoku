import tensorflow as tf
import numpy as np
import random as rd
import copy
from gomoku import *

size = 9
model = tf_model_v()

g = Game()
for i in range(size*size):
    if i % 2 is 0:
        g.input_put()
    else:

        g.aout()
        if g.end_game() != 0:
            break
