import tensorflow as tf
import numpy as np
import random as rd
import copy
from gomoku import *
from search import *
from time import time

size = 9
model_v = tf_model_v()
# model_p = tf_model_p()

w0 = 0
w1 = 0
t = 0
for n in range(100):
    g = Game()
    for i in range(size*size):
        ep = 0
        g.aout()
        # policy_out(g, model_p, scale=True)
        value_out(g, model_v)
        start = time()
        if g.turn == (1 if n % 2 == 0 else -1):
            # g.rand_put()
            # g = policy_choose(g, model_p, randomize=0.1)
            g = value_choose(g, model_v, randomize=0.02)
            # g = alpbet_choose(g, model_v, model_p, 2, width=2, randomize=1e-10)
            # g = montecarlo_policy_move(
            #     g, model_p, trail=1000, threshold=10, randomize=1)
        else:
            # g.rand_put()
            # value_out(g, model_v)
            # g = policy_choose(g, model_p, randomize=0.1)
            g = value_choose(g, model_v, randomize=0.02)
            # g = alpbet_choose(g, model_v, model_p, 2, width=2, randomize=1e-10)
            # g = montecarlo_policy_move(
            #     g, model_p, trail=1000, threshold=10, randomize=1)
            # input('a')
        end = time()
        print("{0} time: {1:.3f}".format(g.turn == (-1 if n %
                                                    2 == 0 else 1), end - start))
        if g.end_game_fast() != 0:
            break
    g.aout()
    # policy_out(g, model_p, scale=True)
    # value_out(g, model_v)
    input('go?')
    eg = g.end_game_fast()
    t += 1
    if eg == 1:
        if n % 2 == 0:
            w0 += 1
        if n % 2 == 1:
            w1 += 1
    elif eg == -1:
        if n % 2 == 1:
            w0 += 1
        if n % 2 == 0:
            w1 += 1
    print(w0, w1, t)
print(w0/t, w1/t)
