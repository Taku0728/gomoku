import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from PIL import Image, ImageDraw
import copy
from time import time
from gomoku import *
from search import *
size = 9
# 計算グラフの構築
n_epoch = 10000
model_v = tf_model_v()
model_p = tf_model_p()

# 学習
trail = 0
g = Game()
loss = []
accuracy = []
for epoch in range(n_epoch):
    x_batch_v = []
    t_batch_v = []
    x_batch_p = []
    t_batch_p = []
    print('epoch %d | ' % epoch, end='')
    start = time()
    while len(x_batch_p) < 256:
        # 自己対戦で学習
        g.__init__()
        g_history = []
        win = 0
        for i in range(size*size):
            # g.iout()
            # policy_out(g, model_p)
            # value_out(g, model_v)
            if rd.random() < 0:
                g.rand_put()
            if rd.random() < 1:
                g = alpbet_choose(g, model_v, model_p, 2,
                                  width=2, randomize=1e-8)
            elif rd.random() < 1:
                g = value_choose(g, model_v, randomize=1e-8)
            else:
                g = policy_choose(g, model_p, randomize=0.1)
            # g.aout()

            # g = montecarlo_move(g, model_v, model_p, 100, 10, 0)
            # ゲームの終了判定
            win = g.end_game_fast()
            g_history.append(copy.deepcopy(g))
            if win != 0:
                break
        # input('next?')
        # g_history = g_history[rd.randrange(len(g_history)):]
        g_temp = copy.deepcopy(g_history)
        for _ in range(3):
            for g_h in g_temp:
                g_h.rotate()
                g_history.append(copy.deepcopy(g_h))
        for g_h in g_temp:
            g_h.reflect()
        for _ in range(4):
            for g_h in g_temp:
                g_history.append(copy.deepcopy(g_h))
                g_h.rotate()

        for g_h in g_history:
            q_value = [0.0] if win is 0 else [-1.0] if win is 1 else [1.0]
            x_batch_v.append(g_h.square)
            t_batch_v.append(q_value)

        for n in range(len(g_history)):
            if n % len(g_temp) == 0:
                continue
            if g_history[n - 1].turn != win:
                continue
            pi, pj = 0, 0
            for i in range(size):
                for j in range(size):
                    if g_history[n - 1].square[i][j] != g_history[n].square[i][j]:
                        pi, pj = i, j
                        break
            t = [0.0 for _ in range(size*size)]
            t[pi*size + pj] = 1.0
            x_batch_p.append(g_history[n - 1].square)
            t_batch_p.append(t)
            if n % len(g_temp) == len(g_temp) - 1:
                for _ in range(3):
                    x_batch_p.append(g_history[n - 1].square)
                    t_batch_p.append(t)

    # 最適化(学習)
    perm = np.random.permutation(len(x_batch_v))
    x_batch_v, t_batch_v = [x_batch_v[p]
                            for p in perm], [t_batch_v[p] for p in perm]
    loss.append(model_v.optimize(x_batch_v, t_batch_v))

    perm = np.random.permutation(len(x_batch_p))
    x_batch_p, t_batch_p = [x_batch_p[p]
                            for p in perm], [t_batch_p[p] for p in perm]
    if len(x_batch_p) > 0:
        accuracy.append(model_p.optimize(x_batch_p, t_batch_p))
    else:
        accuracy.append(np.mean(accuracy[-100:]))
        print('no', end='')
    end = time()
    print("time:{0:.3f} | ".format(end - start), end='')
    print('Ave Loss %.3f | Ave Accuracy %.4f' %
          (np.mean(loss[-100:]), np.mean(accuracy[-100:])))
