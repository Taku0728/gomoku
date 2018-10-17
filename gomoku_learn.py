import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from PIL import Image, ImageDraw
import copy
from gomoku import *

size = 9
# 計算グラフの構築
n_epoch = 10000
model_v = tf_model_v()
# model_p = tf_model_p()

# 学習
trail = 0
g = Game()
for epoch in range(n_epoch):
    x_batch_v = []
    t_batch_v = []
    # x_batch_p = []
    # t_batch_p = []
    print('epoch %d | ' % epoch, end='')
    while len(x_batch_v) < 1024 - 64:
        # while len(x_batch_p) < 1024 - 64:
        # 自己対戦で学習
        g.__init__()
        g_history = []
        win = 0
        for i in range(size*size):
            # g.iout()
            # g = rand_game(g)
            # g = best_game_p(g, model_v)
            g.rand_put()
            # ゲームの終了判定
            win = g.end_game_fast()
            g_history.append(copy.deepcopy(g))
            if win != 0:
                break

        # すべての盤面についてバッチを作る
        point = rd.randrange(len(g_history) - 1) + 1
        # point = rd.randrange(min(len(g_history), int(epoch/100) + 1)) + 1
        # for g_h in g_history[-1:]:
        for g_h in g_history[-point:]:
            x_batch_v.append(g_h.square)
            q_value = [0.0] if win is 0 else [-1.0] if win is 1 else [1.0]
            t_batch_v.append(q_value)

        # for n in range(len(g_history))[1:]:
        #     if g_history[n - 1].turn != win:
        #         continue
        #     pi, pj = 0, 0
        #     for i in range(size):
        #         for j in range(size):
        #             if g_history[n - 1].square[i][j] != g_history[n].square[i][j]:
        #                 pi, pj = i, j
        #                 break

        #     t = [0 for _ in range(size*size)]
        #     t[pi*size + pj] = 1
        #     x_batch_p.append(g_history[n - 1].square)
        #     t_batch_p.append(t)

    # 最適化(学習)
    perm = np.random.permutation(len(x_batch_v))
    x_batch_v, t_batch_v = [x_batch_v[p]
                            for p in perm], [t_batch_v[p] for p in perm]
    cost = model_v.optimize(x_batch_v, t_batch_v)
    # perm = np.random.permutation(len(x_batch_p))
    # x_batch_p, t_batch_p = [x_batch_p[p]
    #                         for p in perm], [t_batch_p[p] for p in perm]
    # accuracy = model_p.optimize(x_batch_p, t_batch_p)
    print('Train loss %.4f | Accuracy %.4f' % (cost, 1.0))
