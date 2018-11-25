import tensorflow as tf
import numpy as np
import random as rd
import copy
from time import time
from gomoku import *
from search import *
import json

size = 9
# 計算グラフの構築
n_learn = 10000
n_epoch = 5
batchsize = 128
gamenumber = 10
model_v = tf_model_v()

# 学習
g = Game()
with open('x_train_p.json') as f:
    x_train_p = json.load(f)
with open('t_train_p.json') as f:
    t_train_p = json.load(f)
for learn in range(n_learn):
    x_train_v = []
    t_train_v = []
    start = time()
    print('learn %d' % learn)
    for i in range(gamenumber):
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
            elif rd.random() < 0:
                g = alpbet_choose(g, model_v, model_p, 2,
                                  width=2, randomize=1e-10)
            elif rd.random() < 1:
                g = value_choose(g, model_v, randomize=0.05)
            # g = montecarlo_move(g, model_v, model_p, 100, 10, 0)
            # ゲームの終了判定
            win = g.end_game_fast()
            g_history.append(copy.deepcopy(g))
            if win != 0:
                break
        # input('next?')
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
            x_train_v.append(g_h.square)
            t_train_v.append(q_value)

        for n in range(len(g_history)):
            if n % len(g_temp) == 0:
                continue
            if g_history[n - 1].turn != win:
                continue
            i, j = 0, 0
            for ij in range(g.size*g.size):
                i, j = int(ij/g.size), ij % g.size
                if g_history[n - 1].square[i][j] != g_history[n].square[i][j]:
                    break
            t = [0.0 for _ in range(size*size)]
            t[i*size + j] = 1.0
            x_train_p.append(g_history[n - 1].square)
            t_train_p.append(t)
            if n % len(g_temp) == len(g_temp) - 1:
                for _ in range(3):
                    x_train_p.append(g_history[n - 1].square)
                    t_train_p.append(t)
    # with open('x_train_p.json', 'w') as f:
    #     json.dump(x_train_p, f, indent=4)
    # with open('t_train_p.json', 'w') as f:
    #     json.dump(t_train_p, f, indent=4)

    ave_loss = 0
    # 最適化(学習)
    print('boards: ', len(x_train_v))
    for epoch in range(n_epoch):
        print('epoch %d | ' % epoch, end='')
        perm_v = np.random.permutation(len(x_train_v))
        x_batch_v = []
        t_batch_v = []
        loss = 0
        for i in range(0, len(x_train_v), batchsize):
            x_batch_v = [x_train_v[j] for j in perm_v[i:i+batchsize]]
            t_batch_v = [t_train_v[j] for j in perm_v[i:i+batchsize]]
            # loss.append(0.1)
            loss += model_v.optimize(x_batch_v, t_batch_v)
        loss /= int(len(x_train_v) / batchsize)
        print("loss {0:.3f}".format(loss))
        ave_loss += loss
    ave_loss /= n_epoch
    end = time()
    print("time:{0:.0f} | ave_loss {1:.3f}".format(
        end - start, ave_loss))
