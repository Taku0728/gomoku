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
n_games = 1000
epochsize = 2048
n_epoch = 10
batchsize = 128
model_v = tf_model_v()
model_p = tf_model_p()

# 学習
trail = 0
g = Game()
logs = []
with open('logs.json') as f:
    logs = json.load(f)

x_train_p_log = []
t_train_p_log = []
for game in range(n_games):
    x_train_v = []
    t_train_v = []
    x_train_p = []
    t_train_p = []
    print('game %d >>>' % game)
    start = time()
    while len(x_train_v) < epochsize or len(x_train_p) < epochsize:
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
            elif rd.random() < 1:
                g = alpbet_choose(g, model_v, model_p, 2,
                                  width=2, randomize=1e-10)
            elif rd.random() < 0:
                g = value_choose(g, model_v, randomize=1e-10)
            else:
                g = policy_choose(g, model_p, randomize=0.1)
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
    ave_loss = 0
    ave_accuracy = 0
    for epoch in range(n_epoch):
        print('epoch %d | ' % epoch, end='')
        if epoch < 1:
            # 最適化(学習)
            perm_v = np.random.permutation(len(x_train_v))
            x_batch_v = []
            t_batch_v = []
            loss = []
            for i in range(0, len(x_train_v), batchsize):
                x_batch_v = [x_train_v[j] for j in perm_v[i:i+batchsize]]
                t_batch_v = [t_train_v[j] for j in perm_v[i:i+batchsize]]
                # loss.append(0.1)
                loss.append(model_v.optimize(x_batch_v, t_batch_v))
            loss = np.mean(loss)
            ave_loss += loss / n_epoch
            print("Loss {0:.3f} | ".format(loss), end='')
    x_train_p_log.extend(x_train_p)
    t_train_p_log.extend(t_train_p)

    end = time()
    print("time:{0:.0f} | ave_loss {1:.3f} | ave_accuracy {2:.4f}".format(
        end - start, ave_loss, ave_accuracy))
    logs.append({'time': int(end - start), 'ave_loss': float(ave_loss),
                 'ave_accuracy': float(ave_accuracy)})
    with open('logs.json', 'w') as f:
        json.dump(logs, f, indent=4)
    with open('x_train_p_log.json', 'w') as f:
        json.dump(x_train_p_log, f, indent=4)
    with open('t_train_p_log.json', 'w') as f:
        json.dump(t_train_p_log, f, indent=4)
