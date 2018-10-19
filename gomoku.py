
# coding: utf-8

# # Tensorflowを用いた強化学習の実装
#
# 機械学習用ライブラリTensorflowを用いて、Pythonで強化学習を実装します。<br>
# 今回のハンズオンでは、五目並べんお実装とモンテカルロ法によって自己対戦で強化学習をさせます。

# In[1]:


import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from PIL import Image, ImageDraw
import copy
import os
import shutil
# from numba import jit, jitclass, int32, float32, typeof, deferred_type
from functools import reduce
from math import exp, log, sqrt
from mymath import *

# In[2]:


size = 9

# 五目並べのプログラム

# @jitclass
class Game:
    # コンストラクタ
    def __init__(self):
        self.size = size
        self.square = [[0 for _ in range(self.size)] for _ in range(self.size)]
        # self.square = np.zeros((size, size), dtype=np.int64)
        self.turn = 1
    # 石を置く

    def count(self):
        n = 0
        for r in self.square:
            for c in r:
                if c != 0:
                    n += 1
        return n

    # @property
    def put(self, row, column):
        if 0 <= row < self.size and 0 <= column < self.size:
            self.square[row][column] = self.turn
        self.turn = -1 if self.turn is 1 else 1
    # 石がおけるかどうか

    def putable(self, row, column):
        if 0 <= row < self.size and 0 <= column < self.size:
            return self.square[row][column] is 0
        else:
            return 0
    # ゲーム終了判定(colorの勝ち)

    def end_game(self):
        for color in [-1, 1]:
            direction = [[-1, -1], [-1, 0], [-1, 1],
                         [0, -1], [0, 1], [1, -1], [1, 0], [-1, -1]]
            for i in range(self.size):
                for j in range(self.size):
                    for d in direction:
                        if self.fives(color, i, j, d):
                            return color
        return 0

    # 五目がつながっている判定
    def fives(self, color, i, j, d):
        number = 0
        while 0 <= i < self.size and 0 <= j < self.size and self.square[i][j] is color:
            number += 1
            i += d[0]
            j += d[1]
        if number >= 5:
            return 1
        else:
            return 0
    # 盤面を表示

    def iout(self):
        img = np.asarray([[[0, 256, 0] if i is 0 else [0, 0, 0] if i is 1 else [
                         256, 256, 256] for i in l] for l in self.square])
        plt.imshow(img)
        plt.show()

    def aout(self):
        for i in self.square:
            for j in i:
                if j is 0:
                    print('ー', end='')
                elif j is 1:
                    print('＊', end='')
                else:
                    print('０', end='')
            print('')
        print('')
    
    def rand_put(self):
        ij = [[i, j] for j in range(self.size) for i in range(self.size) if self.square[i][j] == 0]
        if len(ij) > 0:
            i, j = ij[rd.randrange(len(ij))]
            self.put(i, j)

    # 次のありえるすべての盤面
    def next_nodes(self):
        n = []
        for i in range(self.size):
            for j in range(self.size):
                if self.putable(i, j):
                    n.append(copy.deepcopy(self))
                    n[-1].put(i, j)
        return n

    def game_set(self, search_depth=3):
        nodes = [self]
        for i in range(search_depth + 1):
            if i != search_depth:
                n_nodes = []
                for n in nodes:
                    n_nodes.extend(n.next_nodes())
                nodes = copy.deepcopy(n_nodes)
            for n in nodes:
                eg = n.end_game_fast()
                if eg != 0:
                    return eg
        return 0

    # 入力による着手
    def input_put(self):
        i, j = -1, -1
        while (not 0 <= i < self.size) or (not 0 <= j < self.size) or not self.putable(i, j):
            i, j = map(int, input('input (row, column)').split())
        self.put(i, j)

    # @jit
    def end_game_fast(self):
        for i in self.square:
            color, seq = 0, 0
            for j in i:
                c = j
                if c == color:
                    seq += 1
                    if seq == 5 and c != 0:
                        return c
                else:
                    color = c
                    seq = 1
        size = self.size
        for j in range(size):
            color, seq = 0, 0
            for i in range(size):
                c = self.square[i][j]
                if c == color:
                    seq += 1
                    if seq == 5 and c != 0:
                        return c
                else:
                    color = c
                    seq = 1

        for si in range(size)[:-4]:
            i, j, color, seq = si, 0, 0, 0
            while 0 <= i < size and 0 <= j < size:
                c = self.square[i][j]
                if c == color:
                    seq += 1
                    if seq == 5 and c != 0:
                        return c
                else:
                    color = c
                    seq = 1
                i += 1
                j += 1

        for sj in range(size)[1:-4]:
            i, j, color, seq = 0, sj, 0, 0
            while 0 <= i < size and 0 <= j < size:
                c = self.square[i][j]
                if c == color:
                    seq += 1
                    if seq == 5 and c != 0:
                        return c
                else:
                    color = c
                    seq = 1
                i += 1
                j += 1

        for si in range(size)[:-4]:
            i, j, color, seq = si, size - 1, 0, 0
            while 0 <= i < size and 0 <= j < size:
                c = self.square[i][j]
                if c == color:
                    seq += 1
                    if seq == 5 and c != 0:
                        return c
                else:
                    color = c
                    seq = 1
                i += 1
                j += -1

        for sj in range(size)[4:-1]:
            i, j, color, seq = 0, sj, 0, 0
            while 0 <= i < size and 0 <= j < size:
                c = self.square[i][j]
                if c == color:
                    seq += 1
                    if seq == 5 and c != 0:
                        return c
                else:
                    color = c
                    seq = 1
                i += 1
                j += -1
        return 0
    
    def end_put(self, i, j):
        direction = [[0, 1], [1, 1], [1, 0], [1, -1]]
        color = 1
        if self.square[i][j] == 0:
            return 0
        elif self.square[i][j] == -1:
            color = -1
        for n, d in enumerate(direction):
            li, lj = i, j
            number = 1
            ei0 = min(4, self.size - i - 1) if n != 0 else 4
            ei1 = min(4, i) if n != 0 else 4
            ej0 = 4 if n == 2 else min(4, j) if n == 3 else min(4, self.size - j - 1)
            ej1 = 4 if n == 2 else min(4, self.size - j - 1) if n == 3 else min(4, j)
            e0 = min(ei0, ej0)
            e1 = min(ei1, ej1)

            if e0 + e1 < 4:
                break
            for _ in range(e0):
                li += d[0]
                lj += d[1]
                if self.square[li][lj] != color:
                    break
                number += 1
            li, lj = i, j 
            for _ in range(e1):
                li -= d[0]
                lj -= d[1]
                if self.square[li][lj] != color:
                    break
                number += 1
            if number >= 5:
                return color
        return 0

    def rand_end(self):
        stones = self.count()
        black, white = 0, 0
        if stones % 2 == 0:
            black, white = int((size*size-stones)/2), int((size*size-stones)/2)
        else:
            black, white = int((size*size-1-stones)/2), int((size*size+1-stones)/2)
        ran = np.random.rand((size*size-stones)*2)
        p = 0
        for r in self.square:
            for c in r:
                if c == 0:
                    b, w = ran[p]*black, ran[p+1]*white
                    if b > w:
                        c = 1
                        black -= 1
                    else:
                        c = -1
                        white -= 1
                    p += 2
        return self.end_game_fast()

    def rotate(self):
        s = copy.deepcopy(self.square)
        for i in range(self.size):
            for j in range(self.size):
                self.square[i][j] = s[j][self.size-i-1]
        
    
    def reflect(self):
        s = copy.deepcopy(self.square)
        for i in range(self.size):
            for j in range(self.size):
                self.square[i][j] = s[j][i]

# In[3]:


# In[4]:


def value_choose(g, model, randomize=0, end_move=True):
    if rd.random() < randomize:
        g.rand_put()
        return g
    next_nodes = g.next_nodes()
    next_values = []
    for node in next_nodes:
        # 着手の評価値を予測
        if end_move and node.end_game_fast() == g.turn:
            return node
        if randomize > 0:
            next_values.append((1/randomize)**((-1.0 if g.turn == 1 else 1.0)
                                               * model.out([node.square])[0]))
        else:
            next_values.append((-1.0 if g.turn == 1 else 1.0)
                                               * model.out([node.square])[0])
    if randomize == 0:
        g = next_nodes[rargmax(next_values)]
    else:
        sum_value = sum(next_values)
        # 確率的最適手
        p_value, a_value = 0, rd.random()
        index = 0
        for ind, v in enumerate(next_values):
            p_value += v / sum_value
            if p_value >= a_value:
                index = ind
                break
        g = next_nodes[index]
    return g


def policy_choose(g, model, randomize=0, end_move=False):
    if rd.random() < randomize:
        g.rand_put()
        return g
    y = np.array(model.out([g.square])[0])
    # print(y)
    if end_move:
        next_nodes = g.next_nodes()
        for node in next_nodes:
            # 着手の評価値を予測
            if node.end_game_fast() == g.turn:
                return node
    if randomize > 0:
        y += 1e-10
        for i in range(len(y)):
            if g.square[int(i/g.size)][i % g.size] != 0:
                y[i] = 0
        y /= sum(y)
        p_value, a_value = 0, rd.random()
        index = 0
        for i in range(len(y)):
            p_value += y[i]
            if p_value >= a_value:
                index = i
                break
        g.put(int(index/g.size), index % g.size)
        return g
    pi = rargmax(y)
    i, j = int(pi/g.size), pi % g.size
    while g.square[i][j] != 0:
        y[pi] = -float('inf')
        pi = np.rargmax(y)
        i, j = int(pi/g.size), pi % g.size
    g.put(i, j)
    return g

def next_nodes_policy(g, model, prop=0.66):
    space = size*size - g.count()
    if space < 10:
        return g.next_nodes()
    y = model.out([g.square])[0]
    vy = [[y[i], int(i/size), i%size] for i in range(len(y))]
    gs = []
    vy.sort(key=lambda x: -x[0])
    s = space
    i = 0
    while s > space * prop + 3:
        if g.square[vy[i][1]][vy[i][2]] == 0:
            ng = copy.deepcopy(g)
            ng.put(vy[i][1], vy[i][2])
            gs.append(ng)
            s -= 1
        i += 1
    return gs

def alpbet(g, model_v, model_p, table, depth=4, width=2, alp=-float('inf'), bet=float('inf')):
    key = str(g.square)
    lower, upper = -1, 1
    if key in table:
        lower = table[key]['lower']
        if bet <= lower:
            return lower
        upper = table[key]['upper']
        if upper <= alp:
            return upper
        alp, bet = max(alp, lower), min(bet, upper)
    else:
        table[key] = {}

    eg = g.end_game_fast()
    if eg != 0:
        v = 1 if eg == g.turn else -1
        table[key]['upper'] = table[key]['lower'] = v
        return v
    if depth == 0:
        v = model_v.out([g.square])[0]
        table[key]['upper'] = table[key]['lower'] = v
        return v
    nodes = g.next_nodes()
    # nodes = next_nodes_policy(g, model_p)
    values = []
    for i, n in enumerate(nodes):
        eg = n.end_game_fast()
        values.append([i, (-1 if g.turn is 1 else 1) * model_v.out(
            [n.square])[0] if eg == 0 else 1 if eg == g.turn else -1])
    values.sort(key=lambda x: -x[1])
    limit = min(len(values) - 1, int(sqrt(len(values))*width))
    branch = [nodes[v[0]] for v in values[:limit]]
    v = -1
    for i, b in enumerate(branch):
        if depth == 1 or b.count() == b.size*b.size-1:
            v = values[i][1]
        else:
            v = -alpbet(b, model_v, model_p, table, depth-1, -bet, -alp)
        if v >= bet:
            table[key]['lower'], table[key]['upper'] = v, 1
            return v
        alp = max(alp, v)
    table[key]['upper'] = table[key]['lower'] = v
    return v

def alpbet_choose(g, model_v, model_p, depth=4, width=2, randomize=0):
    eg = g.end_game_fast()
    if eg != 0:
        return 1 if eg == g.turn else -1
    table = {}
    if depth == 0:
        return model_v.out([g.square])[0]
    nodes = g.next_nodes()
    # nodes = next_nodes_policy(g, model_p)
    values = []
    for i, n in enumerate(nodes):
        eg = n.end_game_fast()
        values.append([i, (-1 if g.turn is 1 else 1) * model_v.out(
            [n.square])[0] if eg == 0 else 1 if eg == g.turn else -1])
    values.sort(key=lambda x: -x[1])
    limit = min(len(values) - 1, int(sqrt(len(values))*width))
    # print(width + 1, max([v[1] for v in self.values[:width]]))
    branch = [nodes[v[0]] for v in values[:limit]]
    if randomize == 0:
        alp, bet = -float('inf'), float('inf')
        for i, b in enumerate(branch):
            v = -alpbet(b, model_v, model_p, table, depth-1, width, -bet, -alp)
            if v >= alp:
                g = b
            alp = max(alp, v)
    else:
        values = []
        for i, b in enumerate(branch):
            v = (1/randomize)**-alpbet(b, model_v, model_p, table, depth-1, width)
            values.append(v)
        values = values / np.sum(values)
        # print(values)
        p_value, a_value = 0, rd.random()
        index = 0
        for i, v in enumerate(values):
            p_value += v
            if p_value >= a_value:
                index = i
                break
        g = branch[index]
    return g

def p_play(g, model, randomize=0):
    g1 = copy.deepcopy(g)
    spaces = g1.size*g1.size - g1.count()
    eg = g1.end_game_fast()
    if eg != 0:
        return eg
    for _ in range(spaces):
        i, j = 0, 0
        if rd.random() < randomize:
            ij = [[s, t] for t in range(g.size) for s in range(g1.size) if g1.square[s][t] == 0]
            if len(ij) > 0:
                i, j = ij[rd.randrange(len(ij))]
                g1.put(i, j)
        else:
            y = np.array(model.out([g1.square])[0])
            for i in range(len(y)):
                if g1.square[int(i/g1.size)][i%g1.size] == 0:
                    y[i] = 0
            y = y + 0.0001
            y = y / np.sum(y)
            p_value, a_value = 0, rd.random()
            index = 0
            for i in range(len(y)):
                p_value += y[i]
                if p_value >= a_value:
                    index = i
                    break
            i, j = int(index/g1.size), index%g1.size
            g1.put(i, j)
        ep = g1.end_put(i, j)
        if ep != 0:
            return ep
    return 0

def policy_out(g, model, scale=False):
    y = np.array(model.out([g.square])[0])
    # print(y)
    value_map = [[y[i*g.size+j] for j in range(g.size)] for i in range(g.size)]
    out_map = [[[0, 0, 0] for _ in range(g.size)] for __ in range(g.size)]
    max_v = np.amax(y) if scale else 1
    min_v = np.amin(y) if scale else 0
    print(max_v, min_v)
    for i in range(g.size):
        for j in range(g.size):
            s = g.square[i][j]
            if max_v != min_v:
                v = int((value_map[i][j] - min_v) / (max_v - min_v) * 255)
            else:
                v = int(value_map[i][j] * 255)
            if s == 1:
                out_map[i][j] = [0, 0, 0]
            elif s == -1:
                out_map[i][j] = [255, 255, 255]
            else:
                if v > 127:
                    out_map[i][j] = [v, 255-v, 0]
                else:
                    out_map[i][j] = [0, v, 255-v]
    plt.imshow(np.array(out_map))
    plt.title(u"Policy Image")
    plt.show()

def value_out(g, model):
    out_map = [[[0, 0, 0] for _ in range(g.size)] for __ in range(g.size)]
    max_v = -float('inf')
    min_v = float('inf')
    for i in range(g.size):
        for j in range(g.size):
            s = g.square[i][j]
            v = 0
            if g.square[i][j] == 0:
                new_g = copy.deepcopy(g)
                new_g.put(i, j)
                v = (-1 if g.turn == 1 else 1) * model.out([new_g.square])[0]
                max_v = max(max_v, v)
                min_v = min(min_v, v)
            v = int((v * 255 + 255) / 2)
            if s == 1:
                out_map[i][j] = [0, 0, 0]
            elif s == -1:
                out_map[i][j] = [255, 255, 255]
            else:
                if v > 127:
                    out_map[i][j] = [v, 255-v, 0]
                else:
                    out_map[i][j] = [0, v, 255-v]
    print(max_v, min_v)
    plt.imshow(np.array(out_map))
    plt.title("Value Image")
    plt.show()



# In[5]:


def cnn(x):
    x_image = tf.reshape(x, [-1, size, size, 1])    # [None, size, size, 1]
    conv1 = tf.layers.conv2d(x_image, 128, (3, 3), padding='same',
                             activation=tf.nn.relu)    # [None, size, size, 128]
    conv2 = tf.layers.conv2d(
        conv1, 128, (3, 3), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
        conv2, 128, (3, 3), padding='same', activation=tf.nn.relu)
    # [None, size, size, 128]
    pool2_flat = tf.layers.flatten(conv3)
    dense1 = tf.layers.dense(pool2_flat, 64, activation=tf.nn.relu)
    y = tf.layers.dense(dense1, 1)
    return y


def cnn_p(x):
    x_image = tf.reshape(x, [-1, size, size, 1])    # [None, size, size, 1]
    conv1 = tf.layers.conv2d(x_image, 128, (3, 3), padding='same',
                             activation=tf.nn.relu)    # [None, size, size, 128]
    conv2 = tf.layers.conv2d(
        conv1, 128, (3, 3), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
        conv2, 128, (3, 3), padding='same', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(
        conv3, 128, (3, 3), padding='same', activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(
        conv4, 128, (3, 3), padding='same', activation=tf.nn.relu)
    pool2_flat = tf.layers.flatten(conv5)    # [None, 7*7*64 = 3136]
    dense1 = tf.layers.dense(pool2_flat, size*size, activation=tf.nn.relu)
    y = tf.layers.dense(dense1, size*size, activation=tf.nn.softmax)
    # y = tf.layers.dense(pool2_flat, size*size, activation=tf.nn.softmax)
    return y


class tf_model_v:
    def __init__(self, size=9):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, (None, size, size))
        self.t = tf.placeholder(tf.float32, (None, 1))
        self.y = cnn(self.x)
        self.trail = 0
        self.model_path = './model_v/'
        self.logs_path = './log_v/'
        self.saver = tf.train.Saver()
        self.cost = tf.reduce_mean(tf.square(self.y - self.t))
        self.optimizer = tf.train.GradientDescentOptimizer(2e-2).minimize(self.cost)
        self.train_summary_loss = tf.summary.scalar('train_loss', self.cost)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(
            self.logs_path, graph=tf.get_default_graph())
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        else:
            self.load()

    def load(self):
        self.saver.restore(self.sess, self.model_path)

    def new_logs(self):
        if os.path.exists(self.logs_path):
            shutil.rmtree(self.logs_path)
        os.mkdir(self.logs_path)

    # モデル予測値の出力
    def out(self, X):
        return self.y.eval(feed_dict={self.x: X}, session=self.sess)

    # 最適化
    def optimize(self, X, T):
        _, cost, summary_loss = self.sess.run(
            [self.optimizer, self.cost, self.train_summary_loss], feed_dict={self.x: X, self.t: T})
        # ログの保存
        # self.summary_writer.add_summary(summary_loss, self.trail)
        self.trail += 1
        # モデルの保存
        self.saver.save(self.sess, self.model_path)
        return cost


class tf_model_p:
    def __init__(self, size=9):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, (None, size, size))
        self.t = tf.placeholder(tf.int64, (None, size*size))
        self.y = cnn_p(self.x)
        self.trail = 0
        self.model_path = './model_p/'
        self.logs_path = './log_p/'
        self.saver = tf.train.Saver()
        self.cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=self.t, logits=self.y)
        self.optimizer = tf.train.GradientDescentOptimizer(
            1e-2).minimize(self.cross_entropy)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.t, 1)), tf.float32))
        self.summary_accuracy = tf.summary.scalar('train_loss', self.accuracy)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(
            self.logs_path, graph=tf.get_default_graph())
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        else:
            self.load()

    def load(self):
        self.saver.restore(self.sess, self.model_path)

    def new_logs(self):
        if os.path.exists(self.logs_path):
            shutil.rmtree(self.logs_path)
        os.mkdir(self.logs_path)

    # モデル予測値の出力
    def out(self, X):
        return self.y.eval(feed_dict={self.x: X}, session=self.sess)

    # 最適化
    def optimize(self, X, T):
        _, accuracy_, summary_accuracy_ = self.sess.run(
            [self.optimizer, self.accuracy, self.summary_accuracy], feed_dict={self.x: X, self.t: T})
        # ログの保存
        # self.summary_writer.add_summary(summary_accuracy_, self.trail)
        self.trail += 1
        # モデルの保存
        self.saver.save(self.sess, self.model_path)
        return accuracy_
