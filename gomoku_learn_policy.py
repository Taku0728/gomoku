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
n_epoch = 50
batchsize = 128
model_p = tf_model_p()

# 学習
trail = 0
g = Game()
logs = []
with open('x_train_p.json') as f:
    x_train_p = json.load(f)
with open('t_train_p.json') as f:
    t_train_p = json.load(f)
print("Boards: ", len(x_train_p))
for epoch in range(n_epoch):
    start = time()
    print('epoch %d | ' % epoch, end='')
    perm_p = np.random.permutation(len(x_train_p))
    x_batch_p = []
    t_batch_p = []
    accuracy = []
    for i in range(0, len(x_train_p), batchsize):
        x_batch_p = [x_train_p[j] for j in perm_p[i:i+batchsize]]
        t_batch_p = [t_train_p[j] for j in perm_p[i:i+batchsize]]
        # accuracy.append(0.9)
        accuracy.append(model_p.optimize(x_batch_p, t_batch_p))
    accuracy = np.mean(accuracy)
    end = time()
    print("time:{0:.0f} | accuracy {1:.4f} ".format(end - start, accuracy))
