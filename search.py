import numpy as np
import random as rd
import copy
from gomoku import *
from math import exp, log, sqrt


def getQC(w, v, n, N):
    return w / (n+1.0) + 2*sqrt(abs(v + 1e-6)*2.0*log(N+2.0))/(n+1.0)


class Node:
    def __init__(self, g, threshold, value=0, win=None):
        self.root = copy.deepcopy(g)
        self.value = value
        self.values = []
        self.trail = 0
        self.ntrail = 0
        self.branch = []
        self.leaf = 0
        self.threshold = threshold
        self.dic = {}
        eg = self.root.end_game_fast() if win == None else 0
        self.leaf = (0 if eg == 0 else 1 if eg ==
                     self.root.turn else -1) if win == None else win
        self.win = self.leaf if self.leaf == 0 else float(
            'inf') if self.leaf == 1 else -float('inf')
        self.dict_q = []

    def get_value(self, model_v=None, model_p=None, depth=4):
        if self.leaf != 0:
            return self.leaf
        result = 0
        if self.trail < self.threshold:
            result = self.play(model_v, model_p, depth)
            self.trail += 1
        else:
            if self.trail == self.threshold:
                nodes = next_nodes_policy(self.root, model_p)
                # self.root.aout()
                for i, n in enumerate(nodes):
                    eg = n.end_game_fast()
                    self.values.append([i, 10000**((-1 if n.turn is 1 else 1) * model_v.out(
                        [n.square])[0]), 0 if eg == 0 else 1 if eg == n.turn else -1])
                self.values.sort(key=lambda x: (x[2], x[1]))
                width = min(len(self.values) - 1,
                            int(sqrt(len(self.values))*2))
                s_v = sum([v[1] for v in self.values[:width]])
                for v in self.values[:width]:
                    v[1] /= s_v
                # print(width + 1, max([v[1] for v in self.values[:width]]))
                self.branch = [Node(nodes[v[0]], self.threshold, v[1], v[2])
                               for v in self.values[:width] if v[2] != 1]
                self.trail += 1
            self.ntrail = sum([b.trail + b.ntrail for b in self.branch])
            self.dict_q = [getQC(-b.win, b.value,
                                 b.trail, self.ntrail) for b in self.branch]
            index = self.dict_q.index(max(self.dict_q))
            result = - self.branch[index].get_value(model_v, model_p, depth)
        self.win += result
        return result

    def pi_search(self, model_v, model_p):
        result = 0
        if self.leaf != 0:
            self.trail += 1
            return self.win
        if self.trail < self.threshold:
            result = (-1.0 if self.root.turn == 1 else 1.0) * \
                model_v.out([self.root.square])[0]
            # result = self.play(model_v, model_p, 0, end_move=False)
        else:
            if self.trail == self.threshold:
                self.dict_q = [0 for _ in range(
                    self.root.size*self.root.size)]
                y = model_p.out([self.root.square])[0]
                self.values = 2**y
                self.values = self.values / np.sum(self.values)
                # print(self.values)
            for ij in range(self.root.size*self.root.size):
                if ij in self.dic:
                    b = self.branch[self.dic[ij]]
                    self.dict_q[ij] = getQC(- b.win,
                                            self.values[ij], b.trail, self.trail)
                else:
                    self.dict_q[ij] = getQC(
                        0, self.values[ij], 0, self.trail)
            ij = np.argmax(self.dict_q)
            if ij not in self.dic:
                node = copy.deepcopy(self.root)
                i, j = int(ij / node.size), ij % node.size
                while node.square[i][j] != 0:
                    self.dict_q[ij] = -float('inf')
                    ij = np.argmax(self.dict_q)
                    i, j = int(ij / node.size), ij % node.size
                node.put(i, j)
                self.branch.append(Node(node, self.threshold))
                self.dic[ij] = len(self.branch) - 1
            b = self.branch[self.dic[ij]]
            result = - b.pi_search(model_v, model_p)
        self.trail += 1
        self.win += result
        return result

    def calc_v_value(self, model):
        eg = self.root.end_game_fast()
        if eg != 0:
            return 1 if eg == self.root.turn else -1
        return (-1 if self.root.turn is 1 else 1) * model.out([self.root.square])[0]

    def play(self, model_v, model_p, depth=0, end_move=True):
        game = copy.deepcopy(self.root)
        c = game.count()
        e = 0
        value = 0
        for _ in range(depth):
            e = game.end_game_fast()
            if c >= game.size*game.size or e is not 0:
                break
            game = policy_choose(
                game, model_p, randomize=0.001, end_move=end_move)
            c += 1
        if c < game.size*game.size and e == 0:
            if depth > 0:
                value += 0.5 * (-1 if self.root.turn == 1 else 1) * \
                    model_v.out([game.square])[0]
        else:
            value += 0 if e == 0 else 0.5 if e == self.root.turn else -0.5
        e = game.rand_end()
        value += 0 if e == 0 else 0.5 if e == self.root.turn else -0.5
        if depth == 0:
            value *= 2
        return value


class MonteCarlo_Node:
    def __init__(self, g, threshold, dic, leaf=0):
        self.root = copy.deepcopy(g)
        self.branch_values = []
        self.trails = 0
        self.branch_trails = 0
        self.branch = []
        self.leaf = 0
        self.threshold = threshold
        self.leaf = leaf
        self.win = 0 if self.leaf == 0 else float(
            'inf') if self.leaf == 1 else -float('inf')
        self.dic_q = []
        self.dic_b = {}
        # ss = str(self.root.square)
        # if ss in dic:
        #     self = dic[ss]
        # else:
        #     dic[ss] = self

    def p_search(self, model_p, dic, randomize=0):
        if self.leaf != 0:
            self.trails += 1
            return self.win
        if self.trails < self.threshold:
            # pp = p_play(self.root, model_p, randomize)
            pp = self.root.rand_end()
            result = 0 if pp == 0 else 1 if pp == self.root.turn else -1
            self.trails += 1
        else:
            if self.trails == self.threshold and self.branch_trails == 0:
                self.dic_q = [0 for _ in range(
                    self.root.size*self.root.size)]
                self.values = model_p.out([self.root.square])[0]
                self.values = self.values / np.sum(self.values)
            for ij in range(self.root.size*self.root.size):
                if ij in self.dic_b:
                    b = self.branch[self.dic_b[ij]]
                    self.dic_q[ij] = getQC(- b.win, self.values[ij],
                                           b.trails + b.branch_trails, self.branch_trails)
                else:
                    self.dic_q[ij] = getQC(
                        0, self.values[ij], 0, self.branch_trails)
            ij = np.argmax(self.dic_q)
            if ij not in self.dic_b:
                node = copy.deepcopy(self.root)
                i, j = int(ij / node.size), ij % node.size
                while node.square[i][j] != 0:
                    self.dic_q[ij] = -float('inf')
                    self.values[ij] = -float('inf')
                    ij = np.argmax(self.dic_q)
                    i, j = int(ij / node.size), ij % node.size
                node.put(i, j)
                ep = node.end_put(i, j)
                self.branch.append(MonteCarlo_Node(
                    node, self.threshold, dic, ep))
                self.dic_b[ij] = len(self.branch) - 1
            b = self.branch[self.dic_b[ij]]
            result = - b.p_search(model_p, dic, randomize)
            self.branch_trails += 1
            # self.branch_trails = sum(
            #     [b.trails + b.branch_trails for b in self.branch])
        self.win += result
        return result


def montecarlo_move(g, model_v, model_p, trail=10000, threshold=10, depth=4):
    if g.end_game_fast() != 0:
        return g
    node = Node(g, threshold)
    for _ in range(trail):
        __ = node.pi_search(model_v, model_p)
    routes = [b.trail + b.ntrail for b in node.branch]
    return node.branch[routes.index(max(routes))].root


def montecarlo_policy_move(g, model_p, trail=10000, threshold=10, randomize=0):
    if g.end_game_fast() != 0:
        return g
    dic = {}
    node = MonteCarlo_Node(g, threshold, dic)
    for _ in range(trail):
        v = node.p_search(model_p, dic, randomize)
    routes = [b.trails + b.branch_trails for b in node.branch]
    return node.branch[routes.index(max(routes))].root
