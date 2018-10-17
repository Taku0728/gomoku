import numpy as np
import random as rd


def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    print(indices)
    return rd.choice(indices)
