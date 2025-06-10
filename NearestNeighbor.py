import numpy as np
from numpy.ma.core import argmax


class NearestNeighbor:

    def __init__(self, k, p):

        self.k = k
        self.p = p

    def lp_distance(self, p1, p2):
        """
        compute the distance b etween two points in the lp space
        :param p1: point 1 (vector)
        :param p2: point 2 (vector)
        :return: the distance between the two points (scalar)
        """
        assert p1.shape == p2.shape

        sum = 0
        for i in range(p1.shape[0]):
            sum += (abs(p1[i] - p2[i]))**self.p

        return sum**(1/self.p)

    def find_K_neighbors(self, p, p_set):

        neighbors = [None] * self.k
        for pt in p_set:
            pos, label = pt
            distance = self.lp_distance(pos, p)
            if None in neighbors:
                neighbors[neighbors.index(None)] = (distance, label)
                continue

            if distance < neighbors[0][0]:
                neighbors.append((distance, label))

            neighbors.sort(key=lambda x: x[0], reverse=False)

        label = [0, 0, 0]
        for pt in neighbors:
            label[pt[1]] += 1

        return np.argmax(label)