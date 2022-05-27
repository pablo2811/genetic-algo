import numpy as np


def distance(p1: tuple, p2: tuple):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[0]) ** 2)


def get_random_point_in_circle(r: float, x: float = 0, y: float = 0):
    t = 2 * np.pi * np.random.random()
    u = np.random.random() + np.random.random()
    if u > 1:
        a = 2 - u
    else:
        a = u
    return x + r * a * np.cos(t), y + r * a * np.sin(t)
