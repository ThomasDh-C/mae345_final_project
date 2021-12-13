import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

dists = [169, 167, 170, 175, 189, 205, 223, 240, 257, 269, 277, 277, 266, 254, 253, 244, 234, 225, 216, 216, 216, 207, 208, 208, 208, 208, 209, 212, 212, 212, 212, 212, 222, 231, 228, 228, 227, 227, 217, 196, 218, 208, 198, 188, 178, 178, 178, 137, 138, 138, 139, 140, 141, 172, 173, 173, 173, 185, 192, 199, 176, 183, 201, 208, 203, 203, 203, 203, 204, 193, 194, 170, 170, 169]
test = moving_average(dists, 3)
test2 = moving_average(test, 3)
plt.plot(moving_average(test2, 3))