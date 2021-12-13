import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

dists = [190, 190, 190, 191, 191, 192, 192, 193, 194, 194, 196, 198, 202, 207, 210, 211, 209, 202, 193, 182, 172, 164, 158, 155, 155, 157, 159, 160, 160, 160, 159, 156, 152, 150, 151, 155, 161, 169, 180, 191, 200, 206, 208, 205, 196, 182, 167, 155]
plt.plot(dists)