import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time

blank_image = cv2.imread('wheresthered_og.png')
other = cv2.imread('wheresthered_problematic.png')


ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
im1 = ax1.imshow(blank_image)
im2 = ax2.imshow(other)
# plt.ion()

def update(i):
    if(i%2==0):
        im1.set_data(blank_image)
        im2.set_data(other)
    else:
        im1.set_data(other)
        im2.set_data(blank_image)
ani = FuncAnimation(plt.gcf(), update, interval=100)
plt.show() # blocking though and blocking = false causes it to skip animating

print('hopefully past the setpoint')
# start = time.time()
# iterations = 6
# for i in range(600):
#     # If wanting to see an "animation" of points added, add a pause to allow the plotting to take place
#     # plt.pause(0.01)
#     im1.set_data(blank_image)
#     plt.pause(0.01)
#     im1.set_data(other)
    
# print('Update frequency is ... in Hz')
# print(iterations*2/ (time.time()-start))