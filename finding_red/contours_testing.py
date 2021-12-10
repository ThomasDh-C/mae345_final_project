import cv2
import numpy as np

MIN_CONTOUR_SIZE = 500.0


# ----------- HELPER FUNCS -------------
def red_filter(frame):
    """Turns camera frame into bool array of red objects
    """
    blurred = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #      h    s    v
    llb = (0,   3,   0)
    ulb = (25,  255, 255)
    lb =  (120, 3,   0)
    ub =  (180, 255, 255)

    lowreds = cv2.inRange(hsv_frame, llb, ulb)
    highreds = cv2.inRange(hsv_frame, lb, ub)
    res = cv2.bitwise_or(lowreds, highreds)

    return res


# ----------- CODE UNDER HERE -------------

frame1 = cv2.imread('filtered_test_frame.png')
frame2 = cv2.imread('wheresthered.png')

# find flow = only accepts 1 channel images
# based on https://github.com/ferreirafabio/video2tfrecord/blob/master/video2tfrecord.py
grey_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grey_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(prev=grey_frame1, next=grey_frame2, flow=None, pyr_scale=0.5, levels=4, winsize=15, iterations=4, poly_n=7, poly_sigma=1.5, flags=0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # mag matrix shape is (480, 640)


# find contours
red1 = red_filter(frame1) # shape is also (480, 640)
contours, _ = cv2.findContours(red1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
large_contours = [cont for cont in contours if cv2.contourArea(cont) > MIN_CONTOUR_SIZE]
n_contours = len(large_contours)

# find flow of each contour
av_flows = []
mask = np.zeros(red1.shape,np.uint8)
for idx in range(n_contours):
    # put 1s in every cell
    region = cv2.drawContours(mask, large_contours,  idx, 1, -1) # template, contours, index, num_to_put_in, thickness
    masked_flow_region = cv2.bitwise_and(mag, mag, mask=region) # mask of region
    av_flow = np.sum(masked_flow_region) / np.sum(region)
    av_flows.append(av_flow)

# find centre of min flow
min_flow_index = np.argmin(av_flows) # farthest obj has highest flow
farthest_contour = cv2.drawContours(mask, large_contours, min_flow_index, 1, -1) # template, contours, index, num_to_put_in, thickness
x,y,w,h = cv2.boundingRect(farthest_contour) #top left corner
c_x, c_y = (x+w/2)-640/2, 480-(y+h/2) # from centre bottom
angle_from_vertical = 90-np.arctan2(c_y,c_x)*180/np.pi # fun for debugging
print(angle_from_vertical)



# display for debugging
while(True):
    cv2.imshow('mask',cv2.drawContours(mask, large_contours, min_flow_index, 255, -1))  

    # Hit q to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()