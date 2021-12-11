# COPIED FROM https://github.com/botforge/ColorTrackbar/blob/master/HSV%20Trackbar.py
import cv2 as cv
import numpy as np

# optional argument for trackbars
def nothing(x):
    pass

# named ites for easy reference
barsWindow = 'Bars'
hl = 'H Low'
hh = 'H High'
sl = 'S Low'
sh = 'S High'
vl = 'V Low'
vh = 'V High'

# set up for video capture on camera 0
cap = cv.VideoCapture(0)

# create window for the slidebars
cv.namedWindow(barsWindow, flags = cv.WINDOW_AUTOSIZE)

# create the sliders
cv.createTrackbar(hl, barsWindow, 0, 179, nothing)
cv.createTrackbar(hh, barsWindow, 0, 179, nothing)
cv.createTrackbar(sl, barsWindow, 0, 255, nothing)
cv.createTrackbar(sh, barsWindow, 0, 255, nothing)
cv.createTrackbar(vl, barsWindow, 0, 255, nothing)
cv.createTrackbar(vh, barsWindow, 0, 255, nothing)

# set initial values for sliders
lb = (27, 26, 153)
ub = (60, 255, 255)
cv.setTrackbarPos(hl, barsWindow, lb[0])
cv.setTrackbarPos(hh, barsWindow, ub[0])
cv.setTrackbarPos(sl, barsWindow, lb[1])
cv.setTrackbarPos(sh, barsWindow, ub[1])
cv.setTrackbarPos(vl, barsWindow, lb[2])
cv.setTrackbarPos(vh, barsWindow, ub[2])

while(True):
    frame = cv.imread('wheresthered.png')
    frame = cv.GaussianBlur(frame, (7, 7), 0)
    frame = cv.fastNlMeansDenoisingColored(frame,None,h=10,hColor=10,templateWindowSize=3,searchWindowSize=11)
    frame = cv.GaussianBlur(frame, (7, 7), 0)
    # convert to HSV from BGR
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # read trackbar positions for all
    hul = cv.getTrackbarPos(hl, barsWindow)
    huh = cv.getTrackbarPos(hh, barsWindow)
    sal = cv.getTrackbarPos(sl, barsWindow)
    sah = cv.getTrackbarPos(sh, barsWindow)
    val = cv.getTrackbarPos(vl, barsWindow)
    vah = cv.getTrackbarPos(vh, barsWindow)

    # make array for final values
    HSVLOW = np.array([hul, sal, val])
    HSVHIGH = np.array([huh, sah, vah])

    # apply the range on a mask
    mask = cv.inRange(hsv, HSVLOW, HSVHIGH)
    maskedFrame = cv.bitwise_and(frame, frame, mask = mask)

    # display the camera and masked images
    cv.imshow('Masked', maskedFrame)
    cv.imshow('Camera', frame)

	# check for q to quit program with 5ms delay
    if cv.waitKey(5) & 0xFF == ord('q'):
        break

# clean up our resources
cap.release()
cv.destroyAllWindows()