import cv2
import time

def red_filter(frame):
    """Turns camera frame into bool array of red objects
    """
    blurred = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #      h    s    v
    llb = (0,   3,   0)
    ulb = (28,  255, 255)
    lb =  (120, 3,   0)
    ub =  (180, 255, 255)

    lowreds = cv2.inRange(hsv_frame, llb, ulb)
    highreds = cv2.inRange(hsv_frame, lb, ub)
    res = cv2.bitwise_or(lowreds, highreds)

    return res

start = time.time()
frame = cv2.imread('wheresthered_og.png')
frame2 = cv2.GaussianBlur(frame,(7,7),0)

frame3 = cv2.fastNlMeansDenoisingColored(frame2,None,h=10,hColor=10,templateWindowSize=3,searchWindowSize=11)
# h = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)[:,:,0]
res = red_filter(frame3)

while(True):
    cv2.imshow('frame',frame)  
    cv2.imshow('h',frame3)
    cv2.imshow('red_filter', res)

    # Hit q to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()