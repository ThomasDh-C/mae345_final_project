import cv2

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


frame = cv2.imread('wheresthered_og.png')
frame2 = cv2.GaussianBlur(frame,(7,7),0)

frame3 = cv2.fastNlMeansDenoisingColored(frame2,None,10,10,7,21)
h = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)[:,:,0]
while(True):
    cv2.imshow('frame',frame)  
    cv2.imshow('h',frame3)
    cv2.imshow('red_filter', red_filter(frame3))

    # Hit q to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()