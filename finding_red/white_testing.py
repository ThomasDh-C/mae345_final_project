
import cv2

def white_filter(frame):
    """Filter frame for just the white lines at the bottom of the image
    """
    gauss_frame = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(gauss_frame, cv2.COLOR_BGR2HSV)

    # hsv lower and upper bounds of white (works pretty well but needs more testing)
    lb = (0, 0, 128)
    ub = (179, 20, 256)

    white = cv2.inRange(hsv_frame, lb, ub)
    return white


def green_filter(frame):
    """Filter frame for just the white lines at the bottom of the image
    """
    frame = cv2.GaussianBlur(frame,(7,7),0)
    frame = cv2.fastNlMeansDenoisingColored(frame,None,h=10,hColor=10,templateWindowSize=3,searchWindowSize=11)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv lower and upper bounds of white (works pretty well but needs more testing)
    lb = (27, 26, 153)
    ub = (60, 255, 255)

    green = cv2.inRange(hsv_frame, lb, ub)
    return green

frame = cv2.imread('wheresthered.png')
res = green_filter(frame)
x,y,w,h = cv2.boundingRect(res)

while(True):

    cv2.imshow('frame',green_filter(frame))  



    # Hit q to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()