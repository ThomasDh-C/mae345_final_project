
def blue_filter(frame):
    """Turns camera frame into bool array of dark blue objects
    """
    blurred = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #      h    s    v
    lb =  (102, 3,   0)
    ub =  (131, 255, 255)

    res = cv2.inRange(hsv_frame, ub, lb)

    return res


def collision_time(frame1, frame2):
    """Given a contour finds time to collision""" 
    firstIm = frame1
    secondIm = frame2
    # other option: calcOpticalFlowPyrLK()
    # cv2.check_contours(frame1)
    # flow = cv2.calcOpticalFlowFarneback(firstIm, secondIm, None, 0.5, 4, 15, 4, 7, 1.5, 0)
    # firstIm = secondIm