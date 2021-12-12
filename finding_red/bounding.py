import cv2

# def red_filter(frame):
#     """Turns camera frame into bool array of red objects"""
#     blurred = cv2.GaussianBlur(frame,(7,7),0)
#     hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     #      h    s    v
#     llb = (0,   3,   0)
#     ulb = (28,  255, 255)
#     lb =  (120, 3,   0)
#     ub =  (180, 255, 255)

#     lowreds = cv2.inRange(hsv_frame, llb, ulb)
#     highreds = cv2.inRange(hsv_frame, lb, ub)
#     res = cv2.bitwise_or(lowreds, highreds)

#     return res

color = (255, 0, 0)
CLEAR_CENTER = 30
red_frame = cv2.imread('very_clear_frame.png')
lb, rb = 640/2-CLEAR_CENTER, 640/2+CLEAR_CENTER
cv2.rectangle(red_frame, (int(lb), 0), (int(rb), 480), color, thickness=2)
cv2.imshow('red', red_frame)
cv2.waitKey()