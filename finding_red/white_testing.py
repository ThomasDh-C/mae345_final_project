
import cv2

def white_filter(frame):
    """Filter frame for just the white lines at the bottom of the image
    """
    gauss_frame = cv2.GaussianBlur(frame,(7,7),0)[240:, :, :]
    hsv_frame = cv2.cvtColor(gauss_frame, cv2.COLOR_BGR2HSV)

    # hsv lower and upper bounds of white (works pretty well but needs more testing)
    lb = (0, 0, 128)
    ub = (179, 20, 256)

    white = cv2.inRange(hsv_frame, lb, ub)
    return white


def green_filter(frame):
    """Filter frame for just the white lines at the bottom of the image
    """
    frame = cv2.GaussianBlur(frame, (7, 7), 0)[240:, :, :]
    frame = cv2.fastNlMeansDenoisingColored(frame,None,h=10,hColor=10,templateWindowSize=3,searchWindowSize=11)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv lower and upper bounds of white (works pretty well but needs more testing)
    lb = (29, 75, 85)
    ub = (60, 255, 255)


    green = cv2.inRange(hsv_frame, lb, ub)
    return green

MIN_CONTOUR_SIZE = 100.0 # TODO: tune

frame = cv2.imread('green_4.png')
res = green_filter(frame)

contours, _ = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
large_contours = [cont for cont in contours if cv2.contourArea(cont) > MIN_CONTOUR_SIZE]
bottom_y = []
num = 0
for cont in large_contours:
    x,y,w,h = cv2.boundingRect(cont)
    bottom_y.append(y)

y = min(bottom_y)
print("Y pixel:", y)

cv2.imshow('orig',frame) 

line = cv2.line(frame, (0, y+240), (640, y+240), color=(0, 0, 255), thickness=5)
cv2.imshow('lined',line) 
cv2.imshow('res', res)
# cv2.imshow('pasty', pasty)

cv2.waitKey()

cv2.destroyAllWindows()