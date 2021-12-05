import cv2


frame = cv2.imread('wheresthered.png')
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

while(True):
    cv2.imshow('frame',frame)

    # Compute
    # lb = (145, 3, 250)
    # ub = (155, 40, 255)


    # use rgb
    # smoothing filter to rm noise
    # lose hula hoop so don't filter too hard?

    # HSV ... values still need to be tuned
    llb = (0,3,0)
    ulb = (10,40,255)
    lb = (140, 3, 0)
    ub = (180, 255, 255)
    lowreds = cv2.inRange(hsv_frame, llb, ulb)
    highreds = cv2.inRange(hsv_frame, lb, ub)
    res = cv2.bitwise_or(lowreds, highreds)

    # BGR - doesn't work as red aint only red and red is in EVERYTHING
        # b, g, r
    # lb = (70,50,0)
    # ub = (75,70,255)
    # res = cv2.inRange(frame, lb, ub)
    cv2.imshow('highreds', highreds)
    cv2.imshow('lowreds', lowreds)
    cv2.imshow('res',res)

    # Hit q to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


# contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# def findGreatesContour(contours):
#     largest_area = 0
#     largest_contour_index = -1
#     i = 0
#     total_contours = len(contours)

#     while i < total_contours:
#         area = cv2.contourArea(contours[i])
#         if area > largest_area:
#             largest_area = area
#             largest_contour_index = i
#         i += 1

#     #print(largest_area)

#     return largest_area, largest_contour_index
# largest_area, largest_contour_index = findGreatesContour(contours)

# print(largest_area)