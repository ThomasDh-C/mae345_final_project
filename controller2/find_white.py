import cv2
import numpy as np
from numpy.testing._private.utils import HAS_LAPACK64

def white_lines(frame):
    """Filter frame for just the white lines at the bottom of the image
    """
    gauss_frame = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(gauss_frame, cv2.COLOR_BGR2HSV)

    # hsv lower and upper bounds of white (works pretty well but needs more testing)
    lb = (0, 0, 128)
    ub = (179, 20, 256)

    hsv_frame = hsv_frame[300:]

    lines = cv2.inRange(hsv_frame, lb, ub)

    return lines

def get_starting_y(frame):
    """Compute a starting y coordinate (assuming the centerline of the course is y = 0)
    """
    # still need to figure out the maths on this

    frame = white_lines(frame)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # this probably can change at some point
    row_to_check = 3*frame_height//4

    left_half = frame[:, 0:frame_width//2]
    right_half = frame[:, frame_width//2:frame_width]

    # print("left half shape: ", left_half.shape)
    # print("right half shape: ", right_half.shape)


    ## test code
    # while True:
    #     cv2.imshow('left half', left_half)
    #     cv2.imshow('right half', right_half)

    #     if cv2.waitKey(1) & 0xFF == ord('p'):
    #         cv2.imwrite("left_half.png", left_half)
    #         cv2.imwrite("right_half.png", right_half)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # get the average pixel of the white lines along row_to_check
    left_champ = np.average(np.argmax(left_half[row_to_check]))
    right_champ = np.average(np.argmax(right_half[row_to_check])) + frame_width//2


    # number of pixels off of center line the center of the course is
    center_line = (right_champ + left_champ) / 2
    center_line = -(center_line - frame_width//2)

    white_line_distance = right_champ - left_champ

    px_to_meters = 1.32 / white_line_distance

    ## test code
    # print("left champ: ", left_champ)
    # print("right champ: ", right_champ)
    # print("center line: ", center_line)

    # MATHS: factors to consider in this calculation:
    #   Height at which picture is taken (easy to know)
    #   Which row of the picture considering (adjustable)
    #   Distortion of the camera (very hard, want to try to approximate?)
    # Haven't actually done the measurements yet, just a janky approx
    return center_line * px_to_meters

if __name__ == '__main__':
    """quick test script. After windows stop displaying, y-coordinate is calculated and printed in terminal
    """
    frame = cv2.imread("wheresthered.png")
    
    cv2.imshow('frame', frame)

    while(True):
        white_line_photo = white_lines(frame)

        # cv2.imshow('frame', frame)
        cv2.imshow('white lines',white_line_photo)

        # Hit p to save white lines.
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite("white_lines.png", white_line_photo)

        # Hit q to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Filtered shape: ", white_line_photo.shape)

    # get_starting_y(white_line_photo)

    print("Y-coordinate estimate: ", get_starting_y(white_line_photo))

    cv2.destroyAllWindows()
