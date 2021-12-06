import cv2
import numpy as np

## Filter frame for just the white lines at the bottom of the image
def white_lines(frame):
    gauss_frame = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(gauss_frame, cv2.COLOR_BGR2HSV)

    # hsv lower and upper bounds of white (works pretty well but needs more testing)
    lb = (0, 0, 128)
    ub = (179, 20, 256)

    hsv_frame = hsv_frame[300:]

    lines = cv2.inRange(hsv_frame, lb, ub)

    return lines

## Compute a starting y coordinate (assuming the centerline of the course is y = 0)
def get_starting_y(frame):
    # still need to figure out the maths on this

    frame_height = frame.shape[0]

    while frame_height >= 0:
        frame_height -= 1


    partition_array = np.argpartition(frame[120])
    print("Partition array: ", partition_array)
    return 0

## quick test script. After windows stop displaying, y-coordinate is calculated and printed in terminal
if __name__ == '__main__':
    frame = cv2.imread("wheresthered.png")
    
    cv2.imshow('frame', frame)

    while(True):
        white_line_photo = white_lines(frame)

        # cv2.imshow('frame', frame)
        cv2.imshow('white lines',white_line_photo)

        # Hit p to save white lines.
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite("g:/My Drive/3Junior/Fall/MAE 345/mae345_final_project/finding_red/white_lines.png", white_line_photo)

        # Hit q to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Filtered shape: ", white_line_photo.shape)

    print("Y-coordinate estimate: ", get_starting_y(white_line_photo))

    cv2.destroyAllWindows()
