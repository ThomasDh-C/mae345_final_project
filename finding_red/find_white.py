import cv2

## Filter frame for just the white lines at the bottom of the image
def white_lines(frame):
    #      h s v
    # llb = (0,2,0)
    # ulb = (25,255,255)
    # lb = (120, 3, 0)
    # ub = (180, 255, 255)

    # lowreds = cv2.inRange(hsv_frame, llb, ulb)
    # highreds = cv2.inRange(hsv_frame, lb, ub)
    # res = cv2.bitwise_or(lowreds, highreds)


    gauss_frame = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(gauss_frame, cv2.COLOR_BGR2HSV)
    
    # print(hsv_frame.shape)

    # hsv lower and upper bounds of white (in theory)
    lb = (0, 0, 128)
    ub = (179, 20, 256)

    hsv_frame = hsv_frame[300:]

    lines = cv2.inRange(hsv_frame, lb, ub)

    return lines

## Compute a starting y coordinate (assuming the centerline of the course is y = 0)
def get_starting_y(frame):
    # still need to figure out the maths on this

    return 0

## quick test script. After windows stop displaying, y-coordinate is calculated and printed in terminal
if __name__ == '__main__':
    frame = cv2.imread("g:/My Drive/3Junior/Fall/MAE 345/mae345_final_project/finding_red/wheresthered_problematic.png")
    
    while(True):
        # blurred = cv2.GaussianBlur(frame,(7,7),0)
        # # dst = cv2.fastNlMeansDenoisingColored(frame,None,7,7,7,21) 
        # cv2.imshow('frame',frame)  
        # cv2.imshow('blurredimg',blurred)
        # # cv2.imshow('denoised',dst) 


        # hsv_frame_gauss = red_filter(blurred)
        # # hsv_frame_denoise = red_filter(dst)
        # orig_res = red_filter(frame)

        white_line_photo = white_lines(frame)

        cv2.imshow('frame', frame)
        cv2.imshow('white lines',white_line_photo)
        # cv2.imshow('hsv_frame_denoise', hsv_frame_denoise)
        # cv2.imshow('orig_res',orig_res)


        # Hit q to quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Y-coordinate estimate: ", get_starting_y(white_line_photo))

    cv2.destroyAllWindows()
