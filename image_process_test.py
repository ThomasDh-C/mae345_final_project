import cv2

def red_filter(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #      h s v
    llb = (0,3,0)
    ulb = (20,40,255)
    lb = (120, 3, 0)
    ub = (180, 255, 255)

    lowreds = cv2.inRange(hsv_frame, llb, ulb)
    highreds = cv2.inRange(hsv_frame, lb, ub)
    res = cv2.bitwise_or(lowreds, highreds)

    return res


frame = cv2.imread('wheresthered.png')
while(True):
    blurred = cv2.GaussianBlur(frame,(7,7),0)
    # dst = cv2.fastNlMeansDenoisingColored(frame,None,7,7,7,21) 
    cv2.imshow('frame',frame)  
    cv2.imshow('blurredimg',blurred)
    # cv2.imshow('denoised',dst) 


    hsv_frame_gauss = red_filter(blurred)
    # hsv_frame_denoise = red_filter(dst)
    orig_res = red_filter(frame)

    cv2.imshow('hsv_frame_gauss',hsv_frame_gauss)
    # cv2.imshow('hsv_frame_denoise', hsv_frame_denoise)
    # cv2.imshow('orig_res',orig_res)


    # Hit q to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()