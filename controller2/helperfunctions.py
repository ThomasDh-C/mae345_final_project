import cv2
# Code adapted from: https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/autonomousSequence.py

import time
# CrazyFlie imports:
import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
import numpy as np

## Some helper functions:
## -----------------------------------------------------------------------------------------

def check_crazyflie_available():
    """Inits crazyflie drivers, finds local crazflies, 
    prints names and returns True if they exist
    """
    # Initialize all the CrazyFlie drivers:
    cflib.crtp.init_drivers(enable_debug_driver=False)

    # Scan for Crazyflies in range of the antenna:
    print('Scanning interfaces for Crazyflies...')
    available = cflib.crtp.scan_interfaces()

    # List local CrazyFlie devices:
    print('Crazyflies found:')
    for i in available:
        print(i[0])

    # Check that CrazyFlie devices are available:
    if len(available) == 0:
        print('No Crazyflies found, cannot run example')
        return False
    else: 
        return True

def start_video(camera_number):
    """Returns camera stream and removes junk inital frames 
    """
    cap = cv2.VideoCapture(camera_number)
    
    # Wait until video has started
    while not cap.isOpened():
        time.sleep(.1)

    # Make sure no junk frames
    time.sleep(3)
    return cap

def position_estimate(scf):
    """Kalman estimate of x, y, z position
    """
    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            x = data['kalman.varPX']
            y = data['kalman.varPY']
            z = data['kalman.varPZ']
            
    return x, y, z

def set_pid_controller(cf):
    """Reset PID controller and Kalman position estimate
    """
    print('Initializing PID Controller')
    cf.param.set_value('stabilizer.controller', '1')
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def move_to_setpoint(cf, start, end, v):
    steps = 30
    grad = np.array(start) - np.array(end)
    dist = np.linalg.norm(grad)
    step = grad/steps
    t = dist/v/steps
    
    for step_idx in range(1,steps+1):
        temp_pos = np.array(start) + step*step_idx
        cf.commander.send_position_setpoint(temp_pos[0], temp_pos[1], temp_pos[2], 0)
        time.sleep(t)
    # cf.commander.send_hover_setpoint(end[0], end[1], end[2],0)
    time.sleep(2)

    return end

def land(cf, curr):
    z = curr[2]
    for _ in range(30):
        cf.commander.send_hover_setpoint(0, 0, 0,z)
        time.sleep(0.1)
    # Descend:
    for pos in np.linspace(z,0.0,10):
        cf.commander.send_hover_setpoint(0, 0, 0, pos)
        time.sleep(0.1)
    # Stop all motion:
    for _ in range(10):
        cf.commander.send_stop_setpoint()
        time.sleep(0.1)

def relative_move(cf, start, dx, v):
    end = [start[0]+dx[0], start[1]+dx[1], start[2]+dx[2]]
    return move_to_setpoint(cf, start, end, v)



def find_greatest_contour(contours):
    """Finds greatest contour len=2 tuple of: its area, its index in contours list
    """
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)

    while i < total_contours:
        area = cv2.contourArea(contours[i])
        if area > largest_area:
            largest_area = area
            largest_contour_index = i
        i += 1

    return largest_area, largest_contour_index

def red_filter(frame):
    """Turns camera frame into bool array of red objects
    """
    blurred = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #      h    s    v
    llb = (0,   3,   0)
    ulb = (25,  255, 255)
    lb =  (120, 3,   0)
    ub =  (180, 255, 255)

    lowreds = cv2.inRange(hsv_frame, llb, ulb)
    highreds = cv2.inRange(hsv_frame, lb, ub)
    res = cv2.bitwise_or(lowreds, highreds)

    return res

def check_contours(frame):
    """Checks size of contours in frame. True if largest over size 100.
    """
    print('Checking image:')

    # These define the upper and lower HSV for the red obstacles
    # May change on different drones.
    input_frame = red_filter(frame)

    # Do the contour detection on the input frame
    contours, hierarchy = cv2.findContours(input_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_area, largest_contour_index = find_greatest_contour(contours)

    # print(largest_area)

    if largest_area > 100:
        return True
    else:
        return False

def detection_center(detection):
    """Computes the center x, y coordinates of the book"""
    center_x = (detection[3] + detection[5]) / 2.0 - 0.5
    center_y = (detection[4] + detection[6]) / 2.0 - 0.5
    return (center_x, center_y)

def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0]**2 + vec[1]**2)

def closest_detection(detections):
    """Determines closest detected book"""
    if len(detections) == 0:
        return None
        
    champ = (float('infinity'), float('infinity'))
    champ_detection = None
    for detection in detections:
        # ( _, class_id, confidence, box_x, box_y, box_width, box_height)
        detect_vec = detection_center(detection)
        if norm(detect_vec)< norm(champ):
            champ_detection = detection
            champ = detect_vec
    return champ_detection

def detect_book(model, frame, tracking_label, confidence):
    """Detect if there is a book in the frame"""
    image = frame
    image_height, image_width, _ = image.shape

    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                             swapRB=True)
   
    # forward propagate image
    model.setInput(blob)
    detections = model.forward()

    # select detections that match selected class label
    matching_detections = [d for d in detections[0, 0] if d[1] == tracking_label]

    # select confident detections
    confident_detections = [d for d in matching_detections if d[2] > confidence]

    # get detection closest to center of field of view and draw it
    det = closest_detection(confident_detections)

    return det

def move_to_book(cf, box_x, box_y, box_width, box_height, x_cur, y_cur):
    """Controller for moving to book once detected"""
    set_size = 0.82
    if box_width > set_size or box_height > set_size:
        return True, x_cur, y_cur

    x_command, y_command = x_cur, y_cur
    dx = .08
    ok_region=.1
    
    if box_x<-ok_region:
        y_command+=dx
    if box_x>ok_region:
        y_command-=dx
    
    # only once centered move forward
    if box_x > -ok_region and box_x < ok_region:
        x_command+=dx

    # Set position
    cf.commander.send_position_setpoint(x_command, y_command, 0.5, 0)
    return False, x_command, y_command

def key_press(key, cf_command, cap):
    """Aysnc key press handler
    """
    # quitting
    if key=='q':
        cf_command.land()
        cap.release()
        cv2.destroyAllWindows()
        return False

    # photo operations
    ret, frame = cap.read()
    if ret and key=='t':
        cv2.imshow('frame', red_filter(frame))
    if ret and key=='p':
        # save photo .. do anything
        print('Photo taken')
        cv2.imwrite('original_frame.png', frame)
        cv2.imwrite('red_filtered_frame.png', red_filter(frame))
    
    # listener ends loop if returns False    
    return True