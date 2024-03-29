import cv2
# Code adapted from: https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/autonomousSequence.py

import time
# CrazyFlie imports:
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.position_hl_commander import PositionHlCommander
import numpy as np

## Some helper functions:
## -----------------------------------------------------------------------------------------
def position_estimate(scf):
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
            
    print(x, y, z)
    return x, y, z

# Ascend and hover:
def set_PID_controller(cf):
    # Set the PID Controller:
    print('Initializing PID Controller')
    cf.param.set_value('stabilizer.controller', '1')
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)
    return

# Ascend and hover:
def ascend_and_hover(cf):
    # Ascend:
    for y in range(5):
        cf.commander.send_hover_setpoint(0, 0, 0, y / 10)
        time.sleep(0.1)
    # Hover at 0.5 meters:
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, 0.25)
        time.sleep(0.1)
    return

def findGreatestContour(contours):
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

    #print(largest_area)

    return largest_area, largest_contour_index

def red_filter(frame):
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
    print('Checking image:')
    
    ################
    # cv2.imwrite('wheresthered.png', frame)
    ################

    # These define the upper and lower HSV for the red obstacles
    # May change on different drones.
    input_frame = red_filter(frame)

    # Do the contour detection on the input frame
    contours, hierarchy = cv2.findContours(input_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_area, largest_contour_index = findGreatestContour(contours)

    # print(largest_area)

    if largest_area > 100:
        return True
    else:
        return False


# Follow the setpoint sequence trajectory:
def adjust_position(cf, current_x, current_y):

    print('Adjusting position')
    dist_to_table = 2.59 # in meters
    dist_to_land = 0.559

    # translate along course
    num_steps = 40
    step_size = dist_to_table / num_steps
    for temp_x in np.arange(current_x, current_x+dist_to_table, step_size):
        cf.commander.send_position_setpoint(temp_x, current_y, 0.25, 0)
        time.sleep(0.2)
    current_x = current_x+dist_to_table
    cf.commander.send_hover_setpoint(current_x, current_y, 0.25, 0)
    time.sleep(3)

    # translate up
    for temp_z in np.linspace(.25, 1, 10):
        cf.commander.send_position_setpoint(current_x, current_y, temp_z, 0)
        time.sleep(.3)
    time.sleep(3)
    # translate forwards
    for temp_x in np.linspace(current_x, current_x+dist_to_land, 10):
        cf.commander.send_position_setpoint(current_x, current_y, 1.0, 0)
        time.sleep(0.2)
    current_x = current_x + dist_to_land
    time.sleep(3)

    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)
    return current_x, current_y


# Hover, descend, and stop all motion:
def hover_and_descend(cf):
    print('Descending:')
    # Hover at 0.5 meters:
    for _ in range(30):
        cf.commander.send_hover_setpoint(0, 0, 0, 1.0)
        time.sleep(0.1)
    # Descend:
    for y in range(10):
        cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
        time.sleep(0.1)
    # Stop all motion:
    for _ in range(10):
        cf.commander.send_stop_setpoint()
        time.sleep(0.1)
    return

def closest_detection(detections):
    if len(detections) == 0:
        return None
        
    champ = (float('infinity'), float('infinity'))
    champ_detection = None
    for detection in detections:
        # ( _, class_id, confidence, box_x, box_y, box_width, box_height)
        detect_vec = detection_center(detection)
        dist_to_detect = norm(detect_vec)
        if norm(detect_vec)< norm(champ):
            champ_detection = detection
            champ = detect_vec
    return champ_detection

def detect_book(frame):
    image = frame
    tracking_label = 84
    confidence = 0.2
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

# def move_to_book(cf, box_x, box_y, box_width, box_height, x_cur, y_cur):

    
def show_video_feed(frame):
    cv2.imshow('frame', red_filter(frame))
                
    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite('original_frame.png', frame)
        cv2.imwrite('red_filtered_frame.png', red_filter(frame))


group_number = 12

# Possibly try 0, 1, 2 ...
camera = 0

# Set the URI the Crazyflie will connect to
uri = f'radio://0/{group_number}/2M'

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
else:
    ## Ascent to hover; run the sequence; then descend from hover:
    # Use the CrazyFlie corresponding to team number:
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        # Get the Crazyflie class instance:
        cf = scf.cf
        cap = cv2.VideoCapture(camera)
    
        # Initialize and ascend:
        while not cap.isOpened():
            time.sleep(.1)
        start_t = time.time()
        current_x, current_y = 0.0, 0.0

       
        set_PID_controller(cf)
        ascend_and_hover(cf)
        current_x, current_y = adjust_position(cf, current_x, current_y)
                       
        # Descend and stop all motion:
        hover_and_descend(cf)
        
print("we made it everybody")
# Release the capture
cap.release()
cv2.destroyAllWindows()