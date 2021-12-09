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

TABLE_HEIGHT_METERS = 0.72
SUDDEN_JUMP_METERS = 0.4

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

def z_estimate(scf, log_config):
    """Get the z coordinate estimate
    """
    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            z = data['stateEstimate.z']
            break
    
    print("Z estimate: ", z)
    return z


def set_pid_controller(cf):
    """Reset PID controller and Kalman position estimate
    """
    print('Initializing PID Controller')
    cf.param.set_value('stabilizer.controller', '1')
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)


def move_to_setpoint(scf, start, end, v, log_config):

    cf = scf.cf
    # move along x,y
    z_start, z_end = start[2], end[2]
    end[2] = start[2]
    steps = 30
    grad = np.array(end) - np.array(start)
    dist = np.linalg.norm(grad)
    step = grad/steps
    t = dist/v/steps

    for step_idx in range(1,steps+1):
        temp_pos = np.array(start) + step*step_idx
        cf.commander.send_position_setpoint(temp_pos[0], temp_pos[1], temp_pos[2], 0)

        start_time = time.time()
        while time.time() < start_time + t:
            time.sleep(t/3)
            temp_unused = z_estimate(scf, log_config)
            # continue
            # z_est = z_estimate(scf)
            # if z_start - z_est > SUDDEN_JUMP_METERS:
            #         print("Got to the table!")
            #         start[2] -= TABLE_HEIGHT_METERS
            #         end[2] -= TABLE_HEIGHT_METERS
            #         z_start -= TABLE_HEIGHT_METERS
            #         z_end -= TABLE_HEIGHT_METERS
            #         step[2] -= TABLE_HEIGHT_METERS
            #         cf.commander.send_position_setpoint(temp_pos[0], temp_pos[1], temp_pos[2], 0)
            #         break

    print("in move_to_setpoint: done translating")
    time.sleep(0.2)
    print("in move_to_setpoint: about to move up")

    # move along z
    steps = 10
    t = (z_end-z_start) / v / steps
    for idx in range(1, steps+1):
        cf.commander.send_hover_setpoint(0, 0, 0, z_start + (idx/steps)*(z_end-z_start))
        time.sleep(t)
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, z_end)
        time.sleep(0.1)

    end[2] = z_end
    print("in move_to_setpoint: returning")
    return end
    
def takeoff(cf, height):
    # Ascend:
    for y in range(5):
        cf.commander.send_hover_setpoint(0, 0, 0, y/5 * height)
        time.sleep(0.1)
    # Hover at 0.5 meters:
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, height)
        time.sleep(0.1)

    return [0, 0, height]


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

def relative_move(scf, start, dx, v, log_config):
    end = [start[0]+dx[0], start[1]+dx[1], start[2]+dx[2]]
    return move_to_setpoint(scf, start, end, v, log_config)



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
        
def farthestObstacle(frame1, frame2):
    """Given a contour finds time to collision"""
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 4, 15, 4, 7, 1.5, 0)
    frame2 = frame1
    red = cv2.red_filter(frame1)
    cont = cv2.findContours(red1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    newFlow = []
    contours = check_contours(red)
    for i in range len(contours):
        for j in range len(contours[0]):
            if contours[i][j] == 1:
               newFlow[i][j] = np.linalg.norm(flow[i][j])
            else:
                newFlow[i][j] = 0
     sum = 0
     count = 0
     avgFlows = []
     for contour in len(cont):
         for i in range len(newFlow):
            for j in range len(newFlow[0]):
                if (newFlow[i][j]=!0 && (newFlow[i][j+1]|newFlow[i+1][j]|newFlow[i-1][j]|newFlow[i][j-1]!=0))
                    sum = newFlow[i][j] + sum
                    count = count + 1
         avgFlows[contour] = sum/count  
      farthest = avgFlows.index(min(avgFlows))

    # Return direction vector
      
 
    

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

def detect_book(model, frame, confidence, COLORS, class_names):
    """Detect all books in the frame"""
    image = frame

    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                             swapRB=True)
   
    # forward propagate image
    model.setInput(blob)
    detections = model.forward()
    image_height, image_width, _ = image.shape
    
    # select detections that match selected class label
    for detection in detections[0, 0, :, :]:
        if detection[2] > confidence:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            class_name = class_names[int(class_id)-1]
            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            # draw a rectangle around each detected object
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=1)
            # put the FPS text on top of the frame
            text = class_name + ' ' + '%.2f' % (detection[2])
            cv2.putText(image, text, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)

    cv2.imshow('image', image)

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

def key_press(key, cf, cap, curr):
    """Aysnc key press handler
    """
    print("Key press detected: ", key)
    # quitting
    if key=='q':
        land(cf, curr)
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
