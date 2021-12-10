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
SUDDEN_JUMP_METERS = 0.15
MIN_CONTOUR_SIZE = 300.0

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

def time_averaged_frame(cap):
    """smooth the picture over three frames
    """
    successful_frames = 0
    to_avg = np.ndarray((3, 480, 640, 3), dtype=np.uint32)
    while successful_frames != 3:
        ret, frame = cap.read()
        if ret:
            to_avg[successful_frames] = frame.astype(np.uint32)
            successful_frames += 1
    return True, np.mean(to_avg, axis=0).astype(np.uint8)

def set_pid_controller(cf):
    """Reset PID controller and Kalman position estimate
    """
    print('Initializing PID Controller')
    cf.param.set_value('stabilizer.controller', '1')
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def log_config_setup():
    log_config = LogConfig(name='Position Estimate', period_in_ms=100)
    # log_config.add_variable('stateEstimate.x', 'float')
    # log_config.add_variable('stateEstimate.y', 'float')
    log_config.add_variable('stateEstimate.z', 'float')
    print('log config setup')
    return log_config

# TODO: remove this function
def z_estimate(scf):
    """Get the z coordinate estimate
    """
    # print('Getting z estimate')
    log_config2 = LogConfig(name='Kalman Variance', period_in_ms=100)
    log_config2.add_variable('stateEstimate.z', 'float')
    with SyncLogger(scf, log_config2) as logger:
        for log_entry in logger:
            data = log_entry[1]
            # x = data['stateEstimate.x']
            # y = data['stateEstimate.y']
            z = data['stateEstimate.z']
            break
    
    # print("Z estimate: ", z)
    return z

def pos_estimate(scf):
    """Get the x coordinate estimate
    """
    log_config2 = LogConfig(name='Kalman Variance', period_in_ms=100)
    log_config2.add_variable('stateEstimate.x', 'float')
    log_config2.add_variable('stateEstimate.y', 'float')
    log_config2.add_variable('stateEstimate.z', 'float')

    with SyncLogger(scf, log_config2) as logger:
        for log_entry in logger:
            data = log_entry[1]
            x = data['stateEstimate.x']
            y = data['stateEstimate.y']
            z = data['stateEstimate.z']
            break
    
    return [x, y, z]

def angle_estimate(scf):
    """Get the x coordinate estimate
    """
    log_config2 = LogConfig(name='Kalman Variance', period_in_ms=100)
    log_config2.add_variable('stateEstimate.yaw', 'float')

    with SyncLogger(scf, log_config2) as logger:
        for log_entry in logger:
            data = log_entry[1]
            angle = data['stateEstimate.yaw']
            break
    
    return angle


def move_to_setpoint(scf, start, end, v, big_move):
    cf = scf.cf
    
    # move along x,y
    z_start, z_end = start[2], end[2] # we update both these to - table height if we go over a table
    steps = 30
    xy_grad = np.array(end[0:2]) - np.array(start[0:2])
    xy_dist = np.linalg.norm(xy_grad)
    xy_step, t = xy_grad/steps, xy_dist/v/steps
    z_measured_start, table = z_estimate(scf), False
    # print('about to start xy move')
    for step_idx in range(1,steps+1):
        temp_pos = [start[0]+xy_step[0]*step_idx, start[1]+xy_step[1]*step_idx, z_start]
        cf.commander.send_position_setpoint(temp_pos[0], temp_pos[1], temp_pos[2], 0)
        time.sleep(t)
    time.sleep(0.2)
    # print('about to move in z')
    # move along z
    steps = 10
    if z_start != z_end:
        t = (z_end-z_start) / v / steps
        for idx in range(1, steps+1):
            cf.commander.send_hover_setpoint(0, 0, 0, z_start + (idx/steps)*(z_end-z_start))
            time.sleep(np.linalg.norm(t))
    # (10,20)[big_move]
    for _ in range(10):
        cf.commander.send_hover_setpoint(0, 0, 0, z_end)
        time.sleep(0.1)

    # end[2] = z_end # if go over table have to update this
    # print("in move_to_setpoint: returning")
    return pos_estimate(scf)

def relative_move(scf, start, dx, v, big_move):
    end = [start[0]+dx[0], start[1]+dx[1], start[2]+dx[2]]
    return move_to_setpoint(scf, start, end, v, big_move)

# def look_left(cf, start):
#     cf.commander.send_position_setpoint(start[0], start[1], start[2], 90)

# def look_right(cf, start):
#     cf.commander.send_position_setpoint(start[0], start[1], start[2], -90)

# def look_center(cf, start):
#     cf.commander.send_position_setpoint(start[0], start[1], start[2], 0)

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


def red_filter(frame):
    """Turns camera frame into bool array of red objects
    """
    # blurred_input = cv2.GaussianBlur(frame,(7,7),0)
    # denoised_blurred = cv2.fastNlMeansDenoisingColored(blurred_input,None,10,10,7,21)
    blurred = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #      h    s    v
    llb = (0,   3,   0)
    ulb = (28,  255, 255)
    lb =  (120, 3,   0)
    ub =  (180, 255, 255)

    lowreds = cv2.inRange(hsv_frame, llb, ulb)
    highreds = cv2.inRange(hsv_frame, lb, ub)
    res = cv2.bitwise_or(lowreds, highreds)


    return res

def too_close(large_contours):
    """Return a list of contours that are too close to Dori
    """
    # A contour is too close if its area is too large AND it stretches from the top to the bottom of the frame.
    # If a large contour doesn't go from top to bottom, it is almost certainly just a cluster of far-away pipes
    to_ret = []
    for c in large_contours:
        _, _, _, h = cv2.boundingRect(c)
        # choose some large (but not = 480px) determiner
        # TODO: tune this value
        if h >= 460:
            to_ret.append(c)
    return to_ret

def center_vertical_obs_bottom(red_frame, CLEAR_CENTER):
    """Return the number of pixels to the bottom of the frame that the closest contour occupies
    """
    # crop into frame, compute contours, get bounding box around large contours, return
    current_frame_height = red_frame.shape[0]
    red1 = red_frame[current_frame_height // 2:, :]
    contours, _ = cv2.findContours(red_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    large_contours = [cont for cont in contours if cv2.contourArea(cont) > MIN_CONTOUR_SIZE]
    bottom_y = []
    for cont in large_contours:
        x,y,w,h = cv2.boundingRect(cont)
        c_x = (x+w/2)-640/2                     # from center bottom
        if np.linalg.norm(c_x) < CLEAR_CENTER:
            bottom_y.append(480-(y+h))
    if len(bottom_y) == 0:
        return 480 # max height of frame
    return min(bottom_y)

def rotate_to(scf, curr, current_angle, new_angle):
    # send_hover_setpoint(self, vx, vy, yawrate, zdistance)
    # pos_neg = new_angle-current_angle
    # while angle_estimate(scf)
    cf = scf.cf
    for i in np.linspace(current_angle, new_angle, 100):
        cf.commander.send_position_setpoint(curr[0], curr[1], curr[2], i)
        time.sleep(0.02)
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
        time.sleep(0.1)
    return angle_estimate(scf)

def furthest_obstacle(frame1, frame2, dx):
    """Given a contour returns a next step that should be taken"""
    # ^^ clarification q from jacob: not actually given contours, but frames right?

    # find flow = only accepts 1 channel images
    # based on https://github.com/ferreirafabio/video2tfrecord/blob/master/video2tfrecord.py
    grey_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grey_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev=grey_frame1, next=grey_frame2, flow=None, pyr_scale=0.5, levels=4, winsize=15, iterations=4, poly_n=7, poly_sigma=1.5, flags=0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # mag matrix shape is (480, 640)

    # find contours
    red1 = red_filter(frame1) # shape is also (480, 640)
    contours, _ = cv2.findContours(red1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    large_contours = [cont for cont in contours if cv2.contourArea(cont) > MIN_CONTOUR_SIZE]
    n_contours = len(large_contours)

    # find flow of each contour
    av_flows = []
    mask = np.zeros(red1.shape,np.uint8)
    for idx in range(n_contours):
        # put 1s in every cell
        region = cv2.drawContours(mask, large_contours,  idx, 1, -1) # template, contours, index, num_to_put_in, thickness
        masked_flow_region = cv2.bitwise_and(mag, mag, mask=region) # mask of region
        av_flow = np.sum(masked_flow_region) / np.sum(region)
        av_flows.append(av_flow)

    # find centre of min flow
    min_flow_index = np.argmin(av_flows) # farthest obj has highest flow
    farthest_contour = cv2.drawContours(mask, large_contours, min_flow_index, 1, -1) # template, contours, index, num_to_put_in, thickness
    x,y,w,h = cv2.boundingRect(farthest_contour) #top left corner
    c_x, c_y = (x+w/2)-640/2, 480-(y+h/2) # from centre bottom
    # angle_from_vertical = 90-np.arctan2(c_y,c_x)*180/np.pi # fun for debugging

    step_multiplier = dx/np.linalg.norm([c_x,c_y])
    return [c_x*step_multiplier, c_y*step_multiplier, 0]


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

def find_book(frame):
    return

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
    
    # TODO: no longer need to move forward, so just land
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
    ret, frame = time_averaged_frame(cap)
    if ret and key=='t':
        cv2.imshow('frame', red_filter(frame))
    if ret and key=='p':
        # save photo .. do anything
        print('Photo taken')
        cv2.imwrite('original_frame.png', frame)
        cv2.imwrite('red_filtered_frame.png', red_filter(frame))
    
    # listener ends loop if returns False    
    return True
