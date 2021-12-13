import cv2
# Code adapted from: https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/autonomousSequence.py

import time
# CrazyFlie imports:
import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
import numpy as np
from numpy.lib.function_base import angle

## Some helper functions:
## -----------------------------------------------------------------------------------------

TABLE_HEIGHT_METERS = 0.72
SUDDEN_JUMP_METERS = 0.15
MIN_CONTOUR_SIZE = 100.0

def check_crazyflie_available():
    """Inits crazyflie drivers, finds local crazflies, 
    prints names and returns True if they exist"""
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
    """Returns camera stream and removes junk inital frames"""
    cap = cv2.VideoCapture(camera_number)
    
    # Wait until video has started
    while not cap.isOpened():
        time.sleep(.1)

    # Make sure no junk frames
    time.sleep(3)
    return cap

def time_averaged_frame(cap):
    """Average the picture over three frames"""
    successful_frames = 0
    to_avg = np.ndarray((3, 480, 640, 3), dtype=np.uint32)
    while successful_frames != 3:
        ret, frame = cap.read()
        if ret:
            to_avg[successful_frames] = frame.astype(np.uint32)
            successful_frames += 1
    return True, np.mean(to_avg, axis=0).astype(np.uint8)

def set_pid_controller(cf):
    """Reset PID controller and Kalman position estimate"""
    print('Initializing PID Controller')
    cf.param.set_value('stabilizer.controller', '1')
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def pos_estimate(scf):
    """Get the curr coordinate estimate"""
    log_config = LogConfig(name='Kalman Variance', period_in_ms=30)
    log_config.add_variable('stateEstimate.x', 'float')
    log_config.add_variable('stateEstimate.y', 'float')
    log_config.add_variable('stateEstimate.z', 'float')

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            x = data['stateEstimate.x']
            y = data['stateEstimate.y']
            z = data['stateEstimate.z']
            break
    
    return [x, y, z]

def angle_estimate(scf):
    """Get the angle coordinate estimate"""
    log_config = LogConfig(name='Kalman Variance', period_in_ms=20)
    log_config.add_variable('stateEstimate.yaw', 'float')

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            angle = data['stateEstimate.yaw']
            break
    
    return angle

def move_to_setpoint(scf, start, end, v, big_move):
    """Absolute x,y,z move at a given velocity"""
    cf = scf.cf
    
    # move along x,y
    z_start, z_end = start[2], end[2] # we update both these to - table height if we go over a table
    steps = 30
    xy_grad = np.array(end[0:2]) - np.array(start[0:2])
    xy_dist = np.linalg.norm(xy_grad)
    xy_step, t = xy_grad/steps, xy_dist/v/steps
    for step_idx in range(1,steps+1):
        temp_pos = [start[0]+xy_step[0]*step_idx, start[1]+xy_step[1]*step_idx, z_start]
        cf.commander.send_position_setpoint(temp_pos[0], temp_pos[1], temp_pos[2], 0)
        time.sleep(t)
    time.sleep(0.2)

    # move along z
    steps = 10
    if z_start != z_end:
        t = (z_end-z_start) / v / steps
        for idx in range(1, steps+1):
            cf.commander.send_hover_setpoint(0, 0, 0, z_start + (idx/steps)*(z_end-z_start))
            time.sleep(np.linalg.norm(t))
    
    for _ in range(10):
        cf.commander.send_hover_setpoint(0, 0, 0, z_end)
        time.sleep(0.1)

    return pos_estimate(scf)

def relative_move(scf, start, dx, v, big_move):
    """Relative x,y,z move at a given velocity"""
    end = [start[0]+dx[0], start[1]+dx[1], start[2]+dx[2]]
    return move_to_setpoint(scf, start, end, v, big_move)
    
def left_right_slide_to_start_point(scf, start, end, v_first_slide, v_to_final, cap, CLEAR_CENTER, steps):
    cf, dist_to_obs_center = scf.cf, []

    # move along x,y
    z_start, z_end = start[2], end[2] # we update both these to - table height if we go over a table
    xy_grad = np.array(end[0:2]) - np.array(start[0:2])
    xy_dist = np.linalg.norm(xy_grad)
    xy_step, step_t = xy_grad/steps, xy_dist/v_first_slide/steps
    print('total_d', xy_dist)
    print('step size', np.linalg.norm(xy_step))
    print('xy_step - want neglig_x, neg_y', xy_step)
    start_time = time.time()
    for step_idx in range(1,steps+1):
        # make the move
        temp_pos = [start[0]+xy_step[0]*step_idx, start[1]+xy_step[1]*step_idx, z_start]
        cf.commander.send_position_setpoint(temp_pos[0], temp_pos[1], temp_pos[2], 0)

        # compensate for high speed :)
        curr_pos_time = time.time()
        curr = pos_estimate(scf)
        _, frame = time_averaged_frame(cap)
        red = red_filter(frame) # super accomodating
        dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER) # splits frame in two as discussed
        red_pos_time = time.time()
        curr_pos_step = (red_pos_time-curr_pos_time)*v_first_slide*xy_grad/xy_dist
        dist_to_obs_center.append((dist_center_obs, [curr[0]+curr_pos_step[0],curr[1]+curr_pos_step[1], curr[2]]))
        while time.time()<step_t*step_idx+start_time:
            continue
    
    for _ in range(10):
        cf.commander.send_hover_setpoint(0, 0, 0, z_end)
        time.sleep(0.1)

    print('Starting points I found')
    moving_averaged = moving_average([pos[0] for pos in dist_to_obs_center],7)
    moving_averaged = moving_average(moving_averaged, 3)
    moving_averaged = moving_average(moving_averaged, 3)
    moving_averaged = moving_average(moving_averaged, 3)
    print([int(num) for num in moving_averaged])

    # find_max_index
    max_dist_index = np.argmax(moving_averaged)
    max_dist_val, best_start_point = dist_to_obs_center[max_dist_index][0], dist_to_obs_center[max_dist_index][1]
    
    # if multiple cells with max dist +-2px choose center of largest cluster
    mask_max_dist = [abs(dist-max_dist_val) <= 2 for dist in moving_averaged]
    if sum(mask_max_dist) > 1:
        # list of cluster tuples called groups
        groups, prev = [], False
        for idx, val in enumerate(mask_max_dist):
            # start of new chain of Trues
            if val and not prev:
                groups.append([idx, idx])
            # in chain of trues
            elif val and prev:
                groups[-1][-1] = idx
            # do nothing for Falses
            prev = val

        # find biggest group
        max_group_idx = np.argmax([group[1]-group[0]+1 for group in groups])

        # set best_start_point to centre of group 
        l_idx, r_idx = groups[max_group_idx][0], groups[max_group_idx][1]
        max_group_centre_idx = l_idx + ((r_idx - l_idx) / 2)
        if max_group_centre_idx%1==0.5:
            zeroed_centre_idx = int(max_group_centre_idx)
            l_p, r_p = dist_to_obs_center[zeroed_centre_idx][1], dist_to_obs_center[zeroed_centre_idx+1][1]  
            best_start_point = [(l_p[0]+r_p[0])/2, (l_p[1]+r_p[1])/2, (l_p[2]+r_p[2])/2]
        else:
            best_start_point = dist_to_obs_center[int(max_group_centre_idx)][1]

    curr = pos_estimate(scf)
    return move_to_setpoint(scf, curr, best_start_point, v_to_final, True)

def take_off_slide_left(scf, curr, width, v_first_slide, v_to_final, cap, CLEAR_CENTER):
    """Slides drone to the left, then positions drone where it 
    can see the furthest without being blocked by red"""  
    start, end = curr, [curr[0], width, curr[2]]
    return left_right_slide_to_start_point(scf, start, end, v_first_slide, v_to_final, cap, CLEAR_CENTER, 60)

def forward_slide_to_obs(scf, curr, v, VERY_CLEAR_PX, max_x, CLEAR_CENTER, cap):
    """Slide forward till a red object is too close for a fast move"""
    cf = scf.cf

    dt = .15
    dx = v*dt
    start_time, c = time.time(), 0
    while pos_estimate(scf)[0] < max_x:
        curr[0]+=dx
        cf.commander.send_position_setpoint(curr[0], curr[1], curr[2], 0)
        c+=1

        # only stop sliding if get two positive detections
        _, frame = time_averaged_frame(cap)
        red = red_filter(frame) # super accomodating
        dist = center_vertical_obs_bottom(red, CLEAR_CENTER)
        if dist < VERY_CLEAR_PX: 
            for _ in range(10):
                cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
                time.sleep(0.1)
            _, frame = time_averaged_frame(cap)
            red = red_filter(frame) # super accomodating
            dist = center_vertical_obs_bottom(red, CLEAR_CENTER)
            if dist < VERY_CLEAR_PX: 
                cv2.imwrite(f'imgs/very_clear_frame_raw.png', frame)
                cv2.imwrite(f'imgs/very_clear_frame_filtered.png', red)
                break

        while time.time()<dt*c+start_time:
            continue
    
    # stabilise
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
        time.sleep(0.1)
    return pos_estimate(scf)

def left_right_slide_to_obs(scf, curr, v, VERY_CLEAR_PX, WIDTH, SAFETY_DISTANCE_TO_SIDE, CLEAR_CENTER, cap, right):
    """Slide forward till a red object is too close for a fast move"""
    cf = scf.cf

    dt = .15
    dy = v*dt*(1,-1)[right]
    start_time, c = time.time(), 0

    while (right and curr[1] > SAFETY_DISTANCE_TO_SIDE/4) or (not right and curr[1] < WIDTH-SAFETY_DISTANCE_TO_SIDE/4):
        curr[1]+=dy
        cf.commander.send_position_setpoint(curr[0], curr[1], curr[2], (1,-1)[right]*90)
        # print('position updated')
        c+=1

        # only stop sliding if get two positive detections
        _, frame = time_averaged_frame(cap)
        red = red_filter(frame) # super accomodating
        dist = center_vertical_obs_bottom(red, CLEAR_CENTER)
        # print('distance is', dist,'and checking against', VERY_CLEAR_PX)
        if dist < VERY_CLEAR_PX: 
            for _ in range(10):
                cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
                time.sleep(0.1)
            _, frame = time_averaged_frame(cap)
            red = red_filter(frame) # super accomodating
            dist = center_vertical_obs_bottom(red, CLEAR_CENTER)
            if dist < VERY_CLEAR_PX:
                print('Distance too short: ', dist)
                lr = ['left','right'][right]
                cv2.imwrite(f'imgs/{lr}_stop_point.png', red)
                break

        while time.time()<dt*c+start_time:
            continue
    
    # stabilise
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
        time.sleep(0.1)
    est = pos_estimate(scf)
    # distance = np.linalg.norm(est - np.array(curr))
    # print('difference that shouldnt exist', distance)
    return est


def slide_green(scf, curr, cap, v, GREEN_PX_TOP_BOT_IDEAL, GREEN_MARGIN, GREEN_DX):
    cf = scf.cf

    # Take first frame
    _, frame = time_averaged_frame(cap)
    green = green_filter(frame) # super accomodating
    # _, green_from_top, _, _ = cv2.boundingRect(green)
    green_from_top = px_green_from_top(green)

    # Set up moving average
    green_list, c = [green_from_top]*8, 0 # moving average, tuned from 10
    print(c, ' green from top', green_from_top)

    while abs(np.mean(green_list)-GREEN_PX_TOP_BOT_IDEAL) > GREEN_MARGIN or max(green_list)-min(green_list) > GREEN_MARGIN*2:
        c+=1
        
        # make move
        forwards = (-1, 1)[np.mean(green_list) < GREEN_PX_TOP_BOT_IDEAL]
        steps = 10
        x_dist = forwards*GREEN_DX
        x_step, t = x_dist/steps, abs(x_dist/v/steps)
        # if mean is too large then make move
        if abs(np.mean(green_list)-GREEN_PX_TOP_BOT_IDEAL) > GREEN_MARGIN:
            for _ in range(1,steps+1):
                curr[0]+=x_step
                cf.commander.send_position_setpoint(curr[0], curr[1], curr[2], 0)
                time.sleep(t)
        # fine tuning
        else:
            for _ in range(1,steps+1):
                cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
                time.sleep(0.05)

        _, frame = time_averaged_frame(cap)
        green = green_filter(frame) # super accomodating

        # cv2.imwrite(f'imgs/green_raw_{c}.png', frame)
        # cv2.imwrite(f'imgs/green_{c}.png', green)
        # x, green_from_top, w, h = cv2.boundingRect(green)
        green_from_top = px_green_from_top(green)
        green_list.append(green_from_top)
        green_list.pop(0)
        print(c, ' green from top', np.mean(green_list))
    
    # stabilise
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
        time.sleep(0.1)

    return pos_estimate(scf)

def takeoff(scf, height):
    cf = scf.cf
    """Drone rises from ground to set height"""
    # Ascend:
    for y in range(5):
        cf.commander.send_hover_setpoint(0, 0, 0, y/5 * height)
        time.sleep(0.1)
    # Hover at set height:
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, height)
        time.sleep(0.1)

    return pos_estimate(scf)

def land(cf, curr):
    """Drone falls slowly from current height to ground"""
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
    """Turns camera frame into bool array of red objects"""
    blurred = cv2.GaussianBlur(frame,(7,7),0)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #      h    s    v
    llb = (0,   3,   0)
    ulb = (26,  255, 255)
    lb =  (120, 3,   0)
    ub =  (180, 255, 255)

    lowreds = cv2.inRange(hsv_frame, llb, ulb)
    highreds = cv2.inRange(hsv_frame, lb, ub)
    res = cv2.bitwise_or(lowreds, highreds)

    return res

def center_vertical_obs_bottom(red_frame, CLEAR_CENTER):
    """Return the number of pixels to the bottom of the 
    frame that the closest contour occupies"""
    lb, rb = int(640/2-CLEAR_CENTER), int(640/2+CLEAR_CENTER)
    red_frame = red_frame[:,lb:rb]
    contours, _ = cv2.findContours(red_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    large_contours = [cont for cont in contours if cv2.contourArea(cont) > MIN_CONTOUR_SIZE]
    bottom_y = []
    for cont in large_contours:
        _,y,_,h = cv2.boundingRect(cont)
        bottom_y.append(480-(y+h))
    if len(bottom_y) == 0:
        return 480 # max height of frame
    return min(bottom_y)

# https://www.delftstack.com/howto/python/moving-average-python/
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def rotate_to(scf, curr, current_angle, new_angle):
    """Steadily rotates the drone a given angle. Works best for 90 degrees"""
    cf = scf.cf

    pos_neg = np.sign(current_angle - new_angle)
    abs_ang_remaining = abs(current_angle - new_angle)
    if abs_ang_remaining > 30:
        steps = int(4/90*abs_ang_remaining) # found by testing
        for i in range(steps):
            cf.commander.send_hover_setpoint(0, 0, 1.5*pos_neg*90, curr[2])
            time.sleep(0.1)
        # print('First part of rotate')
        for yawr in np.linspace(pos_neg*90, 0, 5):
            cf.commander.send_hover_setpoint(0, 0, yawr, curr[2])
            time.sleep(0.1)
        cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
    # else:
        # print('Started within 30 of rotation set point so skipping first part of rotate')

    # print('Second part of yaw rate')
    for i in np.linspace(angle_estimate(scf), new_angle, 10):
        cf.commander.send_position_setpoint(curr[0], curr[1], curr[2], i)
        time.sleep(0.15)

    # print('end of rotation now hovering')
    for _ in range(10):
        cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
        time.sleep(0.1)
    return angle_estimate(scf)

def test_rotate(scf, curr, curr_angle):
    "right, zero, left, right"
    curr_angle = rotate_to(scf,curr,curr_angle, -90) # right is -90
    curr_angle = rotate_to(scf,curr,curr_angle, 0) # zero
    curr_angle = rotate_to(scf,curr,curr_angle, 90) # left is 90
    curr_angle = rotate_to(scf,curr,curr_angle, -90) # right is 90 ... should go smoothly turn through the centre
    time.sleep(20) # stop during here
    return curr_angle

def find_book(model, frame, confidence):
    """Finds x coordinate of person taped to book in frame"""
    image = frame
    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                             swapRB=True)
   
    # forward propagate image
    model.setInput(blob)
    detections = model.forward()
    image_height, image_width, _ = image.shape
    
    # select detections that match selected class label and confidence
    # assumes only 1 person detected
    for detection in detections[0, 0, :, :]:
        det_conf, det_class_id = detection[2], detection[1]
        if det_conf > confidence and det_class_id == 1:
            box_x = detection[3] * image_width
            box_width = detection[5] * image_width
            return box_x + box_width/2
    return -1

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

def green_filter(frame):
    """Filter frame for just the green ground"""
    IMWIDTH = 640
    GREENPXBUFFER = 40
    # Crop to the lower half, then to the center of that
    frame = cv2.GaussianBlur(frame,(7,7),0)[240:, (IMWIDTH//2)-GREENPXBUFFER:(IMWIDTH//2)+GREENPXBUFFER, :]
    # frame = cv2.fastNlMeansDenoisingColored(frame,None,h=10,hColor=10,templateWindowSize=3,searchWindowSize=11)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv lower and upper bounds of white (works pretty well but needs more testing)
    lb = (29, 75, 85)
    ub = (60, 255, 255)

    green = cv2.inRange(hsv_frame, lb, ub)
    return green

def px_green_from_top(green_filtered_frame):
    MIN_GREEN_CONTOUR_SIZE = 40.0
    contours, _ = cv2.findContours(green_filtered_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    large_contours = [cont for cont in contours if cv2.contourArea(cont) > MIN_GREEN_CONTOUR_SIZE]
    bottom_y = []
    for cont in large_contours:
        _,y,_,_ = cv2.boundingRect(cont)
        bottom_y.append(y)
    if len(bottom_y) == 0:
        return 0
    return min(bottom_y)

def slide_to_book(scf, curr, v, WIDTH, SAFETY, cap, model, confidence):
    """Slide until Roger is centered in the camera frame
    """
    BOOK_CLEAR_CENTER = 4 # Roger should be in the middle pixels, TODO: tune
    IMWIDTH = 640
    going_left = True
    cf = scf.cf
    dt = 0.15
    
    book_x = 0
    start_time , time_index = time.time(), 0
    # TODO: make this while True and only break when we find Roger?
    while (going_left and curr[1] < WIDTH-SAFETY/4) or (not going_left and curr[1] > SAFETY/4):
        # position update
        dy = v*dt*(-1, 1)[going_left]
        curr[1] += dy
        cf.commander.send_position_setpoint(curr[0], curr[1], curr[2], 0)
        time_index += 1

        # Now look for the book
        _, frame = time_averaged_frame(cap)
        book_x = find_book(model, frame, confidence)
        if ((IMWIDTH//2)-BOOK_CLEAR_CENTER < book_x < (IMWIDTH//2)+BOOK_CLEAR_CENTER):
            break

        # sus time.sleep()?
        while time.time() < dt*time_index+start_time:
            continue

        # update going_left if reaching a boundary
        if (going_left and curr[1] >= WIDTH-SAFETY) or (not going_left and curr[1] <= SAFETY):
            going_left = not going_left

    # stabilize
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, curr[2])
        time.sleep(0.1)

    return pos_estimate(scf)