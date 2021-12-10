""" The skeleton of the final controller design.
"""
import time
import numpy as np
from pynput import keyboard

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from helperfunctions import red_filter, time_averaged_frame
from helperfunctions import check_crazyflie_available, start_video, set_pid_controller, key_press, relative_move, land, takeoff, move_to_setpoint
from helperfunctions import time_averaged_frame, center_vertical_obs_bottom, rotate_to, find_book

# important constants
group_number = 12
camera_number = 0

# TODO: tune these constants
DEFAULT_VELOCITY = 0.8
DX = 0.25
DY = 0.2
SAFETY_DISTANCE_TO_SIDE = .18
SAFETY_DISTANCE_TO_END = 0.15 # reduce later when write whte line detect
SAFETY_PX_TO_OBJ = 60 #px
L_VS_R = 2 #px
BOOK_MARGIN_PX = 30
WIDTH = 1.32
LENGTH = 2.7 
CLEAR_CENTER = 100 #pixel column clear to end needed

# IDEAS
# while loop for moving forwards
# white lines for stopping 

# On turns: need to add time.sleep() to let it scan
# Never moves forwards???
## Turning back is too fast
# If no detections: take a second picture just to make sure

# Initialize the drone's position at the right-hand corner of the obstacle course\
curr = [0, 0, 0]
curr_angle = 0 # -90 = left, 90 = right
reached_table = False

if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # init drone 
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cf
        cap = start_video(camera_number)
        set_pid_controller(cf) # reset now that firmly on the ground

        with keyboard.Listener(on_press= lambda key: key_press(key, cf, cap, curr)) as listener:
            # Take off and move to the left
            curr = takeoff(cf, 0.4)

            # check at 20 positions dist_to_obs where don't have to check l_r as 'safe'
            dist_to_obs_center, steps = [], 8
            # step_size = max((WIDTH-curr[1])/steps, DY)
            for y_targ in np.linspace([1], WIDTH, steps):
                curr = relative_move(scf, curr, [0, DY, 0], DEFAULT_VELOCITY, False)

                _, frame = time_averaged_frame(cap)
                red = red_filter(frame) # super accomodating
                dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER) # splits frame in two as discussed
                dist_to_obs_center.append((dist_center_obs, curr)) 
            # find index with max distance
            max_dist_index = np.argmax([pos[0] for pos in dist_to_obs_center])
            curr = move_to_setpoint(scf, curr, dist_to_obs_center[max_dist_index][1], DEFAULT_VELOCITY, True)
            
            # aligned with furthest obstacle/ no obstacle
            while not reached_table:
                # find distance to closest obstacle
                _, frame = time_averaged_frame(cap)
                red = red_filter(frame) # super accomodating
                dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER)
                clear_in_front = dist_center_obs > SAFETY_PX_TO_OBJ

                if clear_in_front:
                    # find distance to closest obstacle again
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER)
                    clear_in_front = dist_center_obs > SAFETY_PX_TO_OBJ
                    if clear_in_front:
                        curr = relative_move(scf, curr, [DX, 0, 0], DEFAULT_VELOCITY, True)
                
                # peek left and right, determine which way is safe to move in
                # move to best position in that direction
                else:
                    print("Time to look around...")
                    # peek left
                    curr_angle = rotate_to(scf, curr, curr_angle, -90)
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    dist_left = center_vertical_obs_bottom(red, CLEAR_CENTER)

                    # return center, then peek right
                    curr_angle = rotate_to(scf, curr, curr_angle, 0)
                    curr_angle = rotate_to(scf, curr, curr_angle, 90)
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    dist_right = center_vertical_obs_bottom(red, CLEAR_CENTER)

                    # return cetner
                    curr_angle = rotate_to(scf, curr, curr_angle, 0)

                    # determine right or left as costly manoeuver ... if don't find anything go other way
                    go_right = True
                    # don't go right if super close to right edge
                    if curr[1] < SAFETY_DISTANCE_TO_SIDE:
                        go_right=False
                    # don't go right if significantly better to go left
                    if dist_left > dist_right + L_VS_R:
                        go_right=False
                    # don't go right if right is blocked ... duh
                    if dist_right <= SAFETY_PX_TO_OBJ:
                        go_right = False
                    
                    # inch forwards checking distances periodically
                    side_distance = (dist_left, dist_right)[go_right] # false = first, true = second
                    pos_neg = (1,-1)[go_right] # positive y is to the left, negative is to the right
                    dist_to_obs_center = []
                    print("We are going to move right, true or false: ", go_right)
                    # while we are in bounds and aren't too close to an obstacle move in direction indicated
                    while side_distance > SAFETY_PX_TO_OBJ and SAFETY_DISTANCE_TO_SIDE<curr[1]<WIDTH-SAFETY_DISTANCE_TO_SIDE:
                        curr = relative_move(scf, curr, [0, DY*pos_neg, 0], DEFAULT_VELOCITY, False)

                        # forwards
                        curr_angle = rotate_to(scf, curr, curr_angle, 0)
                        _, frame = time_averaged_frame(cap)
                        red = red_filter(frame) # super accomodating
                        dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER)
                        dist_to_obs_center.append((dist_center_obs, curr))

                        # update side_distance
                        curr_angle = rotate_to(scf, curr, curr_angle, 90*pos_neg)
                        _, frame = time_averaged_frame(cap)
                        red = red_filter(frame) # super accomodating
                        side_distance = center_vertical_obs_bottom(red, CLEAR_CENTER)

                        curr_angle = rotate_to(scf, curr, curr_angle, 0)

                    # weird bug where sometimes drifted into bad region between setting pos_neg and while loop
                    # makes dist_to_obs_center sometimes empty
                    if dist_to_obs_center:
                        # move to ideal position again and rotate forward
                        max_dist_index = np.argmax([pos[0] for pos in dist_to_obs_center]) 
                        curr = move_to_setpoint(scf, curr, dist_to_obs_center[max_dist_index][1], DEFAULT_VELOCITY, True)

                    cf.commander.send_position_setpoint(curr[0], curr[1], curr[2], 0)

                    # if none of the positions were good, will pick best position 
                    # fail again to move forward and will search in other direction
                    # as will be closer to one of the sides
            

                # TODO: update reached table with greenness or white lines on ground?
                reached_table = curr[0] > LENGTH - SAFETY_DISTANCE_TO_END # no obstacles in last 0.5m
                if reached_table: break
            
            # reached the end of the obstacles, so fly up to table height
            # TODO: tune the flight height
            curr = relative_move(scf, curr, [0,0,0.5], .1, True)
            
            

            # Centre book in frame 
            in_left_half = (curr[1] - WIDTH/2) > 0
            go_left = not in_left_half
            while True:
                ret, frame = time_averaged_frame(cap)
                # left of frame is 0 line
                # TODO: find_book code
                book_center_px, book_center_py = find_book(frame)
                if ret:
                    if np.norm(book_center_px-320) < BOOK_MARGIN_PX:
                        break # Success!!!
                    # in left half frame - move left
                    elif book_center_px-320 < 0:
                        curr = relative_move(scf, curr, [0, DY/2, 0], DEFAULT_VELOCITY, False)
                    else:
                        curr = relative_move(scf, curr, [0, -DY/2, 0], DEFAULT_VELOCITY, False)
                else:
                    if go_left and (curr[1] - WIDTH) > -SAFETY_DISTANCE_TO_SIDE:
                        go_left=False
                    if (not go_left) and curr[1] < SAFETY_DISTANCE_TO_SIDE:
                        go_left = True
                    curr = relative_move(scf, curr, [0, go_left*DY, 0], DEFAULT_VELOCITY, False)
            
            land(cf, curr)