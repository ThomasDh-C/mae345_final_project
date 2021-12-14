""" The skeleton of the final controller design.
"""
import time
import numpy as np
import cv2
from pynput import keyboard
import matplotlib.pyplot as plt


from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from helperfunctions import *

# -----------------------
# Brief Method:\
# 1. take off from very right corner
# 2. slide to the left taking diistances in the center of the frame 
#    --> smooth these to account for noise and align with best starting pos
# 3. if heads towards farthest obstacle, will forward slide and then stop in front of obstacle, 
#    if there is a gap then forward slides through the gap
# 4. if can't move forwards rotate towards the side (will alternate between sliding left and right for each obstacle) and slide until you hit an obstacle
# 5. Turn forwards, if no obstacles in front keep going forwards. 
#    If obstacel in front do inital move of sliding back to starting point (as know move is safe). Pick best new position
# 5. once reach end of the course, detect the end of the green mat and move forward until reached specified height of green mat's end
# 6. rise upwards and go to the far right of the table.
# 7. begin book detection, slide from right to left to look for person on a book within a certain frame.
# 8. move left and land if detected book is on left side of frame, land if in center, move right and land if detected book is on right side of frame
# ---------------------

# important constants
group_number = 12
camera_number = 0 # 1 for Thomas, 0 for Jacob
tracking_label = 1              # person COCO dataset
confidence = 0.3                # confidence of detection

# TODO: tune these constants
DEFAULT_VELOCITY = 0.8
BIG_DX = 0.4
SMALL_DX = 0.2
DX = 0.25
GREEN_DX = 0.01
BIG_DY = 0.4
SMALL_DY = 0.2
DY = 0.2
VERY_CLEAR_PX = 55 # tuned from 55
SAFETY_PX_TO_OBJ = 30 # tuned from 38

SAFETY_DISTANCE_TO_SIDE = .3
SAFETY_DISTANCE_TO_END = 0.05 # reduce later when write whte line detect
L_VS_R = 2 #px
BOOK_MARGIN_PX = 20
WIDTH = 1.32
LENGTH = 2.75 # tuned up from 2.7 since consistently undershoots 
CLEAR_CENTER = 58 # pixel column clear to end needed, tuned from 58
CLEAR_CENTER_LR = 20 # pixels
GREEN_MARGIN = 5
GREEN_PX_TOP_BOT_IDEAL = 105 # tune up to get closer, down to get further, jacob ideal = 110, thomas ideal = 105

# load the DNN model
model = cv2.dnn.readNet(model='Lab9_Supplement/frozen_inference_graph.pb',
                        config='Lab9_Supplement/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

# TODO: (saturday)
# slide left/right
# implement video feed
# Look at ending book detection function


# IDEAS
# while loop for moving forwards
# white lines for stopping 

# On turns: need to add time.sleep() to let it scan
# Never moves forwards???
## Turning back is too fast
# If no detections: take a second picture just to make sure

# Initialize the drone's position at the right-hand corner of the obstacle course\
curr = [0, 0, 0]
curr_angle = 0 # -90=left  90=right
reached_kalman_end = False

if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # --- Init drone ---
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cf
        cap = start_video(camera_number)
        set_pid_controller(cf) # reset now that firmly on the ground

        # --- Take off ---
        curr = takeoff(scf, 0.35)
        # reset length and widthbased on starting point
        # TODO: does this actually do anything since LENGTH and WIDTH are constants?
        LENGTH+=curr[0]
        WIDTH += curr[1]

        # --- Find best starting y (align with furthest/ no obstacle) ---
        # Don't have to check l_r as 'safe' starting zone
        curr = take_off_slide_left(scf, curr, WIDTH, DEFAULT_VELOCITY*.15, DEFAULT_VELOCITY*.75, cap, CLEAR_CENTER_LR)
        obstacles_avoided = 0
        # --- Move forward, move round obstacles, reach kalman end ---
        print(f"Well positioned at x={curr[1]} and ready to move forward")
        has_checked_left = False
        has_checked_right = False
        while not reached_kalman_end:
            # -- Find distance to closest obstacle 5 times for reliability --
            obj_distance = []
            for i in range(3):            
                _, frame = time_averaged_frame(cap)
                red = red_filter(frame) # super accomodating
                dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER)
                obj_distance.append(dist_center_obs)
            print("pixels to closest center object: ", np.mean(obj_distance))

            # -- Make a big jump if obstacle is far/ non-existent --
            if np.mean(obj_distance) > VERY_CLEAR_PX:
                print("\tSo, making a forward slide")
                curr = forward_slide_to_obs(scf, curr, DEFAULT_VELOCITY*.2, VERY_CLEAR_PX, LENGTH - SAFETY_DISTANCE_TO_END, CLEAR_CENTER, cap)
            # -- Make a small jump if obstacle is far/ non-existent --
            elif np.mean(obj_distance) > SAFETY_PX_TO_OBJ:
                print("\tSo, making a small jump")
                curr = relative_move(scf, curr, [SMALL_DX, 0, 0], DEFAULT_VELOCITY, True)
            
            # -- Obstacle close in front - choose new x position, whilst not crashing --
            # peek left and right, determine which way is safe to move in
            # move to best position in that direction
            else:
                obstacles_avoided+=1
                print("Time to look around...")
                curr = pos_estimate(scf)

                if has_checked_left and has_checked_right:
                    has_checked_left, has_checked_right = False, False
                    print("We have to go back - back to the future!")
                    curr = relative_move(scf, curr, [-0.2, 0, 0], DEFAULT_VELOCITY*.4, True)
                
                # - go right unless: in right half or -
                go_right = True
                if curr[1] - WIDTH/2 > 0:
                    go_right = False
                if has_checked_left:
                    go_right = True
                if has_checked_right:
                    go_right = False

                
                # set flags
                if go_right:
                    has_checked_right = True
                else:
                    has_checked_left = True

                # - 1. rotate in direciton of slide
                curr_angle = rotate_to(scf, curr, curr_angle, (1,-1)[go_right]*90)
                print('Should have turned to right=-90 or left=90, current angle is',curr_angle)

                # - 2. Slide in the safe direction until hit something or hit sides
                curr = pos_estimate(scf)
                start_pos = [curr[0], curr[1], curr[2]] # don't ask me why this is needed but it is
                print('About to slide to the ', ('left', 'right')[go_right])
                curr = left_right_slide_to_obs(scf, curr, DEFAULT_VELOCITY*.2, VERY_CLEAR_PX, WIDTH, SAFETY_DISTANCE_TO_SIDE, CLEAR_CENTER+5, cap, go_right)
                print('Finished sliding')
                end_pos = curr

                # - 3. Rotate back to straight
                curr_angle = rotate_to(scf, curr, curr_angle, 0) 

                # - 4. Slide to new starting point like our takeoff move and choose best new point ... unless distance currently is 480 ... in which case yeet forwards
                _, frame = time_averaged_frame(cap)
                red = red_filter(frame) # super accomodating
                dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER) # splits frame in two as discussed
                print('Distance on extreme of move:', dist_center_obs)
                if dist_center_obs < 150:
                    curr = left_right_slide_to_start_point(scf, end_pos, start_pos, DEFAULT_VELOCITY*.4, DEFAULT_VELOCITY*.5, cap, CLEAR_CENTER_LR, 50)
                

            # -- Check if have reached course end for while loop --
            reached_kalman_end = curr[0] > LENGTH - SAFETY_DISTANCE_TO_END # no obstacles in last 0.5m

        print("Made it to the end of the course, moving to the right side of the course")
        # move to the right side of the course for consistent green measurement
        curr = move_to_setpoint(scf, curr, [curr[0], SAFETY_DISTANCE_TO_SIDE*1.5, curr[2]], DEFAULT_VELOCITY*0.2, True)
        # curr = relative_move(scf, curr, [0, -curr[1] + SAFETY_DISTANCE_TO_SIDE/4, 0], DEFAULT_VELOCITY*.2, True)

        
        # --- fine tune x using green turf ---
        print("Dialing in on the green...")
        curr = slide_green(scf, curr, cap, DEFAULT_VELOCITY/3, GREEN_PX_TOP_BOT_IDEAL, GREEN_MARGIN, GREEN_DX)

        print("Got the green")
        curr = move_to_setpoint(scf, curr, [curr[0], -SAFETY_DISTANCE_TO_SIDE, curr[2]], DEFAULT_VELOCITY*0.2, True)

        # --- end of the obstacles - up to table height ----
        curr = relative_move(scf, curr, [0,0,0.5], .1, True)
        
        # move to the book
        curr = slide_to_book(scf, curr, DEFAULT_VELOCITY*0.1, WIDTH, SAFETY_DISTANCE_TO_SIDE, cap, model, confidence)

        land(cf, curr)
