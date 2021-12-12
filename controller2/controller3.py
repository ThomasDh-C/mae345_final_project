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

# important constants
group_number = 12
camera_number = 0
tracking_label = 1              # person COCO dataset
confidence = 0.3                # confidence of detection

# TODO: tune these constants
DEFAULT_VELOCITY = 0.8
BIG_DX = 0.4
SMALL_DX = 0.2
DX = 0.25
GREEN_DX = 0.05
BIG_DY = 0.4
SMALL_DY = 0.2
DY = 0.2
VERY_CLEAR_PX = 50 # TODO: tuned down from 135
SAFETY_PX_TO_OBJ = 30 # TODO: tuned down from 60

SAFETY_DISTANCE_TO_SIDE = .18
SAFETY_DISTANCE_TO_END = 0.15 # reduce later when write whte line detect
L_VS_R = 2 #px
BOOK_MARGIN_PX = 20
WIDTH = 1.32
LENGTH = 2.7 
CLEAR_CENTER = 40 # pixel column clear to end needed

GREEN_MARGIN = 5 # TODO: tune these
GREEN_PX_TOP_BOT_IDEAL = 95 # TODO: tune these

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
        curr = takeoff(cf, 0.35)

        # --- Find best starting y (align with furthest/ no obstacle) ---
        # Don't have to check l_r as 'safe' starting zone
        curr = take_off_slide_left(scf, curr, WIDTH-SAFETY_DISTANCE_TO_SIDE, DEFAULT_VELOCITY/3, cap, CLEAR_CENTER)
        
        # --- Move forward, move round obstacles, reach kalman end ---
        print(f"Well positioned at x={curr[1]} and ready to move forward")
        while not reached_kalman_end:
            # -- Find distance to closest obstacle 5 times for reliability --
            obj_distance = []
            for i in range(5):            
                _, frame = time_averaged_frame(cap)
                red = red_filter(frame) # super accomodating
                dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER)
                obj_distance.append(dist_center_obs)
            print("pixels to closest center object: ", np.mean(obj_distance))

            # -- Make a big jump if obstacle is far/ non-existent --
            if sum([dist >= VERY_CLEAR_PX for dist in obj_distance])>3:
                print("\tSo, making a forward slide")
                curr = forward_slide_to_obs(scf, curr, DEFAULT_VELOCITY*.5, VERY_CLEAR_PX, LENGTH - SAFETY_DISTANCE_TO_END, CLEAR_CENTER, cap)
                # curr = relative_move(scf, curr, [BIG_DX, 0, 0], DEFAULT_VELOCITY, True)
            # -- Make a small jump if obstacle is far/ non-existent --
            elif sum([dist >= SAFETY_PX_TO_OBJ for dist in obj_distance])>3:
                print("\tSo, making a small jump")
                curr = relative_move(scf, curr, [SMALL_DX, 0, 0], DEFAULT_VELOCITY, True)
            
            # -- Obstacle close in front - choose new x position, whilst not crashing --
            # peek left and right, determine which way is safe to move in
            # move to best position in that direction
            else:
                print("Time to look around...")
                curr = pos_estimate(scf)
                
                # - (if possible will go right) peek right -
                dist_right = 0
                if not curr[1] < SAFETY_DISTANCE_TO_END:
                    print("Peeking right")
                    curr_angle = rotate_to(scf, curr, curr_angle, -90)
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    cv2.imwrite('peek_right.png', frame)
                    cv2.imwrite('peek_right_red.png', red)
                    dist_right = center_vertical_obs_bottom(red, CLEAR_CENTER)
                    print("Dist_right: ", dist_right)
                    curr_angle = rotate_to(scf, curr, curr_angle, 0)
                
                # - (if possible will go left) return center, then peek left -
                dist_left = 0
                if not curr[1] > WIDTH - SAFETY_DISTANCE_TO_SIDE:
                    print("Peeking left")
                    curr_angle = rotate_to(scf, curr, curr_angle, 90)
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    cv2.imwrite('peek_left.png', frame)
                    cv2.imwrite('peek_left_red.png', frame)
                    dist_left = center_vertical_obs_bottom(red, CLEAR_CENTER)
                    print("Dist_left: ", dist_left)


                # - return center -
                curr_angle = rotate_to(scf, curr, curr_angle, 0)

                # - based on measurements determine right or left -
                curr = pos_estimate(scf)
                go_right = True
                # don't go right if super close to right edge               
                if curr[1] < SAFETY_DISTANCE_TO_SIDE:
                    go_right=False
                # don't go right if significantly better to go left
                elif dist_left > dist_right + L_VS_R:
                    go_right=False
                # don't go right if right is blocked ... duh
                elif dist_right <= SAFETY_PX_TO_OBJ:
                    go_right = False
                
                # - Avoid obstacle by laterally moving -
                # - 1. Set up direction constants -
                side_distance = (dist_left, dist_right)[go_right] # false=first_idx  true=second_idx
                pos_neg = (1,-1)[go_right] # positive y is to the left, negative is to the right
                dist_to_obs_center = []
                print("We are going to move right, true or false: ", go_right)

                # - 2. Initially Use big sideways moves if obj to our side is far away -
                extra_side_check = True
                if go_right and SAFETY_DISTANCE_TO_SIDE < curr[1]:
                    extra_side_check = False
                if (not go_right) and curr[1] < WIDTH - SAFETY_DISTANCE_TO_SIDE:
                    extra_side_check = False
                while side_distance >= VERY_CLEAR_PX and extra_side_check:
                    curr = relative_move(scf, curr, [0, BIG_DY*pos_neg, 0], DEFAULT_VELOCITY, False)
                    # forwards
                    # curr_angle = rotate_to(scf, curr, curr_angle, 0) # already facing forward
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER)
                    # save battery by breaking out if it is clear ahead
                    if dist_center_obs >= VERY_CLEAR_PX:
                        # set side_distance high so we skip over next while loop
                        side_distance = 480
                        break
                    dist_to_obs_center.append((dist_center_obs, curr))

                    # update side_distance
                    curr_angle = rotate_to(scf, curr, curr_angle, 90*pos_neg)
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    side_distance = center_vertical_obs_bottom(red, CLEAR_CENTER)

                    curr_angle = rotate_to(scf, curr, curr_angle, 0)
                
                # - 3. Then use little sideways moves to be safer afterward -
                while SAFETY_PX_TO_OBJ < side_distance < VERY_CLEAR_PX and extra_side_check:
                    curr = relative_move(scf, curr, [0, SMALL_DY*pos_neg, 0], DEFAULT_VELOCITY, False)
                    # forwards
                    # curr_angle = rotate_to(scf, curr, curr_angle, 0) # already facing forward
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER)
                    # save battery by breaking out if it is clear ahead
                    if dist_center_obs >= VERY_CLEAR_PX:
                        break
                    dist_to_obs_center.append((dist_center_obs, curr))

                    # update side_distance
                    curr_angle = rotate_to(scf, curr, curr_angle, 90*pos_neg)
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    side_distance = center_vertical_obs_bottom(red, CLEAR_CENTER)

                    curr_angle = rotate_to(scf, curr, curr_angle, 0)
                
                # TODO: @Jacob check we can ignore this bug and that this code handles it ... 
                # seems like if dist_to_obs_center is empty we just don't do the move ... is this expected func?
                # - 4. weird bug where sometimes drifted into bad region between setting pos_neg and while loop
                # makes dist_to_obs_center sometimes empty
                if dist_to_obs_center:
                    # move to ideal position again and rotate forward
                    max_dist_index = np.argmax([pos[0] for pos in dist_to_obs_center]) 
                    curr = move_to_setpoint(scf, curr, dist_to_obs_center[max_dist_index][1], DEFAULT_VELOCITY, True)
                # else ...? see above todo
                cf.commander.send_position_setpoint(curr[0], curr[1], curr[2], 0)

                # - Why we don't check in the other direction here: -
                # if none of the positions were good, will pick best position 
                # then fail again to move forward and will search in other direction
                # as will be closer to the other side to the start of this conditional
            
            # -- Check if have reached course end for while loop --
            reached_kalman_end = curr[0] > LENGTH - SAFETY_DISTANCE_TO_END # no obstacles in last 0.5m
            # if reached_kalman_end: break # TODO: think we can remove this - unnecessary as a while loop!?
        
        # --- fine tune x using green turf ---
        _, frame = time_averaged_frame(cap)
        green = green_filter(frame) # super accomodating
        green_from_top = px_green_from_top(green)
        # x, green_from_top, w, h = cv2.boundingRect(green)
        c = 0
        print(c, ' green from top', green_from_top)
        while np.linalg.norm(green_from_top-GREEN_PX_TOP_BOT_IDEAL) > GREEN_MARGIN:
            c+=1
            print("\tGreen from top: ", green_from_top)
            forwards = (green_from_top-GREEN_PX_TOP_BOT_IDEAL) < 0
            curr = relative_move(scf, curr, [forwards*GREEN_DX, 0, 0], DEFAULT_VELOCITY/3, True)
            _, frame = time_averaged_frame(cap)
            green = green_filter(frame) # super accomodating
            # x, green_from_top, w, h = cv2.boundingRect(green)
            green_from_top = px_green_from_top(green)
            print(c, ' green from top', green_from_top)


        # --- end of the obstacles - up to table height ----
        curr = relative_move(scf, curr, [0,0,0.5], .1, True)
        

        # --- centre book in frame ---
        in_left_half = (curr[1] - WIDTH/2) > 0
        go_left = not in_left_half
        while True:
            ret, frame = time_averaged_frame(cap)
            # left of frame is 0 line
            book_center_px = find_book(model, frame, confidence)
            if ret and book_center_px != -1:
                if np.linalg.norm(book_center_px-320) < BOOK_MARGIN_PX:
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
