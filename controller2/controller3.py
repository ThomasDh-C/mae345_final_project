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
VERY_CLEAR_PX = 57 # tuned from 55
SAFETY_PX_TO_OBJ = 38 # tuned from 38

SAFETY_DISTANCE_TO_SIDE = .3
SAFETY_DISTANCE_TO_END = 0.05 # reduce later when write whte line detect
L_VS_R = 2 #px
BOOK_MARGIN_PX = 20
WIDTH = 1.32
LENGTH = 2.75 # tuned up from 2.7 since consistently undershoots 
CLEAR_CENTER = 70 # pixel column clear to end needed, tuned from 60
CLEAR_CENTER_LR = 20 # pixels
GREEN_MARGIN = 5
GREEN_PX_TOP_BOT_IDEAL = 95

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
            if np.mean(obj_distance) > VERY_CLEAR_PX:
                print("\tSo, making a forward slide")
                curr = forward_slide_to_obs(scf, curr, DEFAULT_VELOCITY*.2, VERY_CLEAR_PX, LENGTH - SAFETY_DISTANCE_TO_END, CLEAR_CENTER, cap)
                # curr = relative_move(scf, curr, [BIG_DX, 0, 0], DEFAULT_VELOCITY, True)
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
                
                # - (if possible will go right) peek right -
                dist_right = 0
                if not curr[1] < SAFETY_DISTANCE_TO_END:
                    print("Peeking right")
                    dists_list = []
                    curr_angle = rotate_to(scf, curr, curr_angle, -90)
                    for i in range(3):
                        _, frame = time_averaged_frame(cap)
                        red = red_filter(frame) # super accomodating
                        # cv2.imwrite(f'imgs/pk_r_{obstacles_avoided}_{i}_.png', frame)
                        # cv2.imwrite(f'imgs/pk_r_{obstacles_avoided}_{i}_r.png', red)
                        dist = center_vertical_obs_bottom(red, CLEAR_CENTER)
                        dists_list.append(dist)
                    dist_right = np.mean(dists_list)
                    print("Dist_right: ", dist_right)
                    
                # - (if possible will go left) return center, then peek left -
                dist_left = 0
                if not curr[1] > WIDTH - SAFETY_DISTANCE_TO_SIDE:
                    print("Peeking left")
                    curr_angle = rotate_to(scf, curr, curr_angle, 90)
                    dists_list = []
                    for i in range(3):
                        _, frame = time_averaged_frame(cap)
                        red = red_filter(frame) # super accomodating
                        # cv2.imwrite(f'imgs/pk_l_{obstacles_avoided}_{i}.png', frame)
                        # cv2.imwrite(f'imgs/pk_l_{obstacles_avoided}_{i}_r.png', red)
                        dist = center_vertical_obs_bottom(red, CLEAR_CENTER)
                        dists_list.append(dist)
                    dist_left = np.mean(dists_list)
                    print("Dist_left: ", dist_left)
                    

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
                print(f'Decided to go:', ('left', 'right')[go_right])
                
                # - Avoid obstacle by laterally moving -
                # skip dist check if d is huge 
                if (go_right and dist_right==480) or (not go_right and dist_left==480):
                    start_pos = curr
                    end_y = (WIDTH, 0)[go_right]
                    end_pos = [curr[0],end_y,curr[2]]
                    curr = left_right_slide_to_start_point(scf, start_pos, end_pos, DEFAULT_VELOCITY*.2, DEFAULT_VELOCITY*.5, cap, CLEAR_CENTER_LR, 50)
                else:
                    # - 1. rotate in direction of obstacle -
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
                    if dist_center_obs < 300:
                        curr = left_right_slide_to_start_point(scf, end_pos, start_pos, DEFAULT_VELOCITY*.2, DEFAULT_VELOCITY*.5, cap, CLEAR_CENTER_LR, 50)
                

            # -- Check if have reached course end for while loop --
            reached_kalman_end = curr[0] > LENGTH - SAFETY_DISTANCE_TO_END # no obstacles in last 0.5m

        print("Made it to the end of the course, moving to the right side of the course")
        # move to the right side of the course for consistent green measurement
        curr = relative_move(scf, curr, [0, -curr[1] + SAFETY_DISTANCE_TO_SIDE/4, 0], DEFAULT_VELOCITY*.2, True)

        # --- fine tune x using green turf ---
        print("Dialing in on the green...")
        curr = slide_green(scf, curr, cap, DEFAULT_VELOCITY/3, GREEN_PX_TOP_BOT_IDEAL, GREEN_MARGIN, GREEN_DX)
        # _, frame = time_averaged_frame(cap)
        # green = green_filter(frame) # super accomodating
        # # green_from_top = px_green_from_top(green)
        # x, green_from_top, w, h = cv2.boundingRect(green)
        # green_list = [green_from_top]*3 # moving average
        # c = 0
        # print(c, ' green from top', green_from_top)
        # while abs(np.mean(green_list)-GREEN_PX_TOP_BOT_IDEAL) > GREEN_MARGIN:
        #     c+=1
        #     print("\tGreen from top: ", np.mean(green_list))
        #     forwards = (np.mean(green_list)-GREEN_PX_TOP_BOT_IDEAL) < 0
        #     curr = relative_move(scf, curr, [forwards*GREEN_DX, 0, 0], DEFAULT_VELOCITY/3, True)
            
        #     _, frame = time_averaged_frame(cap)
        #     green = green_filter(frame) # super accomodating
        #     x, green_from_top, w, h = cv2.boundingRect(green)
        #     # green_from_top = px_green_from_top(green)
        #     green_list.append(green_from_top)
        #     green_list.pop(0)
        #     print(c, ' green from top', np.mean(green_list))


        # --- end of the obstacles - up to table height ----
        curr = relative_move(scf, curr, [0,0,0.5], .1, True)
        
        # move to the book
        curr = slide_to_book(scf, curr, DEFAULT_VELOCITY*0.2, WIDTH, SAFETY_DISTANCE_TO_SIDE, cap, model, confidence)

        # 12/12, 8:47 commented out code below
        # --- centre book in frame ---
        # in_left_half = (curr[1] - WIDTH/2) > 0
        # go_left = not in_left_half
        # while True:
        #     ret, frame = time_averaged_frame(cap)
        #     # left of frame is 0 line
        #     book_center_px = find_book(model, frame, confidence)
        #     if ret and book_center_px != -1:
        #         if np.linalg.norm(book_center_px-320) < BOOK_MARGIN_PX:
        #             break # Success!!!
        #         # in left half frame - move left
        #         elif book_center_px-320 < 0:
        #             curr = relative_move(scf, curr, [0, DY/2, 0], DEFAULT_VELOCITY, False)
        #         else:
        #             curr = relative_move(scf, curr, [0, -DY/2, 0], DEFAULT_VELOCITY, False)
        #     else:
        #         if go_left and (curr[1] - WIDTH) > -SAFETY_DISTANCE_TO_SIDE:
        #             go_left=False
        #         if (not go_left) and curr[1] < SAFETY_DISTANCE_TO_SIDE:
        #             go_left = True
        #         curr = relative_move(scf, curr, [0, go_left*DY, 0], DEFAULT_VELOCITY, False)

        land(cf, curr)
