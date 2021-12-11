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
BIG_DY = 0.4
SMALL_DY = 0.2
DY = 0.2
VERY_CLEAR_PX = 180 # TODO: tune
SAFETY_PX_TO_OBJ = 80 # TODO: tune

SAFETY_DISTANCE_TO_SIDE = .18
SAFETY_DISTANCE_TO_END = 0.15 # reduce later when write whte line detect
L_VS_R = 2 #px
BOOK_MARGIN_PX = 30
WIDTH = 1.32
LENGTH = 2.7 
CLEAR_CENTER = 50 #pixel column clear to end needed

# load the DNN model
model = cv2.dnn.readNet(model='Lab9_Supplement/frozen_inference_graph.pb',
                        config='Lab9_Supplement/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

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

        # with keyboard.Listener(on_press= lambda key: key_press(key, cf, cap, curr)) as listener:
        #create two subplots
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        
        #create two image plots
        _, frame = time_averaged_frame(cap)
        im1 = ax1.imshow(frame)
        im2 = ax2.imshow(frame)
        plt.ion()

        # Take off and move to the left
        curr = takeoff(cf, 0.4)

        # check at 8 positions dist_to_obs where don't have to check l_r as 'safe'
        curr = take_off_slide_left(scf, curr, WIDTH, DEFAULT_VELOCITY/3, cap, CLEAR_CENTER)
        
        # aligned with furthest obstacle/ no obstacle
        while not reached_table:
            print("top of main while loop")
            obj_distance = []
            # find distance to closest obstacle 5 times
            for i in range(5):            
                _, frame = time_averaged_frame(cap)
                im1.set_data(frame)
                red = red_filter(frame) # super accomodating
                dist_center_obs = center_vertical_obs_bottom(red, CLEAR_CENTER)
                obj_distance.append(dist_center_obs)

            dist_center_obs = np.mean(obj_distance)
            print("pixels to closest center object: ", dist_center_obs)

            if sum([dist >= VERY_CLEAR_PX for dist in obj_distance])>3:
                print("\tSo, making a big jump")
                curr = relative_move(scf, curr, [BIG_DX, 0, 0], DEFAULT_VELOCITY, True)
            
            elif sum([dist >= SAFETY_PX_TO_OBJ for dist in obj_distance])>3:
                print("\tSo, making a small jump")
                curr = relative_move(scf, curr, [SMALL_DX, 0, 0], DEFAULT_VELOCITY, True)
            
            # peek left and right, determine which way is safe to move in
            # move to best position in that direction
            else:
                curr = pos_estimate(scf)

                print("Time to look around...")
                
                # peek right
                dist_right = 0
                if not curr[1] < SAFETY_DISTANCE_TO_END:
                    print("Peeking right")
                    curr_angle = rotate_to(scf, curr, curr_angle, -90)
                    _, frame = time_averaged_frame(cap)
                    im2.set_data(frame)
                    red = red_filter(frame) # super accomodating
                    dist_right = center_vertical_obs_bottom(red, CLEAR_CENTER)
                    print("Dist_right: ", dist_right)
                    curr_angle = rotate_to(scf, curr, curr_angle, 0)
                
                
                # return center, then peek left
                dist_left = 0
                if not curr[1] > WIDTH - SAFETY_DISTANCE_TO_SIDE:
                    print("Peeking left")
                    curr_angle = rotate_to(scf, curr, curr_angle, 90)
                    _, frame = time_averaged_frame(cap)
                    red = red_filter(frame) # super accomodating
                    dist_left = center_vertical_obs_bottom(red, CLEAR_CENTER)
                    print("Dist_left: ", dist_left)


                # return cetner
                curr_angle = rotate_to(scf, curr, curr_angle, 0)

                # determine right or left as costly manoeuver ... if don't find anything go other way
                go_right = True
                # don't go right if super close to right edge
                curr = pos_estimate(scf)
                if curr[1] < SAFETY_DISTANCE_TO_SIDE:
                    go_right=False
                # don't go right if significantly better to go left
                elif dist_left > dist_right + L_VS_R:
                    go_right=False
                # don't go right if right is blocked ... duh
                elif dist_right <= SAFETY_PX_TO_OBJ:
                    go_right = False
                
                # inch forwards checking distances periodically
                side_distance = (dist_left, dist_right)[go_right] # false = first, true = second
                pos_neg = (1,-1)[go_right] # positive y is to the left, negative is to the right
                dist_to_obs_center = []
                print("We are going to move right, true or false: ", go_right)

                # use big sideways moves to get us places fast
                while side_distance >= VERY_CLEAR_PX and SAFETY_DISTANCE_TO_SIDE < curr[1] < WIDTH - SAFETY_DISTANCE_TO_SIDE:
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
                
                # use little sideways moves to be safer afterward
                while SAFETY_PX_TO_OBJ < side_distance < VERY_CLEAR_PX and SAFETY_DISTANCE_TO_SIDE < curr[1] < WIDTH - SAFETY_DISTANCE_TO_SIDE:
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
            book_center_px = find_book(model, frame, confidence)
            if ret:
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

        plt.ioff() # due to infinite loop, this gets never called.
        plt.show()