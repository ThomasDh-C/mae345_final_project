""" The skeleton of the final controller design.
"""
import time
import numpy as np
from pynput import keyboard

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from helperfunctions import check_crazyflie_available, start_video, set_pid_controller, key_press, relative_move, land, takeoff

# from find_white import get_starting_y

# important constants

group_number = 12
camera_number = 0

# TODO: tune these constants
DEFAULT_VELOCITY = 0.3
DX = 0.1
SAFETY_DISTANCE = 0.1
BOOK_LOWER_MARGIN_PX = 320-30
BOOK_UPPER_MARGIN_PX = 320+30

# Initialize the drone's position at the right-hand corner of the obstacle course
# TODO: replace boolean flags with expressions based on actual measurements
curr = [0, 0, 0]
reached_table = False
shit_in_front = True
multiple_objects_in_sight = True
distance_to_closest_ob = 0
book_center_px = 0




if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # init drone 
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cf
        cap = start_video(camera_number)
        set_pid_controller(cf) # reset now that firmly on the ground

        with keyboard.Listener(on_press= lambda key: key_press(key, cf, cap, curr)) as listener:
            
            # Take off and move to the left in order to get initial optical flow readings
            curr = takeoff(cf, 0.4)
            curr = relative_move(scf, curr, [0, 0.1, 0], DEFAULT_VELOCITY)
            # TODO: insert optical flow code here
            # TODO: constantly in the loop update distance_to_closest

            while not reached_table:
                if not shit_in_front:
                    curr = relative_move(scf, curr, [DX, 0, 0], DEFAULT_VELOCITY)
                    continue
                elif shit_in_front and multiple_objects_in_sight:
                    # get optical flow and move accordingly
                    # TODO: replace [0, 0, 0] with calculated direction
                    curr = relative_move(scf, curr, [0, 0, 0], DEFAULT_VELOCITY)
                    continue
                elif shit_in_front and not multiple_objects_in_sight:
                    # move to the left if too close to nearest object
                    if distance_to_closest_ob <= SAFETY_DISTANCE:
                        curr = relative_move(scf, curr, [0, DX, 0], DEFAULT_VELOCITY)
                        continue
                    else:
                        curr = relative_move(scf, curr, [DX, 0, 0], DEFAULT_VELOCITY)
                        continue
            
            # reached the end of the obstacles, so fly up to table height
            # TODO: tune the flight height
            curr = relative_move(scf, curr, [0,0,0.75], .1)
            
            # Keep going until we land by the book!
            while True:
                # TODO: update book_center_px
                if book_center_px > BOOK_LOWER_MARGIN_PX and book_center_px < BOOK_UPPER_MARGIN_PX:
                    curr = relative_move(scf, curr, [DX, 0, 0], DEFAULT_VELOCITY)
                    land(cf, curr)
                    break # Success!!!
                elif book_center_px < BOOK_LOWER_MARGIN_PX:
                    curr = relative_move(scf, curr, [0, DX, 0], DEFAULT_VELOCITY)
                    continue
                elif book_center_px > BOOK_UPPER_MARGIN_PX:
                    curr = relative_move(scf, curr, [0, -DX, 0], DEFAULT_VELOCITY)
                    continue
                # IDEA: for this last case we can have it move right until it reaches y = 0, then reverse
                # the direction of motion so it scans back to the left???
                else:
                    curr = relative_move(scf, curr, [0, -DX, 0], DEFAULT_VELOCITY)