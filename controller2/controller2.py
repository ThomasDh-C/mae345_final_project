import time
from pynput import keyboard

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from helperfunctions import check_crazyflie_available, start_video, set_pid_controller, key_press, relative_move, land, takeoff, log_config_setup
from cflib.crazyflie.log import LogConfig
from find_white import get_starting_y

group_number = 12
camera_number = 0

if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # init drone 
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cfq
        cap = start_video(camera_number)
        set_pid_controller(cf) # reset now that firmly on the ground

        curr = [0,0,0]
        
        with keyboard.Listener(on_press= lambda key: key_press(key, cf, cap, curr)) as listener:
            log_config = log_config_setup()


            # fly fly away
            curr = takeoff(cf, .4)
            ####
            print("takeoff successful")
            print("Current Position: ", curr)
            # ret, frame = cap.read()
            # print("I think my y-coordinate is: ", get_starting_y(frame))
            print("starting relative move...")
            ####
            curr = relative_move(scf, curr, [2.59,0,0], .3, log_config)
            ####
            print("done with first relative move")
            print("Position after first relative move: ", curr)
            print("starting second relative move...")
            ####
            curr = relative_move(scf, curr, [0,0,0.75], .1, log_config)
            print("starting third relative move...")
            curr = relative_move(scf, curr, [0.6,0,0], .3, log_config)
            # print("starting fourth relative move...")
            # curr = relative_move(scf, curr, [0.6,0,0], .1, log_config)
            land(cf, curr)

    print("Touchdown")
else: 
    print("DorEye down mayday")
