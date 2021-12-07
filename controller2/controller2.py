import time
from pynput import keyboard

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from helperfunctions import check_crazyflie_available, start_video, set_pid_controller, key_press, relative_move, land, takeoff

group_number = 12
camera_number = 0

if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # init drone 
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cf
        cap = start_video(camera_number)
        set_pid_controller(cf) # reset now that firmly on the ground

        curr = [0,0,0]
        
        with keyboard.Listener(on_press= lambda key: key_press(key, cf, cap, curr)) as listener:
            # fly fly away
            curr = takeoff(cf, .4)
            ####
            print("takeoff successful")
            print("starting relative move...")
            ####
            curr = relative_move(cf, curr, [2.59,0,0], .3)
            ####
            print("done with first relative move")
            print("Position after first relative move: ", curr)
            print("starting second relative move...")
            ####
            curr = relative_move(cf, curr, [0,0,0.75], .1)
            print("starting third relative move...")
            curr = relative_move(cf, curr, [0.6,0,0], .1)
            print("starting fourth relative move...")
            curr = relative_move(cf, curr, [0.6,0,0], .1)
            land(cf, curr)

    print("Touchdown")
else: 
    print("DorEye down mayday")
