import time
from pynput import keyboard

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from controller2.helperfunctions import move_to_setpoint
from helperfunctions import check_crazyflie_available, start_video, set_pid_controller, key_press, relative_move, land

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
            curr = relative_move(cf, curr, [2.59,0,0], .2)
            curr = relative_move(cf, curr, [0,0,0.75], .2)
            curr = relative_move(cf, curr, [0.6,0,0], .2)
            curr = relative_move(cf, curr, [0.6,0,0], .2)
            land(cf, curr)

    print("Touchdown")
else: 
    print("DorEye down mayday")
