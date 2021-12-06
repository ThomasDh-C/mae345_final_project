import time
from pynput import keyboard

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from helperfunctions import check_crazyflie_available, start_video, set_pid_controller, key_press

group_number = 12
camera_number = 0

if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # init drone 
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cf
        cap = start_video(camera_number)
        set_pid_controller(cf) # reset now that firmly on the ground
        cf_command = PositionHlCommander(cf, default_velocity=.2, default_height=.25)
        
        with keyboard.Listener(on_press= lambda key: key_press(key, cf_command, cap)) as listener:
            listener.join() # listen for command q being pressed without while loop ... 
        
            # fly fly away
            cf_command.take_off()
            cf_command.move_distance(2.59,0,0) # relative move to table
            time.sleep(3)
            cf_command.move_distance(0,0,0.75) # relative ascend
            time.sleep(3)
            cf_command.move_distance(0.6,0,0) # relative over table
            time.sleep(3)
            cf_command.land()
    print("Touchdown")
else: 
    print("DorEye down mayday")
