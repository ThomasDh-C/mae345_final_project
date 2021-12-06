import cv2
import time
# CrazyFlie imports:
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.positioning.motion_commander import MotionCommander
import numpy as np
from helperfunctions import detection_center, norm, closest_detection, detect_book, set_pid_controller, key_press

# load the COCO class names
with open('Lab9_Supplement/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')
    
# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
model = cv2.dnn.readNet(model='Lab9_Supplement/frozen_inference_graph.pb',
                        config='Lab9_Supplement/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

group_number = 12
camera_number = 0
tracking_label = 84             # book
confidence = 0.2                # confidence of detection


if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # init drone 
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cf
        cap = start_video(camera_number)
        set_pid_controller(cf) # reset now that firmly on the ground
        # start at table height (75) + 25 cm
        cf_command = PositionHlCommander(cf, default_velocity=.2, default_height=1)
        
        with Listener(on_press= lambda key: key_press(key, cf_command, cap)) as listener:
            listener.join() # listen for command q being pressed without while loop ... 
        
            # fly fly away
            cf_command.take_off()
            cf_command.move_distance(2.59,0,0) # relative move to table
            

    print("Touchdown")
else: 
    print("DorEye down mayday")
