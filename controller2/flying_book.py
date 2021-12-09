import cv2
from pynput import keyboard
import time
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import numpy as np
from helperfunctions import time_averaged_frame
from helperfunctions import start_video, detect_book, check_crazyflie_available, key_press

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
tracking_label = 10             # traffic light COCO dataset
confidence = 0.45               # confidence of detection

if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # init drone 
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cf
        cap = start_video(camera_number)

        ###
        while True:
            ret, frame = cap.read()
            if ret: 
                print('data type required: ', frame.dtype)
                break


        curr = [0,0,0]
        
        with keyboard.Listener(on_press= lambda key: key_press(key, cf, cap, curr)) as listener:
                    
                # Capture frame-by-frame
                while True:
                    # ret, frame = cap.read()
                    ret, frame = time_averaged_frame(cap)
                    if ret:
                        blurred = cv2.GaussianBlur(frame, (3,3), 0)
                        detect_book(model, blurred, confidence, COLORS, class_names)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        print("Touchdown")
else: 
    print("DorEye down mayday")