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
from helperfunctions import check_crazyflie_available, start_video, set_pid_controller, key_press, detection_center, detect_book, move_to_book

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
tracking_label = 84             # book COCO dataset
confidence = 0.2                # confidence of detection

if check_crazyflie_available():
    with SyncCrazyflie(f'radio://0/{group_number}/2M', cf=Crazyflie(rw_cache='./cache')) as scf:
        # init drone 
        # scf = sync_crazyflie obj, cf = crazyflie obj
        cf = scf.cf
        cap = start_video(camera_number)
        set_pid_controller(cf) # reset now that firmly on the ground

        # Initialize and ascend:
        t = time.time()
        elapsed = time.time() - t
        ascended_bool = 0

        # start at table height (75) + 25 cm
        cf_command = PositionHlCommander(cf, default_velocity=.2, default_height=0.75)
        
        with Listener(on_press= lambda key: key_press(key, cf_command, cap)) as listener:
            listener.join() # listen for command q being pressed without while loop ... 

            # get the video frames' width and height
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            # current position
            x_cur = 0
            y_cur = 0
            
            # flag indicating whether to exit the main loop and then descend
            exit_loop = False

            # fly fly away
            cf_command.take_off()

            while cap.isOpened() and not exit_loop:
                # Try to read image
                # ret, frame = cap.read()
                ret, frame = cv2.imread('imgs/85.png')
                if ret:
                    det = detect_book(model, frame, tracking_label, confidence) 
                    if det is not None:
                        # get the class id
                        class_id = det[1]
                        # map the class id to the class 
                        class_name = class_names[int(class_id)-1]
                        color = COLORS[int(class_id)]
                        # get the bounding box coordinates
                        box_x = det[3] * image_width
                        box_y = det[4] * image_height
                        # get the bounding box width and height
                        box_width = det[5] * image_width
                        box_height = det[6] * image_height
                        # draw a rectangle around each detected object
                        cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                        # put the class name text on the detected object
                        cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # If nothing is detected, hover
                    if det is None:
                        print('no detection...hovering')
                        hover(cf)

                    # otherwise  move towards target
                    else:
                        print('detection...tracking')
                        _, _, _, box_x, box_y, box_width, box_height = det
                        box_x, box_y = detection_center(det)
                        exit_loop, x_cur, y_cur = move_to_book(cf, box_x, box_y, box_width, box_height, x_cur, y_cur)

                    # Check image
                    cv2.imshow('image', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                    
                else:
                    print('no image!!')
                
            cap.release()
        
            # Descend and stop all motion:
            hover_and_descend(cf)
        
            cv2.destroyAllWindows()
            print("Touchdown")
else: 
    print("DorEye down mayday")