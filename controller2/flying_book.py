import cv2
from pynput import keyboard
import time
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import numpy as np
from helperfunctions import time_averaged_frame
from helperfunctions import start_video, check_crazyflie_available

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
camera_number = 1
tracking_label = 10             # traffic light COCO dataset
confidence = 0.45               # confidence of detection

def detect_book(model, blurred, confidence, COLORS, class_names):
    """Detect all books in the frame"""
    image = blurred

    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                             swapRB=True)
   
    # forward propagate image
    model.setInput(blob)
    detections = model.forward()
    image_height, image_width, _ = image.shape
    
    # select detections that match selected class label
    first = True
    for detection in detections[0, 0, :, :]:
        if detection[2] > confidence:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            class_name = class_names[int(class_id)-1]
            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            # draw a rectangle around each detected object
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=1)
            # put the FPS text on top of the frame
            text = class_name + ' ' + '%.2f' % (detection[2])
            cv2.putText(image, text, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
            book_x = box_x + box_width/2
            if first:
                first = False
                print('book_x dist from center', abs(book_x-640/2))

    cv2.imshow('image', image)

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
    
        # Capture frame-by-frame
        while True:
            # ret, frame = cap.read()
            # ret, frame = time_averaged_frame(cap)
            ret, frame = cv2.imread('imgs/85.png')
            if ret:
                blurred = cv2.GaussianBlur(frame, (3,3), 0)
                detect_book(model, blurred, confidence, COLORS, class_names)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        print("Touchdown")
else: 
    print("DorEye down mayday")