import cv2
import time

def start_video(camera_number):
    """Returns camera stream and removes junk inital frames"""
    cap = cv2.VideoCapture(camera_number)
    
    # Wait until video has started
    while not cap.isOpened():
        time.sleep(.1)

    # Make sure no junk frames
    time.sleep(3)
    return cap

def fake_red(frame):
    # do fake procesing
    cv2.imshow('frame', frame)
    return frame
# fake check so looks like program
if True:
    # don't have to worry about with stuff as doesn't create a new thread
    cap = start_video(0)

    while True:
        print('start')
        ret, frame = cap.read()
        frame = fake_red(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print('fake loop')
        for i in range(1000): 
            
            time.sleep(.1)
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # time.sleep(100)

        print('end')
        ret, frame = cap.read()
        frame = fake_red(frame)