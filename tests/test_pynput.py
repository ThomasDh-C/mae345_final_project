from pynput import keyboard

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

import time

# Collect events until released
with keyboard.Listener(on_press= lambda key: on_press(key)) as listener:
    # listener.start() # listen for command q being pressed without while loop ... 
    time.sleep(3)
    print('hi')