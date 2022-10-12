# Author: Adi Ojha
# Purpose: Use the code here to get access to the computer's video camera
# Do not change the code here unless there is a bug!
# Copy and paste this code into whatever script you are writing.

import cv2

cam = cv2.VideoCapture(0) # the zero indicates web camera
# if you change 0 to the path to a video, that video will be captured
while True:
    check, frame = cam.read()
    cv2.imshow('video', frame) # make a window, with the name 'video'
    # display the image in this window

    key = cv2.waitKey(1)
    # key=-1 if no key has been pressed
    # else its the ascii value of the key pressed
    if key == 27: # close if 'ESC' is pressed.
        break
cam.release() # releases the video, similar to closing a file after reading it
cv2.destroyAllWindows() # destroy the video window.