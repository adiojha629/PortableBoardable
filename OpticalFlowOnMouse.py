from ctypes.wintypes import RGB
from enum import auto
import numpy as np
import cv2
import matplotlib as plt
# based on https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
p0 = None
pFlag = False
lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Colors for tracking the edges
color = np.random.randint(0, 255, (100, 3))

# First create capture the click of the mouse
def click_event(event,x,y,flags,params):
    '''
    Sets p0 based on the x,y location of the click
    '''
    if event == cv2.EVENT_RBUTTONDOWN:
        # display x,y for debugging
        print(x," ",y)
        # set p0
        global p0, pFlag
        p0 = np.array([[x,y]]).reshape((-1,1,2)).astype(np.float32)
        print(p0)
        # [x,y],
        #                 [x+10,y],
        #                 [x-10,y],
        #                 [x,y+10],
        pFlag = True
buffer = []

def auto_offset(img,x,y):
    #window = img[x-7:x+7][y-7:y+7]      #Make 15x15 window
    thresh = 50  # 100 is ideal thresh for Red
    w_size = 2
    i_max = -1
    j_max = -1
    d_max = -1
    for i in range(x-w_size,x+w_size+1):
        for j in range(y-w_size,y+w_size+1):
            red_component = img[j][i][1]    # [0] = B, [1] = G, [2] = R
            if red_component > thresh:
                d = np.sqrt((x-i)**2 + (y-j)**2)
                if d > d_max and (i,j) not in buffer:
                    i_max = i
                    j_max = j
                    d_max = d
    # done searching
    if i_max == -1:
        print("Did not find red component. Please change threshold or windowsize")
        # once threshold is set, we'll use this branch to remove the line
    else:
        buffer.append((i_max,j_max))
        return i_max,j_max
#def final_offset(img):

if __name__ == '__main__':
    video_file_path = "Videos/chris_ghost_green.mp4"
    cap = cv2.VideoCapture(video_file_path)

    # first loop: just show the first frame and wait for mouse click
    for i in range(10):
        ret, frame = cap.read()
        frame = cv2.resize(frame,(750,500))
    got_click = False
    while not pFlag:
        cv2.imshow('ClickOnPoint',frame) # display the frame
        cv2.setMouseCallback('ClickOnPoint',click_event) # check for clicks
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    x = int(p0[0][0][0])
    y = int(p0[0][0][1])
    print(frame[x][y])
    # colby = input("Press enter to continue program")
    mask = None
    marker = None # has our lines that will be drawn on the whiteboard
    offsetx = 0
    offsety = 0

    whiteboard = np.zeros((500,750,3),dtype=np.uint8) # variable holds our whiteboard

    maskFlag = False
    while(1):
        cv2.imshow("Whiteboard",whiteboard)
        ret, frame = cap.read()
        if ret: 
            frame = cv2.resize(frame,(750,500))
            #frame = cv2.flip(frame,1)
            if not ret: # check that we have a video
                print('No frames grabbed!')
                break
            # set up mask for drawing over image
            if not maskFlag:
                # Create a mask image for drawing purposes
                mask = np.zeros_like(frame)
                marker = np.zeros_like(frame)
                old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                maskFlag = True
            # check if we have a point to follow
            if not pFlag:
                cv2.imshow('ClickOnPoint',frame)
                cv2.setMouseCallback('ClickOnPoint',click_event)
            else: # do optical flow!
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    #a,b = auto_offset(frame,int(a),int(b))
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a-offsetx), int(b-offsety)), (int(c-offsetx), int(d-offsety)), (255,0,0), 2) # blue line
                    marker = cv2.line(marker, (int(a-offsetx), int(b-offsety)), (int(c-offsetx), int(d-offsety)), (255,255,255), 2)
                    frame = cv2.circle(frame, (int(a-offsetx), int(b-offsety)), 5, (255,0,0), -1)
                print(whiteboard.dtype)
                print(mask.dtype)
                #adi = input()
                whiteboard = cv2.add(whiteboard, marker)
                img = cv2.add(frame, mask)
                cv2.imshow('frame', img)
                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                #path.append(p0)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()