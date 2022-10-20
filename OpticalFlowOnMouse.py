import numpy as np
import cv2
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
        p0 = np.array([[x,y],
                        [x+10,y],
                        [x-10,y],
                        [x,y+10],
                        [x,y-10]]).reshape((-1,1,2)).astype(np.float32)
        print(p0)
        pFlag = True

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    mask = None
    maskFlag = False
    while(1):
        ret, frame = cap.read()
        #frame = cv2.flip(frame,1)
        if not ret: # check that we have a video
            print('No frames grabbed!')
            break

        # set up mask for drawing over image
        if not maskFlag:
            # Create a mask image for drawing purposes
            mask = np.zeros_like(frame)
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
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()