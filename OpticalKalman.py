import numpy as np
import cv2
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

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
        f.x = np.array([[float(x)],
                        [float(y)],
                        [0.],
                        [0.]]) # set the state vector, with velocities being 0
        print(p0)
        pFlag = True

if __name__ == '__main__':

    # Set up Kalman Filter
    f = KalmanFilter(dim_x=4,dim_z=2)
    ## State Transition
    f.F = np.array([[1.,0.,100.,0.],
                    [0.,1.,0.,1.],
                    [0.,0.,1.,0.],
                    [0.,0.,0.,1.]])
    # State to Measurement Matrix
    f.H = np.array([[1.,0.,0.,0.],
                    [0.,1.,0.,0.]])
    # State Covariance Matrix
    f.P = np.array([[1000.,0.,0.,0.],
                    [0.,1000.,0.,0.],
                    [0.,0.,1000.,0.],
                    [0.,0.,0.,1000.]])
    # Measurement Covariance Matrix
    f.R = np.array([[10.,0.],
                    [0.,10.]])
    # Process Noise
    f.Q = 0 #Q_discrete_white_noise(dim=4, dt=1, var=0.000000000001)
    # Start Video Camera
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
            # Now run our measurment (p1) through Kalman Filter
            z = np.array(p1[st==1],dtype=float) # we use st==1 to only track points that have been found
            print("This is z ",z," shape is ",z.shape)
            f.predict()
            f.update(z)
            # Select good points for display
            if p1 is not None:
                good_new = f.x[:2].reshape(1,-1) # f.x has x,y,vx,vy. only store the first two elements
                #assert len(good_new) == 2
                print("Good new is ",good_new)
                good_old = p0.ravel().reshape(1,-1) # just get the two values from p0
                #assert len(good_old) == 2
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                print("New is ",new)
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = np.array(good_new,dtype=np.float32).reshape(len(good_new), 1, 2)
            print("p0 at the end is ",p0," the shape is ",p0.shape)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()