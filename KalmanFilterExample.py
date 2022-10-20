# Just an example of how the kalman filter works
# it does not use and webcamera data at all
# adapted from this tutorial: https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

# Libraries
from filterpy.kalman import KalmanFilter
import numpy as np

# Code

def main():
    # Main Method
    f = KalmanFilter(dim_x=2,dim_z=1) # we have two inputs: position and velocity
    # and one measurement of position (so dim_z=1)
    f.x = np.array([[2.],    # position
                [0.]])   # velocity
    f.F = np.array([[1.,1.],
                [0.,1.]]) # the state transition matrix
    # x_new = F * x_old. So our new position = 1*old_position + 1*velocity
    # and our new velocity is the same as our old velocity

    f.H = np.array([[1.,0.]]) # Measurement function
    # idk what this is

    f.P = np.array([[1000.,    0.],
                [   0., 1000.] ]) # covariance matrix
    # represents the uncertainty in our measurements and estimates

    f.R = 5 # measurement noise
    

if __name__ == '__main__':
    main()