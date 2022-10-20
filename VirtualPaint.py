# import cv2
# import numpy as np

# #Idea: use webcame and color detection trackbars to track when colors are present
# #Then we put a point on the screen whenever we see such colors


# #first find the HSV params for each color of marker
# #then, use the params to make masks that only light up if that color is present there
# #in those areas, add the corressponding color (through bit masking) and add this image to the original image

# # Color Masks:
# # Red
# shape_screen = (400,400,3)
# lower_r = np.array([0,0,0])
# upper_r = np.array([16,255,255])
# red = np.full(shape_screen,np.array([0,0,255]),dtype=np.uint8)
# # Green
# lower_g = np.array([61,137,0])
# upper_g = np.array([88,255,255])
# green = np.full(shape_screen,np.array([0,255,0]),dtype=np.uint8)
# # Purple
# lower_p = np.array([123,64,31])
# upper_p = np.array([163,221,255])
# purple = np.full(shape_screen,np.array([255,0,255]),dtype=np.uint8)
# # mask = cv2.inRange(imgHSV,lower,upper) # set to get purple color
# red_marker = np.zeros(shape_screen,dtype=np.uint8)
# gr_marker = np.zeros(shape_screen,dtype=np.uint8)
# pur_marker = np.zeros(shape_screen,dtype=np.uint8)

# cap = cv2.VideoCapture(0) # 0 for default webcam
# cap.set(3,640)# width is id 3, set to 640
# cap.set(4,480)# heigh is id 4, set to 480
# cap.set(10,100)# brightness at 100
# while True:
#     success, img = cap.read() # get frame in img var
#     img = cv2.resize(img, (400, 400)) # since screen is small
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # transform to different color style
#     #Make color masks
#     mask_r = cv2.inRange(imgHSV,lower_r,upper_r)
#     mask_g = cv2.inRange(imgHSV,lower_g,upper_g)
#     mask_p = cv2.inRange(imgHSV,lower_p,upper_p)

#     #Apply mask to colors
#     red_marker = cv2.add(red_marker,cv2.bitwise_and(red,red,mask=mask_r))
#     gr_marker = cv2.add(gr_marker,cv2.bitwise_and(green, green, mask=mask_g))
#     pur_marker = cv2.add(pur_marker,cv2.bitwise_and(purple, purple, mask=mask_p))

#     # print(img.dtype,red_marker.dtype)
#     # adi = input()

#     marker_image = cv2.add(cv2.add(img,gr_marker),pur_marker)
#     # allview = np.hstack([marker_image,mask_p])
#     cv2.imshow("Video",mask_p)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break #if q pressed, video stops
#     if cv2.waitKey(1) & 0xFF == ord('e'):
#         red_marker = np.zeros(shape_screen, dtype=np.uint8)
#         gr_marker = np.zeros(shape_screen, dtype=np.uint8)
#         pur_marker = np.zeros(shape_screen, dtype=np.uint8)
#         #if q pressed, erase green markers

import cv2
import numpy as np
# https://www.youtube.com/watch?v=WQeoO7MI0Bs&t=521s

def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",123,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",163,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",89,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",221,255,empty)
cv2.createTrackbar("Value Min","TrackBars",57,255,empty)
cv2.createTrackbar("Value Max","TrackBars",198,255,empty)
cap = cv2.VideoCapture(0) # 0 for default webcam
cap.set(3,640)# width is id 3, set to 640
cap.set(4,480)# heigh is id 4, set to 480
cap.set(10,100)# brightness at 100
while True:
    # img = cv2.imread("Resources/lena.png")
    success, img = cap.read()  # get frame in img var
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # if q pressed, video stops
    img = cv2.resize(img,(400,400))
    #detect orange color
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")# name of bar, name of window its on
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Value Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Value Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)

    #create mask for filtering
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper) # set to get purple color
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    # HSV is https://en.wikipedia.org/wiki/HSL_and_HSV
    # Not an intuitive model
    cv2.putText(img, "Original", (25, 25), cv2.FONT_ITALIC, 1, (0, 0, 0), 2)
    cv2.putText(imgResult,"Purple Colors",(25,25),cv2.FONT_ITALIC,1,(255,0,255),2)
    imghor = np.hstack((img, imgResult))
    cv2.imshow("Original",img)
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Mask",mask)
    cv2.imshow("Result",imghor)

    cv2.waitKey(1)