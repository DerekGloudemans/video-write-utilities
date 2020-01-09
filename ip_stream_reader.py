import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from datetime import datetime

# Next, try to convert this .mov file into a .mpg file and save it
# Create a VideoCapture object
cap = cv2.VideoCapture("/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_16/cam_1_capture_002.avi")
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('temp.avi',cv2.CAP_FFMPEG,0, 30, (frame_width,frame_height))
last = 0

while(True):
  ret, frame = cap.read()
 
  if ret == True: 
     
    # Write the frame into the file 'output.avi'
    out.write(frame)
 
    # Display the resulting frame    
    cv2.imshow('frame',frame)
    
    
    print(cap.get(0)-last)
    last = cap.get(0)
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else:
    break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 

