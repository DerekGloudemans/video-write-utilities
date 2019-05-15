# Read in a video and use openCV utilities to convert to a series of images
# save each image as a file
# run yolo object detection on each saved file
# load new image
# write new image to new video using videoWriter
# delete images

import  cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from datetime import datetime 



if __name__ == "__main__":
    
    infile = "traffic1.mkv"
    outfile = "test_output1.avi"
    tempimfile = "tempim.jpg"
    
    # open VideoCapture object to covert videoFrames to image files
    cap = cv2.VideoCapture(infile)
    
    # Check if camera opened successfully
    if cap.isOpened():
        retval,im = cap.read()
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB) 
        #plt.imshow(im)
        #plt.title("Sample Image")
    
    else: 
      print("Error opening video stream or file")
      
    frame_counter = 0
    while retval: # VideoCapture.read() returns False if no frame (i.e. end of video)
        
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Operation escaped.")
            break
        
        # save current image as a file
        cv2.imwrite(tempimfile,im) 

        # Define the codec and create VideoWriter object
        frame_height = int(cap.get(4))
        frame_width = int(cap.get(3))

        out = cv2.VideoWriter(outfile,0, 30, (frame_width,frame_height))
        
        loadim = cv2.imread(tempimfile)
        
        os.system('/home/worklab/darknet/darknet detect /home/worklab/darknet/cfg/yolov3.cfg /home/worklab/darknet/yolov3.weights ' +tempimfile)

        
        out.write(loadim)
        
        # attempt to get next frame
        retval,im = cap.read()
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        
        frame_counter = frame_counter + 1
    
    # close VideoCapture and VideoWriter objects
    cap.release()
    out.release()
 
    # Closes all the frames
    cv2.destroyAllWindows()