import cv2
import time
from os import mkdir
import os
from PIL import Image 
import numpy as np

"""
makes a directory with each frame of the input video file as a separate image. 
Useful to get rid of the overhead of video decoding and encoding during detection etc.
"""
def video_to_images(video_file,avg_frame,out_directory = "temp",surname = ""):
    
    avg = cv2.imread(avg_frame)
    prev_diff = None
    
    # open up a videocapture object
    cap = cv2.VideoCapture(video_file)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)
    
    #create directory for new images
    try:  
        mkdir(out_directory)
    except OSError:  
        print ("Creation of the directory %s failed" % out_directory)
    else:  
        print ("Successfully created the directory %s " % out_directory)
    
    start = time.time()
    frame_num = 0

    
    # get first frame
    ret, frame = cap.read()
    
    while ret:  

        #result = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        #result.save(out_directory + "/{}.png".format(frame_num))
        if frame_num % 1 == 0:
            print("On frame: {}, FPS: {:5.2f}".format(frame_num, 1.0 / (time.time() - start)))
            
            frame = cv2.resize(frame,(1920,1080))
            # frame = np.abs(frame-avg)
            
            # if prev_diff is not None:
            #     frame  -= prev_diff
            #     frame = np.clip(frame,0,255)
            # prev_diff = frame.copy()
            
            
            # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # frame = cv2.blur(frame,ksize = (10,10))
            

                
            # frame = cv2.threshold(frame,200,255,cv2.THRESH_BINARY)[1]
            # frame = cv2.blur(frame,ksize = (3,3))
            # frame = cv2.threshold(frame,200,255,cv2.THRESH_BINARY)[1]

            
            #frame = cv2.threshold()
            
            cv2.imwrite("{}/{}.png".format(out_directory,frame_num),frame)
            
        frame_num += 1
        start = time.time()
        # get next frame
        ret, frame = cap.read()
        if frame_num > 20: # early video cutoff
            cap.release()
            break






directory = '/home/worklab/Desktop/frames'
video_file = "/home/worklab/Desktop/p1c2.mp4"
avg_frame = "/home/worklab/Desktop/p1c2_avg.png"
out_directory = os.path.join(directory)
video_to_images(video_file,avg_frame,out_directory=out_directory,surname = "")      