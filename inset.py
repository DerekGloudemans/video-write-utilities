import cv2
import numpy as np
import time

def inset_vid(main_vid,inset_vid,file_out = "video.avi"):
    
    # open capture devices to read video files
    
    cap1 = cv2.VideoCapture(main_vid)
    assert cap1.isOpened(), "Cannot open file \"{}\"".format(main_vid)
        
    cap2 = cv2.VideoCapture(inset_vid)
    assert cap2.isOpened(), "Cannot open file \"{}\"".format(inset_vid)
        
    # determine size of combined frame (assumed that all frames are the same size)
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    
        
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*"MPEG"), 30, (frame_width,frame_height))
    
   
        
    # read first frame from all captures
    frames = []
    #i = 0

    ret1,frame1 = cap1.read()
    ret2,frame2 = cap2.read()
   
   
        
    while ret1 and ret2:
        
        frame2 = cv2.resize(frame2,(1920*2//3,1080*2//3))
        ymin = 80*4//3
        ymax = 280*4//3
        xmin = 100*4//3
        xmax = 500*4//3
        xw = xmax - xmin
        yw = ymax - ymin
        frame1[0:yw,0:xw,:] = frame2[ymin:ymax,xmin:xmax,:]
    
        if file_out != None:
            out.write(frame1)
            
        cv2.imshow("frame",frame1)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        
    
        ret1,frame1 = cap1.read()
        ret2,frame2 = cap2.read()
        
        
      
m = "/home/worklab/Documents/derek/track_i24/output/p1c4_00002_labeled_d2.avi"
inset = "/home/worklab/Documents/derek/track_i24/output/trajectories_p1c4_00002.avi"
inset_vid(m,inset)