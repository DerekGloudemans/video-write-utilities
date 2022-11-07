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
    
    clip = 50
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    
    second_height = int((cap2.get(4)-2*clip)*1920/720) 
    
    frame_height += second_height
        
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*"MPEG"), 30, (frame_width,frame_height))
    
   
        
    # read first frame from all captures
    frames = []
    #i = 0

    ret1,frame1 = cap1.read()
    ret2,frame2 = cap2.read()
    frame_num = 0
   
        
    while ret1 and ret2:
        frame_num += 1        
        print("\r Frame {}".format(frame_num),end = "\r",flush = True)

        
        frame2 = frame2[clip:-clip,:,:]
        frame2 = cv2.resize(frame2,(1920,second_height))
        # frame2 = cv2.resize(frame2,(1920*2//3,1080*2//3))
        # ymin = 80*4//3
        # ymax = 280*4//3
        # xmin = 100*4//3
        # xmax = 500*4//3
        # xw = xmax - xmin
        # yw = ymax - ymin
        # frame1[0:yw,0:xw,:] = frame2[ymin:ymax,xmin:xmax,:]
        
        
        new_frame = np.concatenate((frame1,frame2),axis = 0)
    
        if file_out != None:
            out.write(new_frame)
          
        if False:
            cv2.imshow("frame",new_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        
    
        ret1,frame1 = cap1.read()
        ret2,frame2 = cap2.read()
        
        
for camera in ["p1c3"]: #"p1c1","p1c2","p1c3","p1c4",
    m = "/home/worklab/Documents/derek/i24-dataset-gen/camera_{}_track_outputs_3D_rectified.mp4".format(camera)
    inset = "/home/worklab/Documents/derek/i24-dataset-gen/{}_color2.mp4".format(camera)
    inset_vid(m,inset,file_out = "{}_inset.mp4".format(camera))