"""
Concatenates a list of videos or all videos from a directory into a single video
"""

import cv2
import time
import os

frame_width  = 3840
frame_height = 2160
vid_dir  = "/media/worklab/data_HDD/cv_data/video/1-week-test/Camera_16"
out_file = os.path.join(vid_dir,"combined.avi")


vid_list = [os.path.join(vid_dir,vid) for vid in os.listdir(vid_dir)]
vid_list.sort()


out = cv2.VideoWriter(out_file,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
count = 0
for vid in vid_list:
    print("Processing file: {}".format(vid))
    start_time = time.time()
    cap = cv2.VideoCapture(vid)
                   
    # check if camera opened successfully
    assert cap.isOpened(), "Unable to read video {}".format(vid)
    
    # read all frames
    while(True):
      ret, frame = cap.read()
     
      if ret == True: 
          count += 1
          
          out.write(frame)
          
          if count % 200 == 0:
              hour = count//(30*60*60)
              minute = count%(30*60*60)//(30*60)
              second = count%(30*60)//(30)
              print("Video processed: {}:{}:{}. FPS: {}.".format(hour,minute,second,count/(time.time()-start_time)))

      else:
          cap.release()
          break 

out.release()