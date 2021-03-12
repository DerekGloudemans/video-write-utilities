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
def video_to_images(video_file,out_directory = "temp",surname = ""):
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
            result = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            result.save(out_directory + "/{}.png".format(str(frame_num).zfill(5)))
        frame_num += 1
        start = time.time()
        # get next frame
        ret, frame = cap.read()
        if frame_num > 2000: # early video cutoff
            cap.release()
            break

class ImDirectoryReader():
    def __init__(self,path):
        self.path = path
        self.frame = 0
    
    def read(self):
        try:
            im = cv2.imread(self.path+ "/{}.bmp".format(self.frame))
            self.frame += 1
            return True, im
        except:
            return False, None
        
    def read_frame(self,frame_num):
        try:
            return True, cv2.imread(self.path+ "/{}.bmp".format(frame_num))
        except:
            return False, None
  
    def isOpened(self):
        return True

video_file = '/media/worklab/data_HDD/cv_data/video/traffic_assorted/traffic_0.avi'
out_directory = '/media/worklab/data_HDD/cv_data/video/traffic_assorted/traffic_0_frames'

video_file = '/media/worklab/data_HDD/cv_data/video/1-week-test/Camera_15/cam_5_capture_000.avi'
out_directory='/home/worklab/Desktop/SCOPE/camera_15'


video_file = "/home/worklab/Documents/CV-detection/pipeline_files/track0.avi"
out_directory= "/home/worklab/Documents/CV-detection/pipeline_files"

# directory = '/home/worklab/Desktop/temp'
# sequences = [os.path.join(directory,item) for item in os.listdir(directory)]
# for video_file in sequences:
#         out_directory = "temp"
#         video_to_images(video_file,out_directory,surname = video_file.split("/")[-1])      