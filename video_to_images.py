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
        if frame_num > 20: # early video cutoff
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


video_dir = "/home/worklab/Data/cv/AIC21_Track1_Vehicle_Counting_full/AIC21_Track1_Vehicle_Counting/Dataset_A"
#out_directory= "/home/worklab/Desktop/frames"

directory = '/home/worklab/Desktop/frames'
sequences = [os.path.join(video_dir,item) for item in os.listdir(video_dir)]
for video_file in sequences:
        out_directory = os.path.join(directory,video_file.split("/")[-1].split(".")[0])
        video_to_images(video_file,out_directory,surname = video_file.split("/")[-1])      